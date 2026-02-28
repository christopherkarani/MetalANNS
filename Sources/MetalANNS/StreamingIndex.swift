import Foundation
import MetalANNSCore

/// A continuous-ingest index that uses a two-level merge architecture.
///
/// New vectors land in a small `delta` index. When the delta reaches
/// `StreamingConfiguration.deltaCapacity` it is merged into the frozen
/// `base` index asynchronously (or synchronously for `.blocking` strategy).
/// Search always probes both shards and also covers pre-build pending inserts.
public actor StreamingIndex {
    private enum MetadataValue: Sendable, Codable {
        case string(String)
        case float(Float)
        case int64(Int64)
    }

    private struct PersistedMeta: Sendable, Codable {
        let config: StreamingConfiguration
        let vectorDimension: Int?
        let allVectorsList: [[Float]]
        let allIDsList: [String]
        let deletedIDs: [String]
        let metadataByID: [String: [String: MetadataValue]]
    }

    private var base: ANNSIndex?
    private var delta: ANNSIndex?
    private var mergeTask: Task<Void, Error>?
    private var lastBackgroundMergeError: ANNSError?
    private var _isMerging = false

    private var pendingVectors: [[Float]] = []
    private var pendingIDs: [String] = []

    private var allVectorsList: [[Float]] = []
    private var allIDsList: [String] = []
    private var allIDs: Set<String> = []
    private var deletedIDs: Set<String> = []

    private var idInBase: Set<String> = []
    private var idInDelta: Set<String> = []
    private var metadataByID: [String: [String: MetadataValue]] = [:]
    private var vectorDimension: Int?

    private let config: StreamingConfiguration
    /// Optional metrics sink for streaming operations.
    /// When set, the same instance is propagated to child base/delta indexes so
    /// delegated searches/inserts are recorded exactly once at the child level.
    public var metrics: IndexMetrics? = nil {
        didSet { metricsNeedsPropagation = true }
    }
    private var metricsNeedsPropagation = false

    public init(config: StreamingConfiguration = .default) {
        self.config = config
    }

    public var count: Int {
        allIDsList.count - deletedIDs.count
    }

    public var isMerging: Bool {
        _isMerging
    }

    public func setMetrics(_ metrics: IndexMetrics?) {
        self.metrics = metrics
    }

    public func insert(_ vector: [Float], id: String) async throws {
        try checkBackgroundMergeError()
        guard !allIDs.contains(id) else {
            throw ANNSError.idAlreadyExists(id)
        }
        try validateDimension(of: vector)
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        allIDs.insert(id)
        allIDsList.append(id)
        allVectorsList.append(vector)
        pendingIDs.append(id)
        pendingVectors.append(vector)

        try await flushPendingIntoDelta()
        try await maybeTriggerMerge()
    }

    public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws {
        try checkBackgroundMergeError()
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }

        var seen = Set<String>()
        for id in ids {
            if !seen.insert(id).inserted || allIDs.contains(id) {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        for vector in vectors {
            try validateDimension(of: vector)
        }
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        for id in ids {
            allIDs.insert(id)
            allIDsList.append(id)
        }
        allVectorsList.append(contentsOf: vectors)
        pendingIDs.append(contentsOf: ids)
        pendingVectors.append(contentsOf: vectors)

        try await flushPendingIntoDelta()
        try await maybeTriggerMerge()
    }

    public func search(
        query: [Float],
        k: Int,
        filter: SearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        try checkBackgroundMergeError()
        try validateQueryDimension(query)
        guard k > 0 else {
            return []
        }
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        let searchMetric = metric ?? config.indexConfiguration.metric
        var combined: [SearchResult] = []
        combined.reserveCapacity(k * 2)

        if let base {
            let baseResults = try await base.search(query: query, k: k, filter: filter, metric: metric)
            combined.append(contentsOf: baseResults)
        }

        if let delta {
            let deltaResults = try await delta.search(query: query, k: k, filter: filter, metric: metric)
            combined.append(contentsOf: deltaResults)
        }

        combined.append(contentsOf: pendingSearchResults(query: query, filter: filter, metric: searchMetric))
        return Array(dedupeAndSort(combined).prefix(k))
    }

    public func batchSearch(
        queries: [[Float]],
        k: Int,
        filter: SearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [[SearchResult]] {
        guard !queries.isEmpty else {
            return []
        }

        return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
            for (index, query) in queries.enumerated() {
                group.addTask { [self] in
                    let results = try await self.search(query: query, k: k, filter: filter, metric: metric)
                    return (index, results)
                }
            }

            var ordered = Array<[SearchResult]?>(repeating: nil, count: queries.count)
            for try await (index, results) in group {
                ordered[index] = results
            }
            return ordered.map { $0 ?? [] }
        }
    }

    public func rangeSearch(
        query: [Float],
        maxDistance: Float,
        limit: Int = 1000,
        filter: SearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        try checkBackgroundMergeError()
        try validateQueryDimension(query)
        guard maxDistance > 0 else {
            return []
        }
        guard limit > 0 else {
            return []
        }
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        let searchMetric = metric ?? config.indexConfiguration.metric
        var combined: [SearchResult] = []

        if let base {
            let baseResults = try await base.rangeSearch(
                query: query,
                maxDistance: maxDistance,
                limit: Int.max,
                filter: filter,
                metric: metric
            )
            combined.append(contentsOf: baseResults)
        }

        if let delta {
            let deltaResults = try await delta.rangeSearch(
                query: query,
                maxDistance: maxDistance,
                limit: Int.max,
                filter: filter,
                metric: metric
            )
            combined.append(contentsOf: deltaResults)
        }

        for result in pendingSearchResults(query: query, filter: filter, metric: searchMetric)
        where result.score <= maxDistance {
            combined.append(result)
        }

        return Array(dedupeAndSort(combined).prefix(limit))
    }

    public func setMetadata(_ column: String, value: String, for id: String) async throws {
        guard allIDs.contains(id), !deletedIDs.contains(id) else {
            throw ANNSError.idNotFound(id)
        }

        var row = metadataByID[id] ?? [:]
        row[column] = .string(value)
        metadataByID[id] = row

        if idInBase.contains(id), let base {
            try await base.setMetadata(column, value: value, for: id)
        }
        if idInDelta.contains(id), let delta {
            try await delta.setMetadata(column, value: value, for: id)
        }
    }

    public func setMetadata(_ column: String, value: Float, for id: String) async throws {
        guard allIDs.contains(id), !deletedIDs.contains(id) else {
            throw ANNSError.idNotFound(id)
        }

        var row = metadataByID[id] ?? [:]
        row[column] = .float(value)
        metadataByID[id] = row

        if idInBase.contains(id), let base {
            try await base.setMetadata(column, value: value, for: id)
        }
        if idInDelta.contains(id), let delta {
            try await delta.setMetadata(column, value: value, for: id)
        }
    }

    public func setMetadata(_ column: String, value: Int64, for id: String) async throws {
        guard allIDs.contains(id), !deletedIDs.contains(id) else {
            throw ANNSError.idNotFound(id)
        }

        var row = metadataByID[id] ?? [:]
        row[column] = .int64(value)
        metadataByID[id] = row

        if idInBase.contains(id), let base {
            try await base.setMetadata(column, value: value, for: id)
        }
        if idInDelta.contains(id), let delta {
            try await delta.setMetadata(column, value: value, for: id)
        }
    }

    public func delete(id: String) async throws {
        guard allIDs.contains(id), !deletedIDs.contains(id) else {
            throw ANNSError.idNotFound(id)
        }

        deletedIDs.insert(id)
        idInBase.remove(id)
        idInDelta.remove(id)

        if let base {
            do {
                try await base.delete(id: id)
            } catch let error as ANNSError {
                guard case .idNotFound = error else {
                    throw error
                }
            }
        }

        if let delta {
            do {
                try await delta.delete(id: id)
            } catch let error as ANNSError {
                guard case .idNotFound = error else {
                    throw error
                }
            }
        }

        removePendingID(id)
    }

    public func flush() async throws {
        try checkBackgroundMergeError()
        if metricsNeedsPropagation {
            await synchronizeChildMetricsIfNeeded()
        }

        if let task = mergeTask {
            defer { mergeTask = nil }
            try await task.value
        }

        try await flushPendingIntoDelta()

        if delta != nil || !pendingVectors.isEmpty {
            try await triggerMerge()
        }
    }

    public func save(to url: URL) async throws {
        try checkBackgroundMergeError()
        try await flush()

        guard let base else {
            throw ANNSError.constructionFailed("Nothing to save — index is empty")
        }

        // Capture a consistent metadata snapshot before cross-actor await points.
        let meta = PersistedMeta(
            config: config,
            vectorDimension: vectorDimension,
            allVectorsList: allVectorsList,
            allIDsList: allIDsList,
            deletedIDs: Array(deletedIDs),
            metadataByID: metadataByID
        )
        try Self.validateLoadedMeta(meta)

        let fileManager = FileManager.default
        let parentURL = url.deletingLastPathComponent()
        let tempURL = parentURL.appendingPathComponent(".\(url.lastPathComponent).tmp-\(UUID().uuidString)")
        let tempBaseURL = tempURL.appendingPathComponent("base.anns")
        let tempMetaURL = tempURL.appendingPathComponent("streaming.meta.json")

        try fileManager.createDirectory(at: tempURL, withIntermediateDirectories: true)
        do {
            try await base.save(to: tempBaseURL)
        } catch {
            try? fileManager.removeItem(at: tempURL)
            throw error
        }

        let data = try JSONEncoder().encode(meta)
        do {
            try data.write(to: tempMetaURL, options: .atomic)
            try Self.replaceDirectory(at: url, with: tempURL)
        } catch {
            try? fileManager.removeItem(at: tempURL)
            throw error
        }
    }

    public static func load(from url: URL) async throws -> StreamingIndex {
        let metaURL = url.appendingPathComponent("streaming.meta.json")
        let data = try Data(contentsOf: metaURL)
        let meta = try JSONDecoder().decode(PersistedMeta.self, from: data)
        try validateLoadedMeta(meta)

        let loadedBase = try await ANNSIndex.load(from: url.appendingPathComponent("base.anns"))
        let streaming = StreamingIndex(config: meta.config)
        await streaming.applyLoadedState(base: loadedBase, meta: meta)
        return streaming
    }

    private func applyLoadedState(base: ANNSIndex, meta: PersistedMeta) {
        self.base = base
        self.delta = nil
        self.mergeTask = nil
        self._isMerging = false
        self.pendingVectors = []
        self.pendingIDs = []

        self.allVectorsList = meta.allVectorsList
        self.allIDsList = meta.allIDsList
        self.allIDs = Set(meta.allIDsList)
        self.deletedIDs = Set(meta.deletedIDs)
        self.metadataByID = meta.metadataByID
        self.vectorDimension = meta.vectorDimension ?? meta.allVectorsList.first?.count

        self.idInBase = Set(meta.allIDsList.filter { !self.deletedIDs.contains($0) })
        self.idInDelta = []
        self.metricsNeedsPropagation = true
    }

    private func validateDimension(of vector: [Float]) throws {
        if vector.isEmpty {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }

        if let expected = vectorDimension {
            guard vector.count == expected else {
                throw ANNSError.dimensionMismatch(expected: expected, got: vector.count)
            }
        } else {
            vectorDimension = vector.count
        }
    }

    private func validateQueryDimension(_ query: [Float]) throws {
        guard let expected = vectorDimension else {
            throw ANNSError.indexEmpty
        }
        guard query.count == expected else {
            throw ANNSError.dimensionMismatch(expected: expected, got: query.count)
        }
    }

    private func removePendingID(_ id: String) {
        if let index = pendingIDs.firstIndex(of: id) {
            pendingIDs.remove(at: index)
            pendingVectors.remove(at: index)
        }
    }

    private func adjustedConfiguration(for nodeCount: Int) -> IndexConfiguration {
        var adjusted = config.indexConfiguration
        adjusted.degree = min(adjusted.degree, max(1, nodeCount - 1))
        return adjusted
    }

    private func flushPendingIntoDelta() async throws {
        while !pendingVectors.isEmpty {
            if delta == nil {
                guard pendingVectors.count >= 2 else {
                    return
                }
                let newDelta = try await buildIndex(vectors: pendingVectors, ids: pendingIDs)
                delta = newDelta
                idInDelta = Set(pendingIDs)
                pendingVectors.removeAll(keepingCapacity: true)
                pendingIDs.removeAll(keepingCapacity: true)
                continue
            }

            guard let delta else {
                return
            }

            do {
                try await delta.batchInsert(pendingVectors, ids: pendingIDs)
                idInDelta.formUnion(pendingIDs)
                pendingVectors.removeAll(keepingCapacity: true)
                pendingIDs.removeAll(keepingCapacity: true)
            } catch let error as ANNSError {
                guard case .constructionFailed(let message) = error,
                      message.contains("Index capacity exceeded")
                else {
                    throw error
                }

                if _isMerging {
                    return
                }
                try await triggerMerge()
            }
        }
    }

    private func shouldMerge() async -> Bool {
        if _isMerging {
            return false
        }
        guard let delta else {
            return false
        }
        return await delta.count >= config.deltaCapacity
    }

    private func maybeTriggerMerge() async throws {
        guard await shouldMerge() else {
            return
        }

        switch config.mergeStrategy {
        case .blocking:
            try await triggerMerge()
        case .background:
            startBackgroundMergeIfNeeded()
        }
    }

    private func startBackgroundMergeIfNeeded() {
        guard mergeTask == nil else {
            return
        }

        let task = Task { [self] in
            try await self.triggerMerge()
        }
        mergeTask = task

        Task { [self] in
            do {
                try await task.value
            } catch let error as ANNSError {
                self.recordBackgroundMergeError(error)
            } catch {
                self.recordBackgroundMergeError(
                    .constructionFailed("Background merge failed: \(error)")
                )
            }
            self.clearMergeTaskReference()
        }
    }

    private func clearMergeTaskReference() {
        mergeTask = nil
    }

    private func recordBackgroundMergeError(_ error: ANNSError) {
        lastBackgroundMergeError = error
    }

    private func activeRecords(upperBound: Int) -> (vectors: [[Float]], ids: [String]) {
        let safeUpper = min(upperBound, allIDsList.count)
        guard safeUpper > 0 else {
            return ([], [])
        }

        var vectors: [[Float]] = []
        var ids: [String] = []
        vectors.reserveCapacity(safeUpper)
        ids.reserveCapacity(safeUpper)

        for index in 0..<safeUpper {
            let id = allIDsList[index]
            guard !deletedIDs.contains(id) else {
                continue
            }
            ids.append(id)
            vectors.append(allVectorsList[index])
        }
        return (vectors, ids)
    }

    private func activeRecords(in range: Range<Int>) -> (vectors: [[Float]], ids: [String]) {
        let lower = max(0, range.lowerBound)
        let upper = min(allIDsList.count, range.upperBound)
        guard lower < upper else {
            return ([], [])
        }

        var vectors: [[Float]] = []
        var ids: [String] = []
        vectors.reserveCapacity(upper - lower)
        ids.reserveCapacity(upper - lower)

        for index in lower..<upper {
            let id = allIDsList[index]
            guard !deletedIDs.contains(id) else {
                continue
            }
            ids.append(id)
            vectors.append(allVectorsList[index])
        }
        return (vectors, ids)
    }

    private func buildIndex(vectors: [[Float]], ids: [String]) async throws -> ANNSIndex {
        let index = ANNSIndex(configuration: adjustedConfiguration(for: vectors.count))
        try await index.build(vectors: vectors, ids: ids)
        if let metrics {
            await index.setMetrics(metrics)
        }
        try await applyStoredMetadata(to: index, ids: ids)
        return index
    }

    private func applyStoredMetadata(to index: ANNSIndex, ids: [String]) async throws {
        for id in ids {
            guard let row = metadataByID[id] else {
                continue
            }

            for (column, value) in row {
                switch value {
                case .string(let value):
                    try await index.setMetadata(column, value: value, for: id)
                case .float(let value):
                    try await index.setMetadata(column, value: value, for: id)
                case .int64(let value):
                    try await index.setMetadata(column, value: value, for: id)
                }
            }
        }
    }

    private func triggerMerge() async throws {
        guard !_isMerging else {
            return
        }
        _isMerging = true
        defer { _isMerging = false }

        let snapshotCount = allIDsList.count
        let merged = activeRecords(upperBound: snapshotCount)

        guard !merged.ids.isEmpty else {
            base = nil
            delta = nil
            idInBase.removeAll()
            idInDelta.removeAll()
            pendingVectors.removeAll(keepingCapacity: true)
            pendingIDs.removeAll(keepingCapacity: true)
            return
        }

        guard merged.ids.count >= 2 else {
            base = nil
            delta = nil
            idInBase.removeAll()
            idInDelta.removeAll()
            pendingVectors = merged.vectors
            pendingIDs = merged.ids
            return
        }

        let newBase = try await buildIndex(vectors: merged.vectors, ids: merged.ids)

        base = newBase
        idInBase = Set(merged.ids)
        delta = nil
        idInDelta.removeAll()
        pendingVectors.removeAll(keepingCapacity: true)
        pendingIDs.removeAll(keepingCapacity: true)
        if let metrics {
            await metrics.recordMerge()
        }

        let tail = activeRecords(in: snapshotCount..<allIDsList.count)
        if tail.ids.count >= 2 {
            let newDelta = try await buildIndex(vectors: tail.vectors, ids: tail.ids)
            delta = newDelta
            idInDelta = Set(tail.ids)
        } else if tail.ids.count == 1 {
            pendingVectors = tail.vectors
            pendingIDs = tail.ids
        }

        lastBackgroundMergeError = nil
    }

    private func pendingSearchResults(
        query: [Float],
        filter: SearchFilter?,
        metric: Metric
    ) -> [SearchResult] {
        var results: [SearchResult] = []
        results.reserveCapacity(pendingVectors.count)

        for (vector, id) in zip(pendingVectors, pendingIDs) {
            guard !deletedIDs.contains(id), matchesFilter(for: id, filter: filter) else {
                continue
            }

            let score = SIMDDistance.distance(query, vector, metric: metric)
            results.append(SearchResult(id: id, score: score, internalID: UInt32.max))
        }

        return results
    }

    private func dedupeAndSort(_ results: [SearchResult]) -> [SearchResult] {
        var bestByID: [String: SearchResult] = [:]
        bestByID.reserveCapacity(results.count)

        for result in results {
            if let existing = bestByID[result.id] {
                if result.score < existing.score {
                    bestByID[result.id] = result
                }
            } else {
                bestByID[result.id] = result
            }
        }

        return bestByID.values.sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.id < rhs.id
            }
            return lhs.score < rhs.score
        }
    }

    private func matchesFilter(for id: String, filter: SearchFilter?) -> Bool {
        guard let filter else {
            return true
        }
        let row = metadataByID[id] ?? [:]
        return evaluate(filter: filter, row: row)
    }

    private func evaluate(filter: SearchFilter, row: [String: MetadataValue]) -> Bool {
        switch filter {
        case .equals(column: let column, value: let value):
            if case .string(let current)? = row[column] {
                return current == value
            }
            return false

        case .greaterThan(column: let column, value: let value):
            guard let current = numericValue(from: row[column]) else {
                return false
            }
            return current > value

        case .lessThan(column: let column, value: let value):
            guard let current = numericValue(from: row[column]) else {
                return false
            }
            return current < value

        case .greaterThanInt(column: let column, value: let value):
            guard let current = integerValue(from: row[column]) else {
                return false
            }
            return current > value

        case .lessThanInt(column: let column, value: let value):
            guard let current = integerValue(from: row[column]) else {
                return false
            }
            return current < value

        case .in(column: let column, values: let values):
            if case .string(let current)? = row[column] {
                return values.contains(current)
            }
            return false

        case .and(let filters):
            return filters.allSatisfy { evaluate(filter: $0, row: row) }

        case .or(let filters):
            return filters.contains { evaluate(filter: $0, row: row) }

        case .not(let inner):
            return !evaluate(filter: inner, row: row)
        }
    }

    private func numericValue(from value: MetadataValue?) -> Float? {
        switch value {
        case .float(let value):
            return value
        case .int64(let value):
            return Float(value)
        case .string, .none:
            return nil
        }
    }

    private func integerValue(from value: MetadataValue?) -> Int64? {
        switch value {
        case .int64(let value):
            return value
        case .float, .string, .none:
            return nil
        }
    }

    private func synchronizeChildMetricsIfNeeded() async {
        guard metricsNeedsPropagation else {
            return
        }
        if let base {
            await base.setMetrics(metrics)
        }
        if let delta {
            await delta.setMetrics(metrics)
        }
        metricsNeedsPropagation = false
    }

    private func checkBackgroundMergeError() throws {
        if let lastBackgroundMergeError {
            throw lastBackgroundMergeError
        }
    }

    private static func validateLoadedMeta(_ meta: PersistedMeta) throws {
        guard meta.allVectorsList.count == meta.allIDsList.count else {
            throw ANNSError.corruptFile("Streaming metadata vector and ID counts do not match")
        }

        let allIDSet = Set(meta.allIDsList)
        guard allIDSet.count == meta.allIDsList.count else {
            throw ANNSError.corruptFile("Streaming metadata contains duplicate IDs")
        }
        guard Set(meta.deletedIDs).isSubset(of: allIDSet) else {
            throw ANNSError.corruptFile("Streaming metadata deleted IDs are not a subset of all IDs")
        }

        for id in meta.metadataByID.keys where !allIDSet.contains(id) {
            throw ANNSError.corruptFile("Streaming metadata contains row for unknown ID '\(id)'")
        }

        let resolvedDimension = meta.vectorDimension ?? meta.allVectorsList.first?.count
        if let resolvedDimension {
            guard resolvedDimension > 0 else {
                throw ANNSError.corruptFile("Streaming metadata has invalid vector dimension")
            }
            for vector in meta.allVectorsList where vector.count != resolvedDimension {
                throw ANNSError.corruptFile("Streaming metadata vectors have inconsistent dimensions")
            }
        } else if !meta.allVectorsList.isEmpty {
            throw ANNSError.corruptFile("Streaming metadata is missing vector dimension")
        }
    }

    private static func replaceDirectory(at destinationURL: URL, with sourceURL: URL) throws {
        let fileManager = FileManager.default
        let parentURL = destinationURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: parentURL, withIntermediateDirectories: true)

        if fileManager.fileExists(atPath: destinationURL.path) {
            let backupURL = parentURL.appendingPathComponent(".\(destinationURL.lastPathComponent).backup-\(UUID().uuidString)")
            do {
                try fileManager.moveItem(at: destinationURL, to: backupURL)
                try fileManager.moveItem(at: sourceURL, to: destinationURL)
                try? fileManager.removeItem(at: backupURL)
            } catch {
                if !fileManager.fileExists(atPath: destinationURL.path),
                   fileManager.fileExists(atPath: backupURL.path) {
                    try? fileManager.moveItem(at: backupURL, to: destinationURL)
                }
                throw error
            }
        } else {
            try fileManager.moveItem(at: sourceURL, to: destinationURL)
        }
    }
}
