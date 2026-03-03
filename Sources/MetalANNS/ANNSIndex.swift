import Foundation
import Metal
import MetalANNSCore

public actor ANNSIndex {
    private static let fullGPUMaxEF = 256

    private struct PersistedMetadata: Codable, Sendable {
        let configuration: IndexConfiguration
        let softDeletion: SoftDeletion
        let metadataStore: MetadataStore?
        let idMap: IDMap?
    }

    private var configuration: IndexConfiguration
    private var context: MetalContext?
    private var vectors: (any VectorStorage)?
    private var graph: GraphBuffer?
    private var idMap: IDMap
    private var softDeletion: SoftDeletion
    private var metadataStore: MetadataStore
    private var entryPoint: UInt32
    private var isBuilt: Bool
    private var isReadOnlyLoadedIndex: Bool
    private var mmapLifetime: AnyObject?
    private var pendingRepairIDs: [UInt32] = []
    private var hnsw: HNSWLayers?
    public var metrics: IndexMetrics? = nil

    public init(configuration: IndexConfiguration = .default) {
        self.configuration = configuration
        self.context = try? MetalContext()
        self.vectors = nil
        self.graph = nil
        self.idMap = IDMap()
        self.softDeletion = SoftDeletion()
        self.metadataStore = MetadataStore()
        self.entryPoint = 0
        self.isBuilt = false
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
        self.hnsw = nil
    }

    init(configuration: IndexConfiguration = .default, context: MetalContext?) {
        self.configuration = configuration
        self.context = context
        self.vectors = nil
        self.graph = nil
        self.idMap = IDMap()
        self.softDeletion = SoftDeletion()
        self.metadataStore = MetadataStore()
        self.entryPoint = 0
        self.isBuilt = false
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
        self.hnsw = nil
    }

    public func build(vectors inputVectors: [[Float]], ids: [String]) async throws {
        guard !inputVectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot build index with empty vectors")
        }
        guard inputVectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }

        let dim = inputVectors[0].count
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }
        for vector in inputVectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }

        var seenIDs = Set<String>()
        for id in ids {
            if !seenIDs.insert(id).inserted {
                throw ANNSError.idAlreadyExists(id)
            }
        }
        guard inputVectors.count >= 2 else {
            throw ANNSError.constructionFailed("Build requires at least 2 vectors")
        }
        guard configuration.degree > 0 else {
            throw ANNSError.constructionFailed("Degree must be greater than zero")
        }
        guard configuration.degree < inputVectors.count else {
            throw ANNSError.constructionFailed(
                "Degree \(configuration.degree) must be less than node count \(inputVectors.count)"
            )
        }
        if configuration.metric == .hamming, !configuration.useBinary {
            throw ANNSError.constructionFailed("metric .hamming requires useBinary == true")
        }

        let capacity = max(2, inputVectors.count * 2)
        let device = context?.device
        let vectorBuffer: any VectorStorage
        if configuration.useBinary {
            guard configuration.metric == .hamming else {
                throw ANNSError.constructionFailed("useBinary requires metric == .hamming")
            }
            guard dim % 8 == 0 else {
                throw ANNSError.constructionFailed("Binary index requires dim % 8 == 0, got dim=\(dim)")
            }
            vectorBuffer = try BinaryVectorBuffer(capacity: capacity, dim: dim, device: device)
        } else if configuration.useFloat16 {
            vectorBuffer = try Float16VectorBuffer(capacity: capacity, dim: dim, device: device)
        } else {
            vectorBuffer = try VectorBuffer(capacity: capacity, dim: dim, device: device)
        }

        let graphBuffer = try GraphBuffer(capacity: capacity, degree: configuration.degree, device: device)
        try vectorBuffer.batchInsert(vectors: inputVectors, startingAt: 0)
        vectorBuffer.setCount(inputVectors.count)

        var builtIDMap = IDMap()
        for id in ids {
            guard builtIDMap.assign(externalID: id) != nil else {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        let builtEntryPoint: UInt32
        let cpuVectors = (0..<inputVectors.count).map { vectorBuffer.vector(at: $0) }
        if let context, !configuration.useBinary, configuration.metric != .hamming {
            try await NNDescentGPU.build(
                context: context,
                vectors: vectorBuffer,
                graph: graphBuffer,
                nodeCount: inputVectors.count,
                metric: configuration.metric,
                maxIterations: configuration.maxIterations,
                convergenceThreshold: configuration.convergenceThreshold
            )
            builtEntryPoint = 0
        } else {
            let cpuResult = try await NNDescentCPU.build(
                vectors: cpuVectors,
                degree: configuration.degree,
                metric: configuration.metric,
                maxIterations: configuration.maxIterations,
                convergenceThreshold: configuration.convergenceThreshold
            )

            for nodeID in 0..<cpuResult.graph.count {
                let neighbors = cpuResult.graph[nodeID]
                var neighborIDs = Array(repeating: UInt32.max, count: configuration.degree)
                var neighborDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
                for slot in 0..<min(configuration.degree, neighbors.count) {
                    neighborIDs[slot] = neighbors[slot].0
                    neighborDistances[slot] = neighbors[slot].1
                }
                try graphBuffer.setNeighbors(of: nodeID, ids: neighborIDs, distances: neighborDistances)
            }
            graphBuffer.setCount(inputVectors.count)
            builtEntryPoint = cpuResult.entryPoint
        }

        try GraphPruner.prune(
            graph: graphBuffer,
            vectors: vectorBuffer,
            nodeCount: inputVectors.count,
            metric: configuration.metric
        )

        self.vectors = vectorBuffer
        self.graph = graphBuffer
        self.idMap = builtIDMap
        self.softDeletion = SoftDeletion()
        self.metadataStore = MetadataStore()
        self.entryPoint = builtEntryPoint
        self.isBuilt = true
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
        self.pendingRepairIDs.removeAll()
        self.hnsw = nil
        try rebuildHNSWFromCurrentState()
    }

    public func insert(_ vector: [Float], id: String) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard vector.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: vector.count)
        }
        if idMap.internalID(for: id) != nil {
            throw ANNSError.idAlreadyExists(id)
        }

        guard idMap.canAllocate(1) else {
            throw ANNSError.constructionFailed("Internal ID space exhausted")
        }

        let nextInternalID = Int(idMap.nextInternalID)
        guard nextInternalID < vectors.capacity, nextInternalID < graph.capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
        }

        let graphVector = vectors is BinaryVectorBuffer ? Self.quantizeForHamming(vector) : vector
        let slot = nextInternalID
        let previousVectorCount = vectors.count
        let previousGraphCount = graph.nodeCount

        let metricsRecorder = metrics
        let insertStart = metricsRecorder == nil ? nil : ContinuousClock.now

        do {
            try vectors.insert(vector: graphVector, at: slot)
            if vectors.count < slot + 1 {
                vectors.setCount(slot + 1)
            }

            try IncrementalBuilder.insert(
                vector: graphVector,
                at: slot,
                into: graph,
                vectors: vectors,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )
            if graph.nodeCount < slot + 1 {
                graph.setCount(slot + 1)
            }
            hnsw = nil

            let repairConfig = configuration.repairConfiguration
            if repairConfig.enabled && repairConfig.repairInterval > 0 {
                pendingRepairIDs.append(UInt32(slot))
                if pendingRepairIDs.count >= repairConfig.repairInterval {
                    try triggerRepair(throwOnFailure: false)
                }
            }

            guard let assignedID = idMap.assign(externalID: id), Int(assignedID) == slot else {
                throw ANNSError.constructionFailed("Failed to commit internal ID for '\(id)'")
            }
        } catch {
            vectors.setCount(previousVectorCount)
            graph.setCount(previousGraphCount)

            // Best-effort cleanup of the uncommitted node slot.
            let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
            let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
            try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)

            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Incremental insert failed: \(error)")
        }

        if let metricsRecorder, let insertStart {
            let duration = ContinuousClock.now - insertStart
            await metricsRecorder.recordInsert(durationNs: Self.durationNanoseconds(duration))
        }
    }

    /// Inserts a vector with a numeric (UInt64) key.
    /// For use by Wax's UInt64 frameId-based API.
    public func insert(_ vector: [Float], numericID: UInt64) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard vector.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: vector.count)
        }
        if idMap.internalID(forNumeric: numericID) != nil {
            throw ANNSError.idAlreadyExists(String(numericID))
        }

        guard idMap.canAllocate(1) else {
            throw ANNSError.constructionFailed("Internal ID space exhausted")
        }

        let nextInternalID = Int(idMap.nextInternalID)
        guard nextInternalID < vectors.capacity, nextInternalID < graph.capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
        }

        let graphVector = vectors is BinaryVectorBuffer ? Self.quantizeForHamming(vector) : vector
        let slot = nextInternalID
        let previousVectorCount = vectors.count
        let previousGraphCount = graph.nodeCount

        let metricsRecorder = metrics
        let insertStart = metricsRecorder == nil ? nil : ContinuousClock.now

        do {
            try vectors.insert(vector: graphVector, at: slot)
            if vectors.count < slot + 1 {
                vectors.setCount(slot + 1)
            }

            try IncrementalBuilder.insert(
                vector: graphVector,
                at: slot,
                into: graph,
                vectors: vectors,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )
            if graph.nodeCount < slot + 1 {
                graph.setCount(slot + 1)
            }
            hnsw = nil

            let repairConfig = configuration.repairConfiguration
            if repairConfig.enabled && repairConfig.repairInterval > 0 {
                pendingRepairIDs.append(UInt32(slot))
                if pendingRepairIDs.count >= repairConfig.repairInterval {
                    try triggerRepair(throwOnFailure: false)
                }
            }

            guard let assignedID = idMap.assign(numericID: numericID), Int(assignedID) == slot else {
                throw ANNSError.constructionFailed("Failed to commit internal ID for numeric \(numericID)")
            }
        } catch {
            vectors.setCount(previousVectorCount)
            graph.setCount(previousGraphCount)

            // Best-effort cleanup of the uncommitted node slot.
            let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
            let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
            try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)

            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Incremental insert (numeric) failed: \(error)")
        }

        if let metricsRecorder, let insertStart {
            let duration = ContinuousClock.now - insertStart
            await metricsRecorder.recordInsert(durationNs: Self.durationNanoseconds(duration))
        }
    }

    public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws {
        guard isBuilt, let vectorStorage = self.vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }

        let dim = vectorStorage.dim
        for vector in vectors {
            guard vector.count == dim else {
                throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
            }
        }

        var seenIDs = Set<String>()
        for id in ids {
            if !seenIDs.insert(id).inserted {
                throw ANNSError.idAlreadyExists(id)
            }
            if idMap.internalID(for: id) != nil {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        guard idMap.canAllocate(ids.count) else {
            throw ANNSError.constructionFailed("Internal ID space exhausted")
        }

        let startSlot = Int(idMap.nextInternalID)
        guard startSlot + vectors.count <= vectorStorage.capacity,
              startSlot + vectors.count <= graph.capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
        }

        let slots = Array(startSlot..<(startSlot + vectors.count))
        let previousVectorCount = vectorStorage.count
        let previousGraphCount = graph.nodeCount

        let metricsRecorder = metrics
        let batchInsertStart = metricsRecorder == nil ? nil : ContinuousClock.now
        let insertedCount = vectors.count

        do {
            let graphVectors: [[Float]]
            if vectorStorage is BinaryVectorBuffer {
                graphVectors = vectors.map(Self.quantizeForHamming)
            } else {
                graphVectors = vectors
            }

            for (offset, vector) in graphVectors.enumerated() {
                try vectorStorage.insert(vector: vector, at: slots[offset])
            }
            let newMaxCount = (slots.last ?? 0) + 1
            if vectorStorage.count < newMaxCount {
                vectorStorage.setCount(newMaxCount)
            }

            try BatchIncrementalBuilder.batchInsert(
                vectors: graphVectors,
                startingAt: startSlot,
                into: graph,
                vectorStorage: vectorStorage,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )

            if graph.nodeCount < newMaxCount {
                graph.setCount(newMaxCount)
            }
            hnsw = nil

            let repairConfig = configuration.repairConfiguration
            if repairConfig.enabled {
                pendingRepairIDs.append(contentsOf: slots.map(UInt32.init))
                try triggerRepair(throwOnFailure: false)
            }

            for (offset, id) in ids.enumerated() {
                guard let assignedID = idMap.assign(externalID: id), Int(assignedID) == slots[offset] else {
                    throw ANNSError.constructionFailed("Failed to commit internal ID for '\(id)'")
                }
            }
        } catch {
            vectorStorage.setCount(previousVectorCount)
            graph.setCount(previousGraphCount)
            pendingRepairIDs.removeAll { internalID in
                slots.contains(Int(internalID))
            }

            // Best-effort cleanup for uncommitted slots.
            let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
            let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
            for slot in slots {
                try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)
            }

            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Batch insert failed: \(error)")
        }

        if let metricsRecorder, let batchInsertStart {
            let duration = ContinuousClock.now - batchInsertStart
            await metricsRecorder.recordBatchInsert(
                count: insertedCount,
                durationNs: Self.durationNanoseconds(duration)
            )
        }
    }

    public func repair() throws(ANNSError) {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard isBuilt, vectors != nil, graph != nil else {
            throw ANNSError.indexEmpty
        }
        guard !pendingRepairIDs.isEmpty else {
            return
        }
        try triggerRepair()
    }

    private func triggerRepair(throwOnFailure: Bool = true) throws(ANNSError) {
        guard let vectors, let graph else {
            return
        }
        guard !pendingRepairIDs.isEmpty else {
            return
        }

        let idsToRepair = pendingRepairIDs
        pendingRepairIDs.removeAll(keepingCapacity: true)

        do {
            _ = try GraphRepairer.repair(
                recentIDs: idsToRepair,
                vectors: vectors,
                graph: graph,
                config: configuration.repairConfiguration,
                metric: configuration.metric
            )
            hnsw = nil
        } catch {
            // Preserve pending IDs so repair can be retried later.
            pendingRepairIDs = idsToRepair + pendingRepairIDs
            if throwOnFailure {
                throw error
            }
        }
    }

    public func delete(id: String) throws {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        softDeletion.markDeleted(internalID)
        metadataStore.remove(id: internalID)
    }

    public func compact() async throws {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard softDeletion.deletedCount > 0 else {
            return
        }

        let result = try await IndexCompactor.compact(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            softDeletion: softDeletion,
            metric: configuration.metric,
            degree: configuration.degree,
            context: context,
            maxIterations: configuration.maxIterations,
            convergenceThreshold: configuration.convergenceThreshold,
            useFloat16: configuration.useFloat16,
            useBinary: configuration.useBinary
        )

        var remapping: [UInt32: UInt32] = [:]
        remapping.reserveCapacity(result.idMap.count)
        for oldIndex in 0..<vectors.count {
            let oldID = UInt32(oldIndex)
            if softDeletion.isDeleted(oldID) {
                continue
            }
            let newID: UInt32?
            if let externalID = idMap.externalID(for: oldID) {
                newID = result.idMap.internalID(for: externalID)
            } else if let numericID = idMap.numericID(for: oldID) {
                newID = result.idMap.internalID(forNumeric: numericID)
            } else {
                newID = nil
            }
            guard let newID else {
                continue
            }
            remapping[oldID] = newID
        }

        self.vectors = result.vectors
        self.graph = result.graph
        self.idMap = result.idMap
        self.entryPoint = result.entryPoint
        self.softDeletion = SoftDeletion()
        self.metadataStore = metadataStore.remapped(using: remapping)
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
        self.pendingRepairIDs.removeAll()
        self.hnsw = nil
    }

    public func setMetadata(_ column: String, value: String, for id: String) throws {
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, stringValue: value, for: internalID)
    }

    public func setMetadata(_ column: String, value: Float, for id: String) throws {
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, floatValue: value, for: internalID)
    }

    public func setMetadata(_ column: String, value: Int64, for id: String) throws {
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, intValue: value, for: internalID)
    }

    public func search(
        query: [Float],
        k: Int,
        filter: SearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard query.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
        }
        guard k > 0 else {
            return []
        }

        let searchMetric = metric ?? configuration.metric
        if searchMetric == .hamming, !configuration.useBinary {
            throw ANNSError.searchFailed("metric .hamming requires a binary index")
        }
        let normalizedQuery = (searchMetric == .hamming && configuration.useBinary)
            ? Self.quantizeForHamming(query)
            : query
        let metricsRecorder = metrics
        let searchStart = metricsRecorder == nil ? nil : ContinuousClock.now
        let hasFilter = filter != nil
        let deletedCount = softDeletion.deletedCount
        let effectiveK: Int
        if hasFilter {
            effectiveK = min(vectors.count, k * 4 + deletedCount)
        } else {
            effectiveK = min(vectors.count, k + deletedCount)
        }
        let effectiveEf = max(configuration.efSearch, effectiveK)

        let rawResults: [SearchResult]
        let canAttemptGPU = supportsGPUSearch(for: vectors)
            && searchMetric != .hamming
            && effectiveK <= Self.fullGPUMaxEF
            && effectiveEf <= Self.fullGPUMaxEF

        if let context, canAttemptGPU {
            do {
                rawResults = try await SearchGPU.search(
                    context: context,
                    query: normalizedQuery,
                    vectors: vectors,
                    graph: graph,
                    entryPoint: Int(entryPoint),
                    k: max(1, effectiveK),
                    ef: max(1, effectiveEf),
                    metric: searchMetric
                )
            } catch {
                if configuration.hnswConfiguration.enabled, hnsw == nil {
                    try rebuildHNSWFromCurrentState()
                }

                let extractedVectors = extractVectors(from: vectors)
                let extractedGraph = extractGraph(from: graph)

                if !isReadOnlyLoadedIndex, let hnsw, searchMetric == configuration.metric {
                    rawResults = try await HNSWSearchCPU.search(
                        query: normalizedQuery,
                        vectors: extractedVectors,
                        hnsw: hnsw,
                        baseGraph: extractedGraph,
                        k: max(1, effectiveK),
                        ef: max(1, effectiveEf),
                        metric: searchMetric
                    )
                } else {
                    rawResults = try await BeamSearchCPU.search(
                        query: normalizedQuery,
                        vectors: extractedVectors,
                        graph: extractedGraph,
                        entryPoint: Int(entryPoint),
                        k: max(1, effectiveK),
                        ef: max(1, effectiveEf),
                        metric: searchMetric
                    )
                }
            }
        } else {
            if configuration.hnswConfiguration.enabled, hnsw == nil {
                try rebuildHNSWFromCurrentState()
            }

            let extractedVectors = extractVectors(from: vectors)
            let extractedGraph = extractGraph(from: graph)

            if !isReadOnlyLoadedIndex, let hnsw, searchMetric == configuration.metric {
                rawResults = try await HNSWSearchCPU.search(
                    query: normalizedQuery,
                    vectors: extractedVectors,
                    hnsw: hnsw,
                    baseGraph: extractedGraph,
                    k: max(1, effectiveK),
                    ef: max(1, effectiveEf),
                    metric: searchMetric
                )
            } else {
                rawResults = try await BeamSearchCPU.search(
                    query: normalizedQuery,
                    vectors: extractedVectors,
                    graph: extractedGraph,
                    entryPoint: Int(entryPoint),
                    k: max(1, effectiveK),
                    ef: max(1, effectiveEf),
                    metric: searchMetric
                )
            }
        }

        var filtered = softDeletion.filterResults(rawResults)
        if let filter {
            filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
        }

        let mapped = filtered.compactMap { result -> SearchResult? in
            let externalID = idMap.externalID(for: result.internalID) ?? ""
            let numericID = idMap.numericID(for: result.internalID)
            guard !externalID.isEmpty || numericID != nil else {
                return nil
            }
            return SearchResult(
                id: externalID,
                score: result.score,
                internalID: result.internalID,
                numericID: numericID
            )
        }
        let output = Array(mapped.prefix(k))
        if let metricsRecorder, let searchStart {
            let duration = ContinuousClock.now - searchStart
            await metricsRecorder.recordSearch(durationNs: Self.durationNanoseconds(duration))
        }
        return output
    }

    public func rangeSearch(
        query: [Float],
        maxDistance: Float,
        limit: Int = 1000,
        filter: SearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard query.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
        }
        guard maxDistance >= 0 else {
            return []
        }
        guard limit > 0 else {
            return []
        }

        let searchMetric = metric ?? configuration.metric
        if searchMetric == .hamming, !configuration.useBinary {
            throw ANNSError.searchFailed("metric .hamming requires a binary index")
        }
        let normalizedQuery = (searchMetric == .hamming && configuration.useBinary)
            ? Self.quantizeForHamming(query)
            : query
        let metricsRecorder = metrics
        let searchStart = metricsRecorder == nil ? nil : ContinuousClock.now
        let deletedCount = softDeletion.deletedCount
        let searchK = min(vectors.count, limit + deletedCount)
        let searchEf = min(vectors.count, max(configuration.efSearch, searchK * 2))

        let rawResults: [SearchResult]
        let canAttemptGPU = supportsGPUSearch(for: vectors)
            && searchMetric != .hamming
            && searchK <= Self.fullGPUMaxEF
            && searchEf <= Self.fullGPUMaxEF

        if let context, canAttemptGPU {
            do {
                rawResults = try await SearchGPU.search(
                    context: context,
                    query: normalizedQuery,
                    vectors: vectors,
                    graph: graph,
                    entryPoint: Int(entryPoint),
                    k: max(1, searchK),
                    ef: max(1, searchEf),
                    metric: searchMetric
                )
            } catch {
                if configuration.hnswConfiguration.enabled, hnsw == nil {
                    try rebuildHNSWFromCurrentState()
                }

                let extractedVectors = extractVectors(from: vectors)
                let extractedGraph = extractGraph(from: graph)

                if !isReadOnlyLoadedIndex, let hnsw, searchMetric == configuration.metric {
                    rawResults = try await HNSWSearchCPU.search(
                        query: normalizedQuery,
                        vectors: extractedVectors,
                        hnsw: hnsw,
                        baseGraph: extractedGraph,
                        k: max(1, searchK),
                        ef: max(1, searchEf),
                        metric: searchMetric
                    )
                } else {
                    rawResults = try await BeamSearchCPU.search(
                        query: normalizedQuery,
                        vectors: extractedVectors,
                        graph: extractedGraph,
                        entryPoint: Int(entryPoint),
                        k: max(1, searchK),
                        ef: max(1, searchEf),
                        metric: searchMetric
                    )
                }
            }
        } else {
            if configuration.hnswConfiguration.enabled, hnsw == nil {
                try rebuildHNSWFromCurrentState()
            }

            let extractedVectors = extractVectors(from: vectors)
            let extractedGraph = extractGraph(from: graph)

            if !isReadOnlyLoadedIndex, let hnsw, searchMetric == configuration.metric {
                rawResults = try await HNSWSearchCPU.search(
                    query: normalizedQuery,
                    vectors: extractedVectors,
                    hnsw: hnsw,
                    baseGraph: extractedGraph,
                    k: max(1, searchK),
                    ef: max(1, searchEf),
                    metric: searchMetric
                )
            } else {
                rawResults = try await BeamSearchCPU.search(
                    query: normalizedQuery,
                    vectors: extractedVectors,
                    graph: extractedGraph,
                    entryPoint: Int(entryPoint),
                    k: max(1, searchK),
                    ef: max(1, searchEf),
                    metric: searchMetric
                )
            }
        }

        var filtered = softDeletion.filterResults(rawResults)
        if let filter {
            filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
        }
        let withinRange = filtered.filter { $0.score <= maxDistance }

        let mapped = withinRange.compactMap { result -> SearchResult? in
            let externalID = idMap.externalID(for: result.internalID) ?? ""
            let numericID = idMap.numericID(for: result.internalID)
            guard !externalID.isEmpty || numericID != nil else {
                return nil
            }
            return SearchResult(
                id: externalID,
                score: result.score,
                internalID: result.internalID,
                numericID: numericID
            )
        }
        let output = Array(mapped.prefix(limit))
        if let metricsRecorder, let searchStart {
            let duration = ContinuousClock.now - searchStart
            await metricsRecorder.recordSearch(durationNs: Self.durationNanoseconds(duration))
        }
        return output
    }

    public func batchSearch(
        queries: [[Float]],
        k: Int,
        filter: SearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [[SearchResult]] {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard !queries.isEmpty else {
            return []
        }
        if let metrics {
            await metrics.recordBatchSearch()
        }

        let maxConcurrency = await batchSearchMaxConcurrency()

        return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
            var orderedResults = Array<[SearchResult]?>(repeating: nil, count: queries.count)
            var nextIndex = 0

            for _ in 0..<min(maxConcurrency, queries.count) {
                let idx = nextIndex
                let query = queries[idx]
                nextIndex += 1
                group.addTask { [self] in
                    let result = try await self.search(query: query, k: k, filter: filter, metric: metric)
                    return (idx, result)
                }
            }

            for try await (idx, result) in group {
                orderedResults[idx] = result
                if nextIndex < queries.count {
                    let idx = nextIndex
                    let query = queries[idx]
                    nextIndex += 1
                    group.addTask { [self] in
                        let result = try await self.search(query: query, k: k, filter: filter, metric: metric)
                        return (idx, result)
                    }
                }
            }

            return orderedResults.map { $0! }
        }
    }

    public func setMetrics(_ metrics: IndexMetrics?) {
        self.metrics = metrics
    }

    func batchSearchMaxConcurrencyForTesting() async -> Int {
        await batchSearchMaxConcurrency()
    }

    public func save(to url: URL) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }

        let fileManager = FileManager.default
        let parentURL = url.deletingLastPathComponent()
        let tempDirURL = parentURL.appendingPathComponent(".save-tmp-\(UUID().uuidString)")
        let tempANNS = tempDirURL.appendingPathComponent(url.lastPathComponent)
        let dbURL = URL(fileURLWithPath: Self.databasePath(for: url))
        let tempDB = tempDirURL.appendingPathComponent(dbURL.lastPathComponent)

        do {
            try fileManager.createDirectory(at: tempDirURL, withIntermediateDirectories: true)

            try IndexSerializer.save(
                vectors: vectors,
                graph: graph,
                idMap: idMap,
                entryPoint: entryPoint,
                metric: configuration.metric,
                to: tempANNS
            )

            var db: IndexDatabase? = try IndexDatabase(path: tempDB.path)
            try db?.saveIDMap(idMap)
            try db?.saveConfiguration(configuration)
            try db?.saveSoftDeletion(softDeletion)
            try db?.saveMetadataStore(metadataStore)
            try db?.prepareForFileMove()
            db = nil

            try Self.replaceFile(at: url, with: tempANNS)
            try Self.replaceSQLiteFiles(at: dbURL, with: tempDB)
            Self.removeLegacyMetadataSidecars(for: url)
            try? fileManager.removeItem(at: tempDirURL)
        } catch {
            try? fileManager.removeItem(at: tempDirURL)
            throw error
        }
    }

    public func saveMmapCompatible(to url: URL) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }

        let fileManager = FileManager.default
        let parentURL = url.deletingLastPathComponent()
        let tempDirURL = parentURL.appendingPathComponent(".save-tmp-\(UUID().uuidString)")
        let tempANNS = tempDirURL.appendingPathComponent(url.lastPathComponent)
        let dbURL = URL(fileURLWithPath: Self.databasePath(for: url))
        let tempDB = tempDirURL.appendingPathComponent(dbURL.lastPathComponent)

        do {
            try fileManager.createDirectory(at: tempDirURL, withIntermediateDirectories: true)

            try IndexSerializer.saveMmapCompatible(
                vectors: vectors,
                graph: graph,
                idMap: idMap,
                entryPoint: entryPoint,
                metric: configuration.metric,
                to: tempANNS
            )

            var db: IndexDatabase? = try IndexDatabase(path: tempDB.path)
            try db?.saveIDMap(idMap)
            try db?.saveConfiguration(configuration)
            try db?.saveSoftDeletion(softDeletion)
            try db?.saveMetadataStore(metadataStore)
            try db?.prepareForFileMove()
            db = nil

            try Self.replaceFile(at: url, with: tempANNS)
            try Self.replaceSQLiteFiles(at: dbURL, with: tempDB)
            Self.removeLegacyMetadataSidecars(for: url)
            try? fileManager.removeItem(at: tempDirURL)
        } catch {
            try? fileManager.removeItem(at: tempDirURL)
            throw error
        }
    }

    public static func load(from url: URL) async throws -> ANNSIndex {
        let persistedState = try resolvePersistedState(for: url)
        let initialConfiguration = persistedState.configuration ?? .default
        let index = ANNSIndex(configuration: initialConfiguration)

        let loaded = try IndexSerializer.load(from: url, device: await index.currentDevice())
        let resolvedIDMap = resolveLoadedIDMap(
            persistedIDMap: persistedState.idMap,
            serializerIDMap: loaded.idMap
        )

        var resolvedConfiguration = persistedState.configuration ?? .default
        resolvedConfiguration.metric = loaded.metric
        resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16
        resolvedConfiguration.useBinary = loaded.vectors is BinaryVectorBuffer

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: loaded.vectors,
            graph: loaded.graph,
            idMap: resolvedIDMap,
            entryPoint: loaded.entryPoint,
            softDeletion: persistedState.softDeletion,
            metadataStore: persistedState.metadataStore
        )
        try await index.rebuildHNSWFromCurrentState()

        return index
    }

    public static func loadMmap(from url: URL) async throws -> ANNSIndex {
        let persistedState = try resolvePersistedState(for: url)
        let initialConfiguration = persistedState.configuration ?? .default
        let index = ANNSIndex(configuration: initialConfiguration)
        let loaded = try MmapIndexLoader.load(from: url, device: await index.currentDevice())
        let resolvedIDMap = resolveLoadedIDMap(
            persistedIDMap: persistedState.idMap,
            serializerIDMap: loaded.idMap
        )

        var resolvedConfiguration = persistedState.configuration ?? .default
        resolvedConfiguration.metric = loaded.metric
        resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16
        resolvedConfiguration.useBinary = loaded.isBinary

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: loaded.vectors,
            graph: loaded.graph,
            idMap: resolvedIDMap,
            entryPoint: loaded.entryPoint,
            softDeletion: persistedState.softDeletion,
            metadataStore: persistedState.metadataStore,
            isReadOnlyLoadedIndex: true,
            mmapLifetime: loaded.mmapLifetime
        )
        try await index.rebuildHNSWFromCurrentState()

        return index
    }

    public static func loadDiskBacked(from url: URL) async throws -> ANNSIndex {
        let persistedState = try resolvePersistedState(for: url)
        let initialConfiguration = persistedState.configuration ?? .default
        let index = ANNSIndex(configuration: initialConfiguration)

        let diskBacked = try DiskBackedIndexLoader.load(from: url, device: await index.currentDevice())
        let resolvedIDMap = resolveLoadedIDMap(
            persistedIDMap: persistedState.idMap,
            serializerIDMap: diskBacked.idMap
        )

        var resolvedConfiguration = persistedState.configuration ?? .default
        resolvedConfiguration.metric = diskBacked.metric
        resolvedConfiguration.useFloat16 = diskBacked.vectors.isFloat16
        resolvedConfiguration.useBinary = diskBacked.isBinary

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: diskBacked.vectors,
            graph: diskBacked.graph,
            idMap: resolvedIDMap,
            entryPoint: diskBacked.entryPoint,
            softDeletion: persistedState.softDeletion,
            metadataStore: persistedState.metadataStore,
            isReadOnlyLoadedIndex: true,
            mmapLifetime: diskBacked.mmapLifetime
        )
        try await index.rebuildHNSWFromCurrentState()

        return index
    }

    public var count: Int {
        max(0, idMap.count - softDeletion.deletedCount)
    }

    func streamingActiveExternalIDs() throws -> [String] {
        guard isBuilt, let vectors else {
            throw ANNSError.indexEmpty
        }

        var activeIDs: [String] = []
        activeIDs.reserveCapacity(max(0, idMap.count - softDeletion.deletedCount))
        for slot in 0..<vectors.count {
            let internalID = UInt32(slot)
            guard !softDeletion.isDeleted(internalID) else {
                continue
            }
            guard let externalID = idMap.externalID(for: internalID) else {
                continue
            }
            activeIDs.append(externalID)
        }
        return activeIDs
    }

    func streamingActiveRecords() throws -> (vectors: [[Float]], ids: [String]) {
        guard isBuilt, let vectors else {
            throw ANNSError.indexEmpty
        }

        var activeVectors: [[Float]] = []
        var activeIDs: [String] = []
        activeVectors.reserveCapacity(max(0, idMap.count - softDeletion.deletedCount))
        activeIDs.reserveCapacity(max(0, idMap.count - softDeletion.deletedCount))

        for slot in 0..<vectors.count {
            let internalID = UInt32(slot)
            guard !softDeletion.isDeleted(internalID) else {
                continue
            }
            guard let externalID = idMap.externalID(for: internalID) else {
                continue
            }
            activeIDs.append(externalID)
            activeVectors.append(vectors.vector(at: slot))
        }

        return (activeVectors, activeIDs)
    }

    private func batchSearchMaxConcurrency() async -> Int {
        if let context {
            return await context.queuePool.queues.count
        }
        return max(1, ProcessInfo.processInfo.activeProcessorCount)
    }

    private func currentDevice() -> MTLDevice? {
        context?.device
    }

    private func supportsGPUSearch(for vectors: any VectorStorage) -> Bool {
        !(vectors is DiskBackedVectorBuffer) && !(vectors is BinaryVectorBuffer)
    }

    private func applyLoadedState(
        configuration: IndexConfiguration,
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        softDeletion: SoftDeletion,
        metadataStore: MetadataStore = MetadataStore(),
        isReadOnlyLoadedIndex: Bool = false,
        mmapLifetime: AnyObject? = nil
    ) {
        self.configuration = configuration
        self.vectors = vectors
        self.graph = graph
        self.idMap = idMap
        self.entryPoint = entryPoint
        self.softDeletion = softDeletion
        self.metadataStore = metadataStore
        self.isBuilt = true
        self.isReadOnlyLoadedIndex = isReadOnlyLoadedIndex
        self.mmapLifetime = mmapLifetime
        self.pendingRepairIDs.removeAll()
        self.hnsw = nil
    }

    private func rebuildHNSWFromCurrentState() throws(ANNSError) {
        guard configuration.hnswConfiguration.enabled else {
            hnsw = nil
            return
        }
        guard let vectors, let graph else {
            hnsw = nil
            return
        }
        guard vectors.count > 0, graph.nodeCount > 0 else {
            hnsw = nil
            return
        }

        hnsw = try HNSWBuilder.buildLayers(
            vectors: vectors,
            graph: extractGraph(from: graph),
            nodeCount: vectors.count,
            metric: configuration.metric,
            config: configuration.hnswConfiguration
        )
    }

    private func extractVectors(from vectors: any VectorStorage) -> [[Float]] {
        (0..<vectors.count).map { vectors.vector(at: $0) }
    }

    private func extractGraph(from graph: GraphBuffer) -> [[(UInt32, Float)]] {
        (0..<graph.nodeCount).map { nodeID in
            let ids = graph.neighborIDs(of: nodeID)
            let distances = graph.neighborDistances(of: nodeID)
            return zip(ids, distances)
                .filter { $0.0 != UInt32.max }
                .map { ($0.0, $0.1) }
        }
    }

    private nonisolated static func quantizeForHamming(_ vector: [Float]) -> [Float] {
        vector.map { $0 >= 0 ? 1.0 : 0.0 }
    }

    private nonisolated static func metadataURL(for fileURL: URL) -> URL {
        URL(fileURLWithPath: fileURL.path + ".meta.json")
    }

    private nonisolated static func metadataDBURL(for fileURL: URL) -> URL {
        URL(fileURLWithPath: fileURL.path + ".meta.db")
    }

    private nonisolated static func removeLegacyMetadataSidecars(for fileURL: URL) {
        let fileManager = FileManager.default
        let metadataDB = metadataDBURL(for: fileURL)
        let sidecars = [
            metadataURL(for: fileURL),
            metadataDB,
            URL(fileURLWithPath: metadataDB.path + "-wal"),
            URL(fileURLWithPath: metadataDB.path + "-shm"),
        ]
        for sidecar in sidecars where fileManager.fileExists(atPath: sidecar.path) {
            try? fileManager.removeItem(at: sidecar)
        }
    }

    private nonisolated static func databasePath(for fileURL: URL) -> String {
        fileURL.deletingPathExtension().appendingPathExtension("db").path
    }

    private nonisolated static func replaceFile(at destination: URL, with source: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            _ = try fm.replaceItemAt(destination, withItemAt: source)
        } else {
            try fm.moveItem(at: source, to: destination)
        }
    }

    private nonisolated static func replaceSQLiteFiles(at destinationDB: URL, with sourceDB: URL) throws {
        let fm = FileManager.default

        func replace(_ source: URL, _ destination: URL) throws {
            guard fm.fileExists(atPath: source.path) else {
                return
            }
            if fm.fileExists(atPath: destination.path) {
                _ = try fm.replaceItemAt(destination, withItemAt: source)
            } else {
                try fm.moveItem(at: source, to: destination)
            }
        }

        func replaceSidecar(suffix: String) throws {
            let source = URL(fileURLWithPath: sourceDB.path + suffix)
            let destination = URL(fileURLWithPath: destinationDB.path + suffix)
            if fm.fileExists(atPath: source.path) {
                try replace(source, destination)
            } else if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
        }

        try replace(sourceDB, destinationDB)
        try replaceSidecar(suffix: "-wal")
        try replaceSidecar(suffix: "-shm")
    }

    private nonisolated static func durationNanoseconds(_ duration: Duration) -> UInt64 {
        let components = duration.components
        let seconds = components.seconds > 0 ? UInt64(components.seconds) : 0
        let attoseconds = components.attoseconds > 0 ? UInt64(components.attoseconds) : 0
        return seconds &* 1_000_000_000 &+ attoseconds / 1_000_000_000
    }

    private nonisolated static func hasFreshDatabase(at dbPath: String, forANNS annsPath: String) -> Bool {
        let fm = FileManager.default
        guard
            fm.fileExists(atPath: dbPath),
            fm.fileExists(atPath: annsPath),
            let dbAttrs = try? fm.attributesOfItem(atPath: dbPath),
            let annsAttrs = try? fm.attributesOfItem(atPath: annsPath),
            let dbDate = dbAttrs[.modificationDate] as? Date,
            let annsDate = annsAttrs[.modificationDate] as? Date
        else {
            return false
        }
        return dbDate >= annsDate
    }

    private nonisolated static func resolvePersistedState(for fileURL: URL) throws -> (
        configuration: IndexConfiguration?,
        softDeletion: SoftDeletion,
        metadataStore: MetadataStore,
        idMap: IDMap?
    ) {
        let dbPath = databasePath(for: fileURL)
        if hasFreshDatabase(at: dbPath, forANNS: fileURL.path) {
            do {
                let db = try IndexDatabase(path: dbPath)
                return (
                    configuration: try db.loadConfiguration(),
                    softDeletion: try db.loadSoftDeletion(),
                    metadataStore: try db.loadMetadataStore(),
                    idMap: try db.loadIDMap()
                )
            } catch {
                let legacy = try loadPersistedMetadataIfPresent(from: fileURL)
                return (
                    configuration: legacy?.configuration,
                    softDeletion: legacy?.softDeletion ?? SoftDeletion(),
                    metadataStore: legacy?.metadataStore ?? MetadataStore(),
                    idMap: legacy?.idMap
                )
            }
        }

        let legacy = try loadPersistedMetadataIfPresent(from: fileURL)
        return (
            configuration: legacy?.configuration,
            softDeletion: legacy?.softDeletion ?? SoftDeletion(),
            metadataStore: legacy?.metadataStore ?? MetadataStore(),
            idMap: legacy?.idMap
        )
    }

    private nonisolated static func resolveLoadedIDMap(
        persistedIDMap: IDMap?,
        serializerIDMap: IDMap
    ) -> IDMap {
        if let persistedIDMap, persistedIDMap.count == serializerIDMap.count {
            return persistedIDMap
        }
        return serializerIDMap
    }

    // MARK: - Legacy JSON Sidecar (backward compatibility only)

    /// Loads metadata from legacy sidecar formats.
    /// Used only when no fresh `.db` file exists (pre-migration indexes).
    /// Tries `.meta.db` (SQLiteStructuredStore) first, then `.meta.json`.
    /// Do not call this directly — use `resolvePersistedState(for:)`.
    private nonisolated static func loadPersistedMetadataIfPresent(from fileURL: URL) throws -> PersistedMetadata? {
        do {
            if let sqliteMeta = try SQLiteStructuredStore.load(PersistedMetadata.self, from: metadataDBURL(for: fileURL)) {
                return sqliteMeta
            }
        } catch {
            // Keep JSON fallback path for legacy metadata recovery.
        }

        let metadataURL = metadataURL(for: fileURL)
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            return nil
        }

        let data = try Data(contentsOf: metadataURL)
        return try JSONDecoder().decode(PersistedMetadata.self, from: data)
    }
}
