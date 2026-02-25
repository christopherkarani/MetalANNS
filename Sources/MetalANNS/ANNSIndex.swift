import Foundation
import Metal
import MetalANNSCore

public actor ANNSIndex {
    private struct PersistedMetadata: Codable, Sendable {
        let configuration: IndexConfiguration
        let softDeletion: SoftDeletion
        let metadataStore: MetadataStore?
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

        let capacity = max(2, inputVectors.count * 2)
        let device = context?.device
        let vectorBuffer: any VectorStorage
        if configuration.useFloat16 {
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
        if let context {
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
                vectors: inputVectors,
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

        let nextInternalID = idMap.count
        guard nextInternalID < vectors.capacity, nextInternalID < graph.capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
        }

        guard let assignedID = idMap.assign(externalID: id) else {
            throw ANNSError.idAlreadyExists(id)
        }

        let slot = Int(assignedID)
        try vectors.insert(vector: vector, at: slot)
        if vectors.count < slot + 1 {
            vectors.setCount(slot + 1)
        }

        try IncrementalBuilder.insert(
            vector: vector,
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

        let startSlot = idMap.count
        guard startSlot + vectors.count <= vectorStorage.capacity,
              startSlot + vectors.count <= graph.capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
        }

        var slots: [Int] = []
        slots.reserveCapacity(ids.count)
        for id in ids {
            guard let assignedID = idMap.assign(externalID: id) else {
                throw ANNSError.idAlreadyExists(id)
            }
            slots.append(Int(assignedID))
        }

        for (offset, vector) in vectors.enumerated() {
            try vectorStorage.insert(vector: vector, at: slots[offset])
        }
        let newMaxCount = (slots.last ?? 0) + 1
        if vectorStorage.count < newMaxCount {
            vectorStorage.setCount(newMaxCount)
        }

        try BatchIncrementalBuilder.batchInsert(
            vectors: vectors,
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
    }

    public func delete(id: String) throws {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        softDeletion.markDeleted(internalID)
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

        let previousIDMap = idMap
        let previousMetadataStore = metadataStore

        let result = try await IndexCompactor.compact(
            vectors: vectors,
            graph: graph,
            idMap: previousIDMap,
            softDeletion: softDeletion,
            metric: configuration.metric,
            degree: configuration.degree,
            context: context,
            maxIterations: configuration.maxIterations,
            convergenceThreshold: configuration.convergenceThreshold,
            useFloat16: configuration.useFloat16
        )

        var internalIDMapping: [UInt32: UInt32] = [:]
        internalIDMapping.reserveCapacity(previousIDMap.count)
        for oldIndex in 0..<previousIDMap.count {
            let oldInternalID = UInt32(oldIndex)
            guard let externalID = previousIDMap.externalID(for: oldInternalID) else {
                continue
            }
            guard let newInternalID = result.idMap.internalID(for: externalID) else {
                continue
            }
            internalIDMapping[oldInternalID] = newInternalID
        }

        self.vectors = result.vectors
        self.graph = result.graph
        self.idMap = result.idMap
        self.entryPoint = result.entryPoint
        self.softDeletion = SoftDeletion()
        self.metadataStore = previousMetadataStore.remapped(using: internalIDMapping)
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
    }

    public func setMetadata(_ column: String, value: String, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, stringValue: value, for: internalID)
    }

    public func setMetadata(_ column: String, value: Float, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, floatValue: value, for: internalID)
    }

    public func setMetadata(_ column: String, value: Int64, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
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

        let deletedCount = softDeletion.deletedCount
        let hasFilter = filter != nil
        let effectiveK: Int
        if hasFilter {
            effectiveK = min(vectors.count, k * 4 + deletedCount)
        } else {
            effectiveK = min(vectors.count, k + deletedCount)
        }
        let effectiveEf = max(configuration.efSearch, effectiveK)
        let searchMetric = metric ?? configuration.metric

        let rawResults: [SearchResult]
        if let context {
            rawResults = try await FullGPUSearch.search(
                context: context,
                query: query,
                vectors: vectors,
                graph: graph,
                entryPoint: Int(entryPoint),
                k: max(1, effectiveK),
                ef: max(1, effectiveEf),
                metric: searchMetric
            )
        } else {
            rawResults = try await BeamSearchCPU.search(
                query: query,
                vectors: extractVectors(from: vectors),
                graph: extractGraph(from: graph),
                entryPoint: Int(entryPoint),
                k: max(1, effectiveK),
                ef: max(1, effectiveEf),
                metric: searchMetric
            )
        }

        var filtered = softDeletion.filterResults(rawResults)
        if let filter {
            filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
        }
        let mapped = filtered.compactMap { result -> SearchResult? in
            guard let externalID = idMap.externalID(for: result.internalID) else {
                return nil
            }
            return SearchResult(id: externalID, score: result.score, internalID: result.internalID)
        }
        return Array(mapped.prefix(k))
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
        guard limit > 0 else {
            return []
        }

        let deletedCount = softDeletion.deletedCount
        let hasFilter = filter != nil
        let searchK: Int
        if hasFilter {
            searchK = min(vectors.count, limit * 4 + deletedCount)
        } else {
            searchK = min(vectors.count, limit + deletedCount)
        }
        let searchEf = min(vectors.count, max(configuration.efSearch, searchK * 2))
        let searchMetric = metric ?? configuration.metric

        let rawResults: [SearchResult]
        if let context {
            rawResults = try await FullGPUSearch.search(
                context: context,
                query: query,
                vectors: vectors,
                graph: graph,
                entryPoint: Int(entryPoint),
                k: max(1, searchK),
                ef: max(1, searchEf),
                metric: searchMetric
            )
        } else {
            rawResults = try await BeamSearchCPU.search(
                query: query,
                vectors: extractVectors(from: vectors),
                graph: extractGraph(from: graph),
                entryPoint: Int(entryPoint),
                k: max(1, searchK),
                ef: max(1, searchEf),
                metric: searchMetric
            )
        }

        var filtered = softDeletion.filterResults(rawResults)
        if let filter {
            filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
        }
        let withinRange = filtered.filter { $0.score <= maxDistance }
        let mapped = withinRange.compactMap { result -> SearchResult? in
            guard let externalID = idMap.externalID(for: result.internalID) else {
                return nil
            }
            return SearchResult(id: externalID, score: result.score, internalID: result.internalID)
        }
        return Array(mapped.prefix(limit))
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

        let maxConcurrency = context != nil ? 4 : max(1, ProcessInfo.processInfo.activeProcessorCount)

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

    public func save(to url: URL) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }

        try IndexSerializer.save(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            entryPoint: entryPoint,
            metric: configuration.metric,
            to: url
        )

        let metadata = PersistedMetadata(
            configuration: configuration,
            softDeletion: softDeletion,
            metadataStore: metadataStore
        )
        let metadataData = try JSONEncoder().encode(metadata)
        try metadataData.write(to: Self.metadataURL(for: url), options: .atomic)
    }

    public func saveMmapCompatible(to url: URL) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }

        try IndexSerializer.saveMmapCompatible(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            entryPoint: entryPoint,
            metric: configuration.metric,
            to: url
        )

        let metadata = PersistedMetadata(
            configuration: configuration,
            softDeletion: softDeletion,
            metadataStore: metadataStore
        )
        let metadataData = try JSONEncoder().encode(metadata)
        try metadataData.write(to: Self.metadataURL(for: url), options: .atomic)
    }

    public static func load(from url: URL) async throws -> ANNSIndex {
        let persistedMetadata = try loadPersistedMetadataIfPresent(from: url)
        let initialConfiguration = persistedMetadata?.configuration ?? .default
        let index = ANNSIndex(configuration: initialConfiguration)

        let loaded = try IndexSerializer.load(from: url, device: await index.currentDevice())

        var resolvedConfiguration = persistedMetadata?.configuration ?? .default
        resolvedConfiguration.metric = loaded.metric
        resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: loaded.vectors,
            graph: loaded.graph,
            idMap: loaded.idMap,
            entryPoint: loaded.entryPoint,
            softDeletion: persistedMetadata?.softDeletion ?? SoftDeletion(),
            metadataStore: persistedMetadata?.metadataStore ?? MetadataStore()
        )

        return index
    }

    public static func loadMmap(from url: URL) async throws -> ANNSIndex {
        let persistedMetadata = try loadPersistedMetadataIfPresent(from: url)
        let initialConfiguration = persistedMetadata?.configuration ?? .default
        let index = ANNSIndex(configuration: initialConfiguration)
        let loaded = try MmapIndexLoader.load(from: url, device: await index.currentDevice())

        var resolvedConfiguration = persistedMetadata?.configuration ?? .default
        resolvedConfiguration.metric = loaded.metric
        resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: loaded.vectors,
            graph: loaded.graph,
            idMap: loaded.idMap,
            entryPoint: loaded.entryPoint,
            softDeletion: persistedMetadata?.softDeletion ?? SoftDeletion(),
            metadataStore: persistedMetadata?.metadataStore ?? MetadataStore(),
            isReadOnlyLoadedIndex: true,
            mmapLifetime: loaded.mmapLifetime
        )

        return index
    }

    public var count: Int {
        max(0, idMap.count - softDeletion.deletedCount)
    }

    private func currentDevice() -> MTLDevice? {
        context?.device
    }

    private func applyLoadedState(
        configuration: IndexConfiguration,
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        softDeletion: SoftDeletion,
        metadataStore: MetadataStore,
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

    private nonisolated static func metadataURL(for fileURL: URL) -> URL {
        URL(fileURLWithPath: fileURL.path + ".meta.json")
    }

    private nonisolated static func loadPersistedMetadataIfPresent(from fileURL: URL) throws -> PersistedMetadata? {
        let metadataURL = metadataURL(for: fileURL)
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            return nil
        }

        let data = try Data(contentsOf: metadataURL)
        return try JSONDecoder().decode(PersistedMetadata.self, from: data)
    }
}
