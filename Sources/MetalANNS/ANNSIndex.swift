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
    private var pendingRepairIDs: [UInt32] = []
    private var hnsw: HNSWLayers?
    private var quantizedHNSW: QuantizedHNSWLayers?

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
        self.quantizedHNSW = nil
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
        self.quantizedHNSW = nil
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
        self.pendingRepairIDs.removeAll()
        self.hnsw = nil
        self.quantizedHNSW = nil
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
        hnsw = nil
        quantizedHNSW = nil

        let repairConfig = configuration.repairConfiguration
        if repairConfig.enabled && repairConfig.repairInterval > 0 {
            pendingRepairIDs.append(UInt32(slot))
            if pendingRepairIDs.count >= repairConfig.repairInterval {
                try triggerRepair()
            }
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
        hnsw = nil
        quantizedHNSW = nil

        let repairConfig = configuration.repairConfiguration
        if repairConfig.enabled {
            var idsToRepair = pendingRepairIDs
            idsToRepair.reserveCapacity(idsToRepair.count + slots.count)
            idsToRepair.append(contentsOf: slots.map(UInt32.init))
            pendingRepairIDs.removeAll(keepingCapacity: true)

            if !idsToRepair.isEmpty {
                _ = try GraphRepairer.repair(
                    recentIDs: idsToRepair,
                    vectors: vectorStorage,
                    graph: graph,
                    config: repairConfig,
                    metric: configuration.metric
                )
            }
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

    private func triggerRepair() throws(ANNSError) {
        guard let vectors, let graph else {
            return
        }
        guard !pendingRepairIDs.isEmpty else {
            return
        }

        let idsToRepair = pendingRepairIDs
        pendingRepairIDs.removeAll(keepingCapacity: true)

        _ = try GraphRepairer.repair(
            recentIDs: idsToRepair,
            vectors: vectors,
            graph: graph,
            config: configuration.repairConfiguration,
            metric: configuration.metric
        )
        hnsw = nil
        quantizedHNSW = nil
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
            useFloat16: configuration.useFloat16
        )

        var remapping: [UInt32: UInt32] = [:]
        remapping.reserveCapacity(result.idMap.count)
        for oldIndex in 0..<vectors.count {
            let oldID = UInt32(oldIndex)
            if softDeletion.isDeleted(oldID) {
                continue
            }
            guard let externalID = idMap.externalID(for: oldID),
                  let newID = result.idMap.internalID(for: externalID) else {
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
        self.quantizedHNSW = nil
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
        if let context, supportsGPUSearch(for: vectors) {
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
            if configuration.hnswConfiguration.enabled, hnsw == nil {
                try rebuildHNSWFromCurrentState()
            }

            let extractedVectors = extractVectors(from: vectors)
            let extractedGraph = extractGraph(from: graph)

            if let quantizedHNSW {
                rawResults = try await QuantizedHNSWSearchCPU.search(
                    query: query,
                    vectors: extractedVectors,
                    hnsw: quantizedHNSW,
                    baseGraph: extractedGraph,
                    k: max(1, effectiveK),
                    ef: max(1, effectiveEf),
                    metric: searchMetric
                )
            } else if let hnsw {
                rawResults = try await HNSWSearchCPU.search(
                    query: query,
                    vectors: extractedVectors,
                    hnsw: hnsw,
                    baseGraph: extractedGraph,
                    k: max(1, effectiveK),
                    ef: max(1, effectiveEf),
                    metric: searchMetric
                )
            } else {
                rawResults = try await BeamSearchCPU.search(
                    query: query,
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
        guard maxDistance > 0 else {
            return []
        }
        guard limit > 0 else {
            return []
        }

        let searchMetric = metric ?? configuration.metric
        let deletedCount = softDeletion.deletedCount
        let searchK = min(vectors.count, limit + deletedCount)
        let searchEf = min(vectors.count, max(configuration.efSearch, searchK * 2))

        let rawResults: [SearchResult]
        if let context, supportsGPUSearch(for: vectors) {
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
            if configuration.hnswConfiguration.enabled, hnsw == nil {
                try rebuildHNSWFromCurrentState()
            }

            let extractedVectors = extractVectors(from: vectors)
            let extractedGraph = extractGraph(from: graph)

            if let quantizedHNSW {
                rawResults = try await QuantizedHNSWSearchCPU.search(
                    query: query,
                    vectors: extractedVectors,
                    hnsw: quantizedHNSW,
                    baseGraph: extractedGraph,
                    k: max(1, searchK),
                    ef: max(1, searchEf),
                    metric: searchMetric
                )
            } else if let hnsw {
                rawResults = try await HNSWSearchCPU.search(
                    query: query,
                    vectors: extractedVectors,
                    hnsw: hnsw,
                    baseGraph: extractedGraph,
                    k: max(1, searchK),
                    ef: max(1, searchEf),
                    metric: searchMetric
                )
            } else {
                rawResults = try await BeamSearchCPU.search(
                    query: query,
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

    func batchSearchMaxConcurrencyForTesting() async -> Int {
        await batchSearchMaxConcurrency()
    }

    func configurationForTesting() -> IndexConfiguration {
        configuration
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
        try await index.rebuildHNSWFromCurrentState()

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
        try await index.rebuildHNSWFromCurrentState()

        return index
    }

    public static func loadDiskBacked(from url: URL) async throws -> ANNSIndex {
        let persistedMetadata = try loadPersistedMetadataIfPresent(from: url)
        let initialConfiguration = persistedMetadata?.configuration ?? .default
        let index = ANNSIndex(configuration: initialConfiguration)

        let diskBacked = try DiskBackedIndexLoader.load(from: url, device: await index.currentDevice())

        var resolvedConfiguration = persistedMetadata?.configuration ?? .default
        resolvedConfiguration.metric = diskBacked.metric
        resolvedConfiguration.useFloat16 = diskBacked.vectors.isFloat16

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: diskBacked.vectors,
            graph: diskBacked.graph,
            idMap: diskBacked.idMap,
            entryPoint: diskBacked.entryPoint,
            softDeletion: persistedMetadata?.softDeletion ?? SoftDeletion(),
            metadataStore: persistedMetadata?.metadataStore ?? MetadataStore(),
            isReadOnlyLoadedIndex: true,
            mmapLifetime: diskBacked.mmapLifetime
        )
        try await index.rebuildHNSWFromCurrentState()

        return index
    }

    public var count: Int {
        max(0, idMap.count - softDeletion.deletedCount)
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
        !(vectors is DiskBackedVectorBuffer)
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
        self.quantizedHNSW = nil
    }

    private func rebuildHNSWFromCurrentState() throws(ANNSError) {
        guard configuration.hnswConfiguration.enabled else {
            hnsw = nil
            quantizedHNSW = nil
            return
        }
        guard let vectors, let graph else {
            hnsw = nil
            quantizedHNSW = nil
            return
        }
        if context != nil {
            hnsw = nil
            quantizedHNSW = nil
            return
        }
        guard vectors.count > 0, graph.nodeCount > 0 else {
            hnsw = nil
            quantizedHNSW = nil
            return
        }

        let extractedGraph = extractGraph(from: graph)
        let builtHNSW = try HNSWBuilder.buildLayers(
            vectors: vectors,
            graph: extractedGraph,
            nodeCount: vectors.count,
            metric: configuration.metric,
            config: configuration.hnswConfiguration
        )
        hnsw = builtHNSW

        if configuration.quantizedHNSWConfiguration.useQuantizedEdges,
           builtHNSW.maxLayer > 0 {
            let extractedVectors = extractVectors(from: vectors)
            quantizedHNSW = try? QuantizedHNSWBuilder.build(
                from: builtHNSW,
                vectors: extractedVectors,
                config: configuration.quantizedHNSWConfiguration,
                metric: configuration.metric
            )
        } else {
            quantizedHNSW = nil
        }
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
