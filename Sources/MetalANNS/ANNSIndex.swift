import Foundation
import Metal
import MetalANNSCore

public actor ANNSIndex {
    private struct PersistedMetadata: Codable, Sendable {
        let configuration: IndexConfiguration
        let softDeletion: SoftDeletion
    }

    private var configuration: IndexConfiguration
    private var context: MetalContext?
    private var vectors: VectorBuffer?
    private var graph: GraphBuffer?
    private var idMap: IDMap
    private var softDeletion: SoftDeletion
    private var entryPoint: UInt32
    private var isBuilt: Bool

    public init(configuration: IndexConfiguration = .default) {
        self.configuration = configuration
        self.context = try? MetalContext()
        self.vectors = nil
        self.graph = nil
        self.idMap = IDMap()
        self.softDeletion = SoftDeletion()
        self.entryPoint = 0
        self.isBuilt = false
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
        let vectorBuffer = try VectorBuffer(capacity: capacity, dim: dim, device: device)
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

        self.vectors = vectorBuffer
        self.graph = graphBuffer
        self.idMap = builtIDMap
        self.softDeletion = SoftDeletion()
        self.entryPoint = builtEntryPoint
        self.isBuilt = true
    }

    public func insert(_ vector: [Float], id: String) async throws {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
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

    public func delete(id: String) throws {
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        softDeletion.markDeleted(internalID)
    }

    public func search(query: [Float], k: Int) async throws -> [SearchResult] {
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
        let effectiveK = min(vectors.count, k + deletedCount)
        let effectiveEf = max(configuration.efSearch, effectiveK)

        let rawResults: [SearchResult]
        if let context {
            rawResults = try await SearchGPU.search(
                context: context,
                query: query,
                vectors: vectors,
                graph: graph,
                entryPoint: Int(entryPoint),
                k: max(1, effectiveK),
                ef: max(1, effectiveEf),
                metric: configuration.metric
            )
        } else {
            rawResults = try await BeamSearchCPU.search(
                query: query,
                vectors: extractVectors(from: vectors),
                graph: extractGraph(from: graph),
                entryPoint: Int(entryPoint),
                k: max(1, effectiveK),
                ef: max(1, effectiveEf),
                metric: configuration.metric
            )
        }

        let filtered = softDeletion.filterResults(rawResults)
        let mapped = filtered.compactMap { result -> SearchResult? in
            guard let externalID = idMap.externalID(for: result.internalID) else {
                return nil
            }
            return SearchResult(id: externalID, score: result.score, internalID: result.internalID)
        }
        return Array(mapped.prefix(k))
    }

    public func batchSearch(queries: [[Float]], k: Int) async throws -> [[SearchResult]] {
        var allResults: [[SearchResult]] = []
        allResults.reserveCapacity(queries.count)
        for query in queries {
            let results = try await search(query: query, k: k)
            allResults.append(results)
        }
        return allResults
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

        let metadata = PersistedMetadata(configuration: configuration, softDeletion: softDeletion)
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

        await index.applyLoadedState(
            configuration: resolvedConfiguration,
            vectors: loaded.vectors,
            graph: loaded.graph,
            idMap: loaded.idMap,
            entryPoint: loaded.entryPoint,
            softDeletion: persistedMetadata?.softDeletion ?? SoftDeletion()
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
        vectors: VectorBuffer,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        softDeletion: SoftDeletion
    ) {
        self.configuration = configuration
        self.vectors = vectors
        self.graph = graph
        self.idMap = idMap
        self.entryPoint = entryPoint
        self.softDeletion = softDeletion
        self.isBuilt = true
    }

    private func extractVectors(from vectors: VectorBuffer) -> [[Float]] {
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
