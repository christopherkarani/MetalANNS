import Foundation
import Metal
import MetalANNSCore

/// Vectors handed to `build`. Either form is copied exactly once, into the
/// index's (GPU-shared) vector storage.
internal enum BuildInput: Sendable {
    case rows([[Float]])
    /// Row-major `count × dimension` buffer.
    case flat([Float], dimension: Int)

    var count: Int {
        switch self {
        case .rows(let rows): rows.count
        case .flat(let values, let dimension): dimension > 0 ? values.count / dimension : 0
        }
    }

    var dimension: Int {
        switch self {
        case .rows(let rows): rows.first?.count ?? 0
        case .flat(_, let dimension): dimension
        }
    }

    func validate(idCount: Int) throws {
        switch self {
        case .rows(let rows):
            guard !rows.isEmpty else {
                throw ANNSError.constructionFailed("Cannot build index with empty vectors")
            }
            guard rows.count == idCount else {
                throw ANNSError.constructionFailed("Vector and ID counts do not match")
            }
            let dim = rows[0].count
            guard dim > 0 else {
                throw ANNSError.dimensionMismatch(expected: 1, got: 0)
            }
            for row in rows where row.count != dim {
                throw ANNSError.dimensionMismatch(expected: dim, got: row.count)
            }
            try VectorValidation.requireFinite(rows: rows)
        case .flat(let values, let dimension):
            guard dimension > 0 else {
                throw ANNSError.dimensionMismatch(expected: 1, got: dimension)
            }
            guard !values.isEmpty else {
                throw ANNSError.constructionFailed("Cannot build index with empty vectors")
            }
            guard values.count % dimension == 0 else {
                throw ANNSError.constructionFailed(
                    "Flat vector buffer length \(values.count) is not a multiple of dimension \(dimension)"
                )
            }
            guard values.count / dimension == idCount else {
                throw ANNSError.constructionFailed("Vector and ID counts do not match")
            }
            try VectorValidation.requireFinite(flat: values, dimension: dimension)
        }
    }

    func write(into storage: any VectorStorage) throws {
        switch self {
        case .rows(let rows):
            try storage.batchInsert(vectors: rows, startingAt: 0)
        case .flat(let values, let dimension):
            if let floatStorage = storage as? VectorBuffer {
                try floatStorage.batchInsert(flat: values, startingAt: 0)
            } else {
                for row in 0..<(values.count / dimension) {
                    let start = row * dimension
                    try storage.insert(vector: Array(values[start..<(start + dimension)]), at: row)
                }
            }
        }
    }

    /// Nested rows for the CPU construction paths. Only materialized for the
    /// flat form on CPU builds (small graphs, or binary indexes).
    func rows() -> [[Float]] {
        switch self {
        case .rows(let rows):
            return rows
        case .flat(let values, let dimension):
            return (0..<(values.count / dimension)).map { row in
                Array(values[(row * dimension)..<((row + 1) * dimension)])
            }
        }
    }
}

extension GraphIndex {
    /// Builds the index from nested rows.
    ///
    /// Any count ≥ 1 is accepted. When the count does not exceed
    /// `configuration.degree`, the exact neighbour graph is built directly
    /// (NN-Descent needs more nodes than the degree); it fills in as vectors are
    /// inserted. Throws `ANNSError.nonFiniteVector` for NaN/±inf input and
    /// `CancellationError` if the calling task is cancelled.
    public func build(vectors inputVectors: [[Float]], ids: [String]) async throws {
        try await build(.rows(inputVectors), ids: ids)
    }

    /// Builds the index from one row-major buffer of `ids.count × dimension`
    /// floats. Avoids materializing `[[Float]]`: the buffer is copied once,
    /// straight into the index's vector storage.
    public func build(flatVectors: [Float], dimension: Int, ids: [String]) async throws {
        try await build(.flat(flatVectors, dimension: dimension), ids: ids)
    }

    internal func build(_ input: BuildInput, ids: [String]) async throws {
        await beginWrite()
        defer { endWrite() }

        guard !isBuilt else {
            throw ANNSError.constructionFailed("Index is already built. Create a new GraphIndex to rebuild.")
        }
        try input.validate(idCount: ids.count)

        var seenIDs = Set<String>()
        for id in ids {
            if !seenIDs.insert(id).inserted {
                throw ANNSError.idAlreadyExists(id)
            }
        }
        guard configuration.degree > 0 else {
            throw ANNSError.constructionFailed("Degree must be greater than zero")
        }
        if configuration.metric == .hamming, !configuration.useBinary {
            throw ANNSError.constructionFailed("metric .hamming requires useBinary == true")
        }
        try Task.checkCancellation()

        let nodeCount = input.count
        let dim = input.dimension
        // Exactly the build size: inserts grow storage geometrically, so build
        // does not pre-pay headroom in peak memory.
        let capacity = nodeCount
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
        try input.write(into: vectorBuffer)
        vectorBuffer.setCount(nodeCount)

        var builtIDMap = IDMap()
        for id in ids {
            guard builtIDMap.assign(externalID: id) != nil else {
                throw ANNSError.idAlreadyExists(id)
            }
        }
        try Task.checkCancellation()

        let builtEntryPoint: UInt32
        if nodeCount <= configuration.degree {
            let rows = configuration.useBinary ? input.rows().map(Self.quantizeForHamming) : input.rows()
            try ExactGraphBuilder.build(vectors: rows, into: graphBuffer, metric: configuration.metric)
            builtEntryPoint = 0
        } else {
            builtEntryPoint = try await buildNNDescentGraph(
                input: input,
                vectors: vectorBuffer,
                graph: graphBuffer,
                nodeCount: nodeCount
            )
            try GraphPruner.prune(
                graph: graphBuffer,
                vectors: vectorBuffer,
                nodeCount: nodeCount,
                metric: configuration.metric
            )
        }
        try Task.checkCancellation()

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

        if configuration.searchMode == .fast,
            configuration.metric != .hamming,
            let prepared = vectorBuffer as? VectorBuffer
        {
            IVFFlatSearch.prepare(
                vectors: prepared,
                metric: configuration.metric,
                nlist: configuration.ivfListCount
            )
        }
    }

    private func buildNNDescentGraph(
        input: BuildInput,
        vectors vectorBuffer: any VectorStorage,
        graph graphBuffer: GraphBuffer,
        nodeCount: Int
    ) async throws -> UInt32 {
        if let context, shouldUseGPUConstruction(nodeCount: nodeCount) {
            try await NNDescentGPU.build(
                context: context,
                vectors: vectorBuffer,
                graph: graphBuffer,
                nodeCount: nodeCount,
                metric: configuration.metric,
                maxIterations: configuration.maxIterations,
                convergenceThreshold: configuration.convergenceThreshold
            )
            return 0
        }

        let cpuVectors = configuration.useBinary ? input.rows().map(Self.quantizeForHamming) : input.rows()
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
        graphBuffer.setCount(nodeCount)
        return cpuResult.entryPoint
    }
}
