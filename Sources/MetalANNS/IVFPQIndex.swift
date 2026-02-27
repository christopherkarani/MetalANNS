import Foundation
import MetalANNSCore

public actor IVFPQIndex: Sendable {
    private let config: IVFPQConfiguration
    private let capacity: Int
    private let dimension: Int
    private let context: MetalContext?

    private var coarseCentroids: [[Float]] = []
    private var pq: ProductQuantizer?
    private var vectorBuffer: PQVectorBuffer?
    private var invertedLists: [[UInt32]] = []
    private var coarseAssignments: [UInt32] = []
    private var idMap = IDMap()
    private var isTrained = false

    public init(capacity: Int, dimension: Int, config: IVFPQConfiguration) throws {
        guard capacity > 0 else {
            throw ANNSError.constructionFailed("IVFPQIndex capacity must be greater than zero")
        }
        guard dimension > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: dimension)
        }
        guard config.numCentroids == 256 else {
            throw ANNSError.constructionFailed("IVFPQIndex requires numCentroids == 256 (UInt8 codes)")
        }

        self.capacity = capacity
        self.dimension = dimension
        self.config = config
        self.context = try? MetalContext()
    }

    public func train(vectors: [[Float]]) async throws {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot train IVFPQIndex with empty vectors")
        }
        guard vectors.count >= config.numCentroids else {
            throw ANNSError.constructionFailed("Need at least \(config.numCentroids) vectors for PQ training")
        }
        guard vectors.count >= config.numCoarseCentroids else {
            throw ANNSError.constructionFailed(
                "Need at least \(config.numCoarseCentroids) vectors for coarse quantizer"
            )
        }

        for vector in vectors where vector.count != dimension {
            throw ANNSError.dimensionMismatch(expected: dimension, got: vector.count)
        }

        let coarseResult = try KMeans.cluster(
            vectors: vectors,
            k: config.numCoarseCentroids,
            maxIterations: config.trainingIterations,
            metric: config.metric,
            seed: 42
        )
        let trainedCoarseCentroids = coarseResult.centroids

        var residuals: [[Float]] = []
        residuals.reserveCapacity(vectors.count)
        for (vectorIndex, vector) in vectors.enumerated() {
            let cluster = coarseResult.assignments[vectorIndex]
            let centroid = trainedCoarseCentroids[cluster]
            residuals.append(residual(vector: vector, centroid: centroid))
        }

        let trainedPQ = try ProductQuantizer.train(
            vectors: residuals,
            numSubspaces: config.numSubspaces,
            centroidsPerSubspace: config.numCentroids,
            maxIterations: config.trainingIterations
        )

        self.coarseCentroids = trainedCoarseCentroids
        self.pq = trainedPQ
        self.vectorBuffer = try PQVectorBuffer(capacity: capacity, dim: dimension, pq: trainedPQ)
        self.invertedLists = Array(repeating: [], count: trainedCoarseCentroids.count)
        self.coarseAssignments = []
        self.idMap = IDMap()
        self.isTrained = true
    }

    public func add(vectors: [[Float]], ids: [String]) async throws {
        guard isTrained, let vectorBuffer else {
            throw ANNSError.constructionFailed("IVFPQIndex must be trained before add()")
        }
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }
        guard idMap.count + vectors.count <= capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded")
        }

        var seen = Set<String>()
        for id in ids {
            if !seen.insert(id).inserted || idMap.internalID(for: id) != nil {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        for vector in vectors where vector.count != dimension {
            throw ANNSError.dimensionMismatch(expected: dimension, got: vector.count)
        }

        for (offset, vector) in vectors.enumerated() {
            guard let internalID = idMap.assign(externalID: ids[offset]) else {
                throw ANNSError.idAlreadyExists(ids[offset])
            }

            let cluster = nearestCoarseCentroid(for: vector)
            let centroid = coarseCentroids[cluster]
            let residualVector = residual(vector: vector, centroid: centroid)

            try vectorBuffer.insert(vector: residualVector, at: Int(internalID))
            invertedLists[cluster].append(internalID)
            coarseAssignments.append(UInt32(cluster))
        }
    }

    public func search(query: [Float], k: Int, nprobe: Int? = nil) async -> [SearchResult] {
        guard isTrained, let vectorBuffer else {
            return []
        }
        guard k > 0 else {
            return []
        }
        guard query.count == dimension else {
            return []
        }
        guard !coarseCentroids.isEmpty else {
            return []
        }

        let probeCount = max(1, min(nprobe ?? config.nprobe, coarseCentroids.count))
        let centroidScores = coarseCentroids.enumerated().map { (index, centroid) in
            (index, SIMDDistance.distance(query, centroid, metric: config.metric))
        }.sorted { $0.1 < $1.1 }

        var merged: [SearchResult] = []
        merged.reserveCapacity(probeCount * max(k, 16))

        for (clusterIndex, _) in centroidScores.prefix(probeCount) {
            let queryResidual = residual(vector: query, centroid: coarseCentroids[clusterIndex])
            for internalID in invertedLists[clusterIndex] {
                let score = vectorBuffer.approximateDistance(
                    query: queryResidual,
                    to: internalID,
                    metric: config.metric
                )
                guard let externalID = idMap.externalID(for: internalID) else {
                    continue
                }
                merged.append(SearchResult(id: externalID, score: score, internalID: internalID))
            }
        }

        merged.sort { $0.score < $1.score }
        if merged.count > k {
            merged.removeSubrange(k...)
        }
        return merged
    }

    public var count: Int {
        idMap.count
    }

    public func estimatedVectorCodeBytes() -> Int {
        vectorBuffer?.compressedCodeBytes ?? 0
    }

    public func estimatedMemoryBytes() -> Int {
        let coarseBytes = coarseCentroids.count * dimension * MemoryLayout<Float>.stride
        let codebookBytes: Int = if let pq {
            pq.numSubspaces * pq.centroidsPerSubspace * pq.subspaceDimension * MemoryLayout<Float>.stride
        } else {
            0
        }
        let vectorCodeBytes = vectorBuffer?.compressedCodeBytes ?? 0
        let invertedListBytes = invertedLists.reduce(0) { partial, list in
            partial + (list.count * MemoryLayout<UInt32>.stride)
        }
        let assignmentBytes = coarseAssignments.count * MemoryLayout<UInt32>.stride
        return coarseBytes + codebookBytes + vectorCodeBytes + invertedListBytes + assignmentBytes
    }

    private func nearestCoarseCentroid(for vector: [Float]) -> Int {
        var bestIndex = 0
        var bestDistance = Float.greatestFiniteMagnitude

        for (index, centroid) in coarseCentroids.enumerated() {
            let distance = SIMDDistance.distance(vector, centroid, metric: config.metric)
            if distance < bestDistance {
                bestDistance = distance
                bestIndex = index
            }
        }
        return bestIndex
    }

    private func residual(vector: [Float], centroid: [Float]) -> [Float] {
        zip(vector, centroid).map { $0 - $1 }
    }
}
