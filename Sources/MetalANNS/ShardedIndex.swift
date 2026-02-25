import Foundation
import MetalANNSCore

public actor ShardedIndex {
    private let numShards: Int
    private let nprobe: Int
    private let configuration: IndexConfiguration

    private var centroids: [[Float]] = []
    private var shards: [ANNSIndex] = []
    private var isBuilt = false

    public init(
        numShards: Int = 16,
        nprobe: Int = 4,
        configuration: IndexConfiguration = .default
    ) {
        let normalizedShards = max(1, numShards)
        self.numShards = normalizedShards
        self.nprobe = max(1, min(nprobe, normalizedShards))
        self.configuration = configuration
    }

    public func build(vectors: [[Float]], ids: [String]) async throws(ANNSError) {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot build sharded index with empty vectors")
        }
        guard vectors.count > 1 else {
            throw ANNSError.constructionFailed("Sharded index requires at least 2 vectors")
        }
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }

        let dim = vectors[0].count
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }

        for vector in vectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }

        var seenIDs = Set<String>()
        for id in ids {
            if !seenIDs.insert(id).inserted {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        let effectiveShards = min(numShards, vectors.count)

        let kmeans = try KMeans.cluster(
            vectors: vectors,
            k: effectiveShards,
            maxIterations: 20,
            metric: configuration.metric,
            seed: 42
        )

        var shardVectors = Array(repeating: [[Float]](), count: effectiveShards)
        var shardIDs = Array(repeating: [String](), count: effectiveShards)

        for i in 0..<vectors.count {
            let shardIndex = kmeans.assignments[i]
            shardVectors[shardIndex].append(vectors[i])
            shardIDs[shardIndex].append(ids[i])
        }

        let eligibleTargets = shardVectors.indices.filter { shardVectors[$0].count >= 2 }
        if !eligibleTargets.isEmpty {
            for shardIndex in shardVectors.indices where shardVectors[shardIndex].count == 1 {
                let loneVector = shardVectors[shardIndex][0]
                var targetShard = eligibleTargets[0]
                var bestDistance = SIMDDistance.distance(
                    loneVector,
                    kmeans.centroids[targetShard],
                    metric: configuration.metric
                )

                for candidate in eligibleTargets.dropFirst() {
                    let distance = SIMDDistance.distance(
                        loneVector,
                        kmeans.centroids[candidate],
                        metric: configuration.metric
                    )
                    if distance < bestDistance {
                        bestDistance = distance
                        targetShard = candidate
                    }
                }

                shardVectors[targetShard].append(loneVector)
                shardIDs[targetShard].append(shardIDs[shardIndex][0])
                shardVectors[shardIndex].removeAll(keepingCapacity: true)
                shardIDs[shardIndex].removeAll(keepingCapacity: true)
            }
        }

        var builtShards: [ANNSIndex] = []
        var builtCentroids: [[Float]] = []

        for shardIndex in 0..<effectiveShards {
            guard !shardVectors[shardIndex].isEmpty else {
                continue
            }

            var shardConfiguration = configuration
            shardConfiguration.degree = min(configuration.degree, max(1, shardVectors[shardIndex].count - 1))

            let shard = ANNSIndex(configuration: shardConfiguration)
            try await shard.build(vectors: shardVectors[shardIndex], ids: shardIDs[shardIndex])
            builtShards.append(shard)
            builtCentroids.append(kmeans.centroids[shardIndex])
        }

        self.shards = builtShards
        self.centroids = builtCentroids
        self.isBuilt = true
    }

    @concurrent
    public func search(
        query: [Float],
        k: Int,
        filter: SearchFilter? = nil,
        metric: Metric? = nil
    ) async throws(ANNSError) -> [SearchResult] {
        let snapshot = await searchSnapshot()

        guard snapshot.isBuilt, !snapshot.shards.isEmpty, !snapshot.centroids.isEmpty else {
            throw ANNSError.indexEmpty
        }
        guard k > 0 else {
            return []
        }

        guard let firstCentroid = snapshot.centroids.first else {
            throw ANNSError.indexEmpty
        }
        guard query.count == firstCentroid.count else {
            throw ANNSError.dimensionMismatch(expected: firstCentroid.count, got: query.count)
        }

        let searchMetric = metric ?? snapshot.metric

        let centroidDistances = snapshot.centroids.enumerated().map { (index, centroid) in
            (index, SIMDDistance.distance(query, centroid, metric: searchMetric))
        }.sorted { $0.1 < $1.1 }

        let probeCount = min(snapshot.nprobe, snapshot.shards.count)
        let probeIndices = centroidDistances.prefix(probeCount).map { $0.0 }

        var mergedResults: [SearchResult] = []
        mergedResults.reserveCapacity(probeCount * k)

        for shardIndex in probeIndices {
            let shardResults = try await snapshot.shards[shardIndex].search(
                query: query,
                k: k,
                filter: filter,
                metric: metric
            )
            mergedResults.append(contentsOf: shardResults)
        }

        mergedResults.sort { $0.score < $1.score }
        return Array(mergedResults.prefix(k))
    }

    @concurrent
    public var count: Int {
        get async {
            let shards = await shardsSnapshot()
            var total = 0
            for shard in shards {
                total += await shard.count
            }
            return total
        }
    }

    func shardSizes() async -> [Int] {
        var sizes: [Int] = []
        sizes.reserveCapacity(shards.count)

        for shard in shards {
            sizes.append(await shard.count)
        }

        return sizes
    }

    private func searchSnapshot() -> (
        isBuilt: Bool,
        shards: [ANNSIndex],
        centroids: [[Float]],
        nprobe: Int,
        metric: Metric
    ) {
        (
            isBuilt: isBuilt,
            shards: shards,
            centroids: centroids,
            nprobe: nprobe,
            metric: configuration.metric
        )
    }

    private func shardsSnapshot() -> [ANNSIndex] {
        shards
    }
}
