import Foundation

public enum KMeans {
    public struct Result {
        public let centroids: [[Float]]
        public let assignments: [Int]

        public init(centroids: [[Float]], assignments: [Int]) {
            self.centroids = centroids
            self.assignments = assignments
        }
    }

    public static func cluster(
        vectors: [[Float]],
        k: Int,
        maxIterations: Int = 20,
        metric: Metric = .l2,
        seed: UInt64 = 42
    ) throws(ANNSError) -> Result {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot cluster empty vectors")
        }
        guard k > 0, k <= vectors.count else {
            throw ANNSError.constructionFailed("k must be between 1 and vector count")
        }

        let dim = vectors[0].count
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }

        for vector in vectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }

        var centroids = initializeCentroids(vectors: vectors, k: k, metric: metric, seed: seed)
        var assignments = [Int](repeating: 0, count: vectors.count)

        for _ in 0..<maxIterations {
            var changed = false

            for i in 0..<vectors.count {
                var bestCluster = 0
                var bestDistance = Float.greatestFiniteMagnitude

                for c in 0..<k {
                    let distance = SIMDDistance.distance(vectors[i], centroids[c], metric: metric)
                    if distance < bestDistance {
                        bestDistance = distance
                        bestCluster = c
                    }
                }

                if assignments[i] != bestCluster {
                    assignments[i] = bestCluster
                    changed = true
                }
            }

            if !changed {
                break
            }

            var sums = [[Float]](repeating: [Float](repeating: 0, count: dim), count: k)
            var counts = [Int](repeating: 0, count: k)

            for i in 0..<vectors.count {
                let cluster = assignments[i]
                counts[cluster] += 1
                for d in 0..<dim {
                    sums[cluster][d] += vectors[i][d]
                }
            }

            for c in 0..<k {
                guard counts[c] > 0 else {
                    continue
                }
                let scale = 1.0 / Float(counts[c])
                for d in 0..<dim {
                    centroids[c][d] = sums[c][d] * scale
                }
            }
        }

        return Result(centroids: centroids, assignments: assignments)
    }

    private static func initializeCentroids(
        vectors: [[Float]],
        k: Int,
        metric: Metric,
        seed: UInt64
    ) -> [[Float]] {
        var rng = RandomNumberGenerator64(seed: seed)
        var centroids: [[Float]] = []

        let firstIndex = Int.random(in: 0..<vectors.count, using: &rng)
        centroids.append(vectors[firstIndex])

        for _ in 1..<k {
            var distances = [Float](repeating: 0, count: vectors.count)
            var totalDistance: Float = 0

            for i in 0..<vectors.count {
                var minDistance = Float.greatestFiniteMagnitude
                for centroid in centroids {
                    let distance = SIMDDistance.distance(vectors[i], centroid, metric: metric)
                    if distance < minDistance {
                        minDistance = distance
                    }
                }
                distances[i] = minDistance
                totalDistance += minDistance
            }

            if totalDistance <= 0 {
                let randomIndex = Int.random(in: 0..<vectors.count, using: &rng)
                centroids.append(vectors[randomIndex])
                continue
            }

            var threshold = Float.random(in: 0..<totalDistance, using: &rng)
            var chosenIndex = vectors.count - 1

            for i in 0..<vectors.count {
                threshold -= distances[i]
                if threshold <= 0 {
                    chosenIndex = i
                    break
                }
            }

            centroids.append(vectors[chosenIndex])
        }

        return centroids
    }
}

private struct RandomNumberGenerator64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 1 : seed
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
