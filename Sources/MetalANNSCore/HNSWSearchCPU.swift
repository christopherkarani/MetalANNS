import Foundation

public enum HNSWSearchCPU {
    /// Search using HNSW layer hierarchy and beam search at layer 0.
    public static func search(
        query: [Float],
        vectors: [[Float]],
        hnsw: HNSWLayers,
        baseGraph: [[(UInt32, Float)]],
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws(ANNSError) -> [SearchResult] {
        guard k > 0 else {
            return []
        }
        guard !vectors.isEmpty else {
            throw ANNSError.indexEmpty
        }
        guard vectors.count == baseGraph.count else {
            throw ANNSError.searchFailed("Graph size does not match vector count")
        }
        let dim = vectors[0].count
        guard query.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: query.count)
        }
        for vector in vectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }

        var currentEntry = Int(hnsw.entryPoint)
        if hnsw.maxLayer > 0 {
            for layer in stride(from: hnsw.maxLayer, through: 1, by: -1) {
                currentEntry = Int(
                    try greedySearchLayer(
                        query: query,
                        vectors: vectors,
                        hnsw: hnsw,
                        layer: layer,
                        entryPoint: currentEntry,
                        metric: metric
                    )
                )
            }
        }

        do {
            return try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: baseGraph,
                entryPoint: currentEntry,
                k: k,
                ef: ef,
                metric: metric
            )
        } catch let error as ANNSError {
            throw error
        } catch {
            throw ANNSError.searchFailed("HNSW layer-0 beam search failed: \(error.localizedDescription)")
        }
    }

    static func greedySearchLayer(
        query: [Float],
        vectors: [[Float]],
        hnsw: HNSWLayers,
        layer: Int,
        entryPoint: Int,
        metric: Metric
    ) throws(ANNSError) -> UInt32 {
        guard layer > 0, layer <= hnsw.maxLayer else {
            throw ANNSError.searchFailed("Invalid layer for greedy search")
        }
        guard entryPoint >= 0, entryPoint < vectors.count else {
            throw ANNSError.searchFailed("Entry point out of bounds")
        }

        var current = UInt32(entryPoint)
        var currentDistance = SIMDDistance.distance(query, vectors[entryPoint], metric: metric)
        var improved = true
        var iterations = 0
        let maxIterations = 128

        while improved && iterations < maxIterations {
            improved = false
            iterations += 1

            guard let neighbors = hnsw.neighbors(of: current, at: layer) else {
                break
            }

            var bestNode = current
            var bestDistance = currentDistance

            for neighborID in neighbors {
                if neighborID == UInt32.max || Int(neighborID) >= vectors.count {
                    continue
                }
                let neighborDistance = SIMDDistance.distance(
                    query,
                    vectors[Int(neighborID)],
                    metric: metric
                )
                if neighborDistance < bestDistance {
                    bestNode = neighborID
                    bestDistance = neighborDistance
                }
            }

            if bestNode != current {
                current = bestNode
                currentDistance = bestDistance
                improved = true
            }
        }

        return current
    }
}
