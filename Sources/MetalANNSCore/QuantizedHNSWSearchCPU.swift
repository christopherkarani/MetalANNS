import Foundation

public enum QuantizedHNSWSearchCPU {
    /// Search using quantized HNSW skip layers with ADC in greedy phase.
    /// Layer 0 uses full-precision BeamSearchCPU.
    public static func search(
        query: [Float],
        vectors: [[Float]],
        hnsw: QuantizedHNSWLayers,
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
        guard query.count == vectors[0].count else {
            throw ANNSError.dimensionMismatch(expected: vectors[0].count, got: query.count)
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
            throw ANNSError.searchFailed("Quantized HNSW layer-0 beam search failed: \(error)")
        }
    }

    /// Greedy descent on a single skip layer using ADC when PQ is available.
    static func greedySearchLayer(
        query: [Float],
        vectors: [[Float]],
        hnsw: QuantizedHNSWLayers,
        layer: Int,
        entryPoint: Int,
        metric: Metric
    ) throws(ANNSError) -> UInt32 {
        guard layer > 0, layer <= hnsw.maxLayer else {
            throw ANNSError.searchFailed("Invalid layer for quantized greedy search")
        }
        guard entryPoint >= 0, entryPoint < vectors.count else {
            throw ANNSError.searchFailed("Entry point out of bounds")
        }
        guard let qLayer = hnsw.quantizedLayer(at: layer) else {
            throw ANNSError.searchFailed("No quantized layer at \(layer)")
        }

        let skipLayer = qLayer.base
        let pq = qLayer.pq
        let adcTable = pq?.distanceTable(query: query, metric: metric)

        var current = UInt32(entryPoint)
        var currentDistance: Float

        if let pq, let table = adcTable,
           let layerIdx = skipLayer.nodeToLayerIndex[current],
           Int(layerIdx) < qLayer.codes.count {
            currentDistance = approximateWithTable(
                table,
                codes: qLayer.codes[Int(layerIdx)],
                numSubspaces: pq.numSubspaces
            )
        } else {
            currentDistance = SIMDDistance.distance(query, vectors[entryPoint], metric: metric)
        }

        var improved = true
        var iterations = 0
        let maxIterations = 128

        while improved && iterations < maxIterations {
            improved = false
            iterations += 1

            guard let layerIdxRaw = skipLayer.nodeToLayerIndex[current] else {
                break
            }
            let neighbors = skipLayer.adjacency[Int(layerIdxRaw)]

            for neighborID in neighbors {
                if neighborID == UInt32.max || Int(neighborID) >= vectors.count {
                    continue
                }

                let neighborDistance: Float
                if let pq, let table = adcTable,
                   let neighborLayerIndex = skipLayer.nodeToLayerIndex[neighborID],
                   Int(neighborLayerIndex) < qLayer.codes.count {
                    neighborDistance = approximateWithTable(
                        table,
                        codes: qLayer.codes[Int(neighborLayerIndex)],
                        numSubspaces: pq.numSubspaces
                    )
                } else {
                    neighborDistance = SIMDDistance.distance(query, vectors[Int(neighborID)], metric: metric)
                }

                if neighborDistance < currentDistance {
                    current = neighborID
                    currentDistance = neighborDistance
                    improved = true
                }
            }
        }

        return current
    }

    /// Inline ADC lookup to reuse a precomputed table per layer.
    @inline(__always)
    private static func approximateWithTable(
        _ table: [[Float]],
        codes: [UInt8],
        numSubspaces: Int
    ) -> Float {
        guard table.count >= numSubspaces, codes.count >= numSubspaces else {
            return Float.greatestFiniteMagnitude
        }

        var distance: Float = 0
        for subspace in 0..<numSubspaces {
            let code = Int(codes[subspace])
            if code >= table[subspace].count {
                return Float.greatestFiniteMagnitude
            }
            distance += table[subspace][code]
        }
        return distance
    }
}
