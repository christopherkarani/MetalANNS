import Foundation

public enum QuantizedHNSWBuilder {
    /// Build quantized HNSW skip layers from a complete base HNSW structure.
    public static func build(
        from hnsw: HNSWLayers,
        vectors: [[Float]],
        config: QuantizedHNSWConfiguration,
        metric: Metric
    ) throws(ANNSError) -> QuantizedHNSWLayers {
        _ = metric
        guard hnsw.maxLayer > 0 else {
            return QuantizedHNSWLayers(
                quantizedLayers: [],
                maxLayer: 0,
                mL: hnsw.mL,
                entryPoint: hnsw.entryPoint
            )
        }
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot build QuantizedHNSWLayers with empty vectors")
        }

        let dim = vectors[0].count
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }
        for vector in vectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
        guard Int(hnsw.entryPoint) < vectors.count else {
            throw ANNSError.constructionFailed("HNSW entry point out of vector bounds")
        }
        guard hnsw.layers.count == hnsw.maxLayer else {
            throw ANNSError.constructionFailed(
                "HNSW layer count mismatch: layers=\(hnsw.layers.count), maxLayer=\(hnsw.maxLayer)"
            )
        }

        let effectiveSubspaces = largestDivisorOf(dim, atMost: config.pqSubspaces)
        guard effectiveSubspaces > 0 else {
            throw ANNSError.constructionFailed(
                "No valid pqSubspaces <= \(config.pqSubspaces) divides dimension \(dim)"
            )
        }

        var quantizedLayers: [QuantizedSkipLayer] = []
        quantizedLayers.reserveCapacity(hnsw.layers.count)

        for skipLayer in hnsw.layers {
            try validate(skipLayer: skipLayer, vectorCount: vectors.count)
            let nodeCount = skipLayer.layerIndexToNode.count
            if nodeCount >= 256, config.useQuantizedEdges {
                let layerVectors: [[Float]]
                do {
                    layerVectors = try skipLayer.layerIndexToNode.map { nodeID in
                        let index = Int(nodeID)
                        guard index >= 0, index < vectors.count else {
                            throw ANNSError.constructionFailed("Layer node ID out of vector bounds: \(nodeID)")
                        }
                        return vectors[index]
                    }
                } catch let error as ANNSError {
                    throw error
                } catch {
                    throw ANNSError.constructionFailed("Failed collecting layer vectors: \(error.localizedDescription)")
                }

                do {
                    let pq = try ProductQuantizer.train(
                        vectors: layerVectors,
                        numSubspaces: effectiveSubspaces,
                        centroidsPerSubspace: 256,
                        maxIterations: 20
                    )
                    let codes = try layerVectors.map { try pq.encode(vector: $0) }
                    quantizedLayers.append(QuantizedSkipLayer(base: skipLayer, pq: pq, codes: codes))
                } catch {
                    quantizedLayers.append(QuantizedSkipLayer(base: skipLayer, pq: nil, codes: []))
                }
            } else {
                quantizedLayers.append(QuantizedSkipLayer(base: skipLayer, pq: nil, codes: []))
            }
        }

        return QuantizedHNSWLayers(
            quantizedLayers: quantizedLayers,
            maxLayer: hnsw.maxLayer,
            mL: hnsw.mL,
            entryPoint: hnsw.entryPoint
        )
    }

    /// Returns the largest integer <= `maxValue` that divides `n` evenly.
    /// Returns 0 if no such divisor exists > 0.
    static func largestDivisorOf(_ n: Int, atMost maxValue: Int) -> Int {
        guard n > 0, maxValue > 0 else {
            return 0
        }

        let cap = min(maxValue, n)
        for candidate in stride(from: cap, through: 1, by: -1) {
            if n.isMultiple(of: candidate) {
                return candidate
            }
        }
        return 0
    }

    private static func validate(skipLayer: SkipLayer, vectorCount: Int) throws(ANNSError) {
        let nodeCount = skipLayer.layerIndexToNode.count
        guard skipLayer.adjacency.count == nodeCount else {
            throw ANNSError.constructionFailed("Skip-layer adjacency count does not match layer node count")
        }

        for (globalNodeID, layerIndexRaw) in skipLayer.nodeToLayerIndex {
            let layerIndex = Int(layerIndexRaw)
            guard layerIndex >= 0, layerIndex < nodeCount else {
                throw ANNSError.constructionFailed("Skip-layer node mapping index out of bounds")
            }
            guard Int(globalNodeID) < vectorCount else {
                throw ANNSError.constructionFailed("Skip-layer node ID out of vector bounds")
            }
            guard skipLayer.layerIndexToNode[layerIndex] == globalNodeID else {
                throw ANNSError.constructionFailed("Skip-layer node mappings are inconsistent")
            }
        }

        for nodeID in skipLayer.layerIndexToNode where Int(nodeID) >= vectorCount {
            throw ANNSError.constructionFailed("Skip-layer layerIndexToNode contains out-of-bounds node ID")
        }
    }
}
