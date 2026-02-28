import Foundation

public enum HNSWInserter {
    /// Insert a single node into existing HNSW skip layers (layers 1+).
    /// Layer 0 / base graph is handled separately by IncrementalBuilder.
    public static func insert(
        vector: [Float],
        nodeID: UInt32,
        into layers: inout HNSWLayers,
        vectorStorage: any VectorStorage,
        config: HNSWConfiguration,
        metric: Metric
    ) throws(ANNSError) {
        guard vector.count == vectorStorage.dim else {
            throw ANNSError.dimensionMismatch(expected: vectorStorage.dim, got: vector.count)
        }
        guard Int(nodeID) < vectorStorage.count else {
            throw ANNSError.constructionFailed("Node ID out of bounds")
        }
        guard Int(nodeID) == vectorStorage.count - 1 else {
            throw ANNSError.constructionFailed("HNSW incremental insertion requires append-only nodeID")
        }

        if layers.maxLayer > 0 {
            for layer in 1...layers.maxLayer {
                if layers.layers[layer - 1].nodeToLayerIndex[nodeID] != nil {
                    throw ANNSError.constructionFailed("Node already present in HNSW skip layers")
                }
            }
        }

        let nodeLevel = HNSWBuilder.assignLevel(mL: layers.mL, maxLayers: config.maxLayers)
        if nodeLevel == 0 {
            return
        }

        var currentNodeID = layers.entryPoint
        var currentDist = SIMDDistance.distance(
            vector,
            vectorStorage.vector(at: Int(currentNodeID)),
            metric: metric
        )

        if layers.maxLayer >= nodeLevel + 1 {
            for layer in stride(from: layers.maxLayer, through: nodeLevel + 1, by: -1) {
                guard !adjacencyOf(currentNodeID, at: layer, in: layers).isEmpty else {
                    continue
                }

                var improved = true
                while improved {
                    improved = false
                    let neighbors = adjacencyOf(currentNodeID, at: layer, in: layers)
                    for neighborID in neighbors {
                        guard Int(neighborID) < vectorStorage.count else {
                            continue
                        }
                        let distance = SIMDDistance.distance(
                            vector,
                            vectorStorage.vector(at: Int(neighborID)),
                            metric: metric
                        )
                        if distance < currentDist {
                            currentDist = distance
                            currentNodeID = neighborID
                            improved = true
                        }
                    }
                }
            }
        }

        if nodeLevel > layers.maxLayer {
            let previousMaxLayer = layers.maxLayer
            while layers.layers.count < nodeLevel {
                layers.layers.append(SkipLayer())
            }
            if previousMaxLayer + 1 <= nodeLevel {
                for layer in previousMaxLayer + 1 ... nodeLevel {
                    var skipLayer = layers.layers[layer - 1]
                    skipLayer.nodeToLayerIndex[nodeID] = 0
                    skipLayer.layerIndexToNode = [nodeID]
                    skipLayer.adjacency = [[]]
                    layers.layers[layer - 1] = skipLayer
                }
                layers.maxLayer = nodeLevel
                layers.entryPoint = nodeID
            }
        }

        for layer in stride(from: nodeLevel, through: 1, by: -1) {
            var skipLayer = layers.layers[layer - 1]

            var candidates: [(nodeID: UInt32, dist: Float)] = []
            candidates.reserveCapacity(skipLayer.layerIndexToNode.count)

            for existingID in skipLayer.layerIndexToNode where existingID != nodeID {
                let existingIndex = Int(existingID)
                guard existingIndex < vectorStorage.count else {
                    continue
                }
                let distance = SIMDDistance.distance(
                    vector,
                    vectorStorage.vector(at: existingIndex),
                    metric: metric
                )
                candidates.append((nodeID: existingID, dist: distance))
            }

            candidates.sort { $0.dist < $1.dist }
            let limit = min(config.M, candidates.count)
            let neighbors = candidates.prefix(limit).map(\.nodeID)

            if let existingIndex = skipLayer.nodeToLayerIndex[nodeID] {
                let layerIndex = Int(existingIndex)
                guard layerIndex < skipLayer.adjacency.count else {
                    throw ANNSError.constructionFailed("Invalid HNSW skip-layer state")
                }
                skipLayer.adjacency[layerIndex] = neighbors
            } else {
                let newLayerIndex = UInt32(skipLayer.layerIndexToNode.count)
                skipLayer.nodeToLayerIndex[nodeID] = newLayerIndex
                skipLayer.layerIndexToNode.append(nodeID)
                skipLayer.adjacency.append(neighbors)
            }

            for neighborID in neighbors {
                guard let nbrLayerIndex = skipLayer.nodeToLayerIndex[neighborID] else {
                    continue
                }

                let nbrLayerIdx = Int(nbrLayerIndex)
                guard nbrLayerIdx < skipLayer.adjacency.count else {
                    continue
                }

                var nbrAdj = skipLayer.adjacency[nbrLayerIdx]
                nbrAdj.append(nodeID)
                if nbrAdj.count > config.M {
                    let nbrVec = vectorStorage.vector(at: Int(neighborID))
                    nbrAdj = nbrAdj
                        .map { candidateID in
                            let distance = SIMDDistance.distance(
                                nbrVec,
                                vectorStorage.vector(at: Int(candidateID)),
                                metric: metric
                            )
                            return (candidateID, distance)
                        }
                        .sorted { (lhs: (UInt32, Float), rhs: (UInt32, Float)) in
                            lhs.1 < rhs.1
                        }
                        .prefix(config.M)
                        .map { (candidateID: UInt32, _) in
                            candidateID
                        }
                }
                skipLayer.adjacency[nbrLayerIdx] = nbrAdj
            }

            layers.layers[layer - 1] = skipLayer
        }
    }

    private static func adjacencyOf(_ nodeID: UInt32, at layer: Int, in layers: HNSWLayers) -> [UInt32] {
        guard layer > 0, layer <= layers.maxLayer else {
            return []
        }
        guard let layerIndex = layers.layers[layer - 1].nodeToLayerIndex[nodeID] else {
            return []
        }
        let localIndex = Int(layerIndex)
        guard localIndex >= 0, localIndex < layers.layers[layer - 1].adjacency.count else {
            return []
        }
        return layers.layers[layer - 1].adjacency[localIndex]
    }
}
