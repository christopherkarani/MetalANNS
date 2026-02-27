import Foundation
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "HNSWBuilder")

public enum HNSWBuilder {
    private static let defaultML = 1.4426950408889634

    /// Build HNSW skip layers from a complete base graph.
    public static func buildLayers(
        vectors: any VectorStorage,
        graph: [[(UInt32, Float)]],
        nodeCount: Int,
        metric: Metric,
        config: HNSWConfiguration = .default
    ) throws(ANNSError) -> HNSWLayers {
        guard config.enabled else {
            return HNSWLayers()
        }
        guard nodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot build HNSW layers with empty graph")
        }
        guard nodeCount == graph.count else {
            throw ANNSError.constructionFailed("Graph size does not match node count")
        }
        guard vectors.count >= nodeCount else {
            throw ANNSError.constructionFailed("Vector storage count is less than node count")
        }

        var nodeLevels = Array(repeating: 0, count: nodeCount)
        var maxLayerAssigned = 0

        for nodeID in 0..<nodeCount {
            let level = assignLevel(mL: defaultML, maxLayers: config.maxLayers)
            nodeLevels[nodeID] = level
            maxLayerAssigned = max(maxLayerAssigned, level)
        }

        var entryPoint: UInt32 = 0
        var entryLevel = nodeLevels[0]
        for nodeID in 1..<nodeCount {
            if nodeLevels[nodeID] > entryLevel {
                entryLevel = nodeLevels[nodeID]
                entryPoint = UInt32(nodeID)
            }
        }

        logger.debug("Assigned HNSW levels: maxLayer=\(maxLayerAssigned), entryPoint=\(entryPoint)")

        var skipLayers: [SkipLayer] = []
        if maxLayerAssigned > 0 {
            for layer in 1...maxLayerAssigned {
                let skipLayer = buildSkipLayer(
                    at: layer,
                    nodeLevels: nodeLevels,
                    vectors: vectors,
                    nodeCount: nodeCount,
                    metric: metric,
                    M: config.M
                )
                skipLayers.append(skipLayer)
            }
        }

        return HNSWLayers(
            layers: skipLayers,
            maxLayer: maxLayerAssigned,
            mL: defaultML,
            entryPoint: entryPoint
        )
    }

    /// Assign level via exponential distribution:
    /// floor(-ln(uniform(0, 1)) * mL), capped by maxLayers.
    static func assignLevel(mL: Double, maxLayers: Int) -> Int {
        guard maxLayers > 0 else {
            return 0
        }
        let uniform = Double.random(in: Double.ulpOfOne..<1.0)
        let raw = Int(floor(-log(uniform) * mL))
        return min(max(0, raw), maxLayers)
    }

    private static func buildSkipLayer(
        at layer: Int,
        nodeLevels: [Int],
        vectors: any VectorStorage,
        nodeCount: Int,
        metric: Metric,
        M: Int
    ) -> SkipLayer {
        var nodesAtLayer: [UInt32] = []
        nodesAtLayer.reserveCapacity(nodeCount / 2)
        for nodeID in 0..<nodeCount where nodeLevels[nodeID] >= layer {
            nodesAtLayer.append(UInt32(nodeID))
        }

        guard !nodesAtLayer.isEmpty else {
            return SkipLayer()
        }

        var nodeToLayerIndex: [UInt32: UInt32] = [:]
        nodeToLayerIndex.reserveCapacity(nodesAtLayer.count)
        var layerIndexToNode: [UInt32] = []
        layerIndexToNode.reserveCapacity(nodesAtLayer.count)

        for (index, nodeID) in nodesAtLayer.enumerated() {
            let localIndex = UInt32(index)
            nodeToLayerIndex[nodeID] = localIndex
            layerIndexToNode.append(nodeID)
        }

        var adjacency = Array(repeating: [UInt32](), count: nodesAtLayer.count)

        for (layerIndex, nodeID) in nodesAtLayer.enumerated() {
            let sourceVector = vectors.vector(at: Int(nodeID))
            var candidates: [(UInt32, Float)] = []
            candidates.reserveCapacity(max(0, nodesAtLayer.count - 1))

            for otherID in nodesAtLayer where otherID != nodeID {
                let otherVector = vectors.vector(at: Int(otherID))
                let distance = SIMDDistance.distance(sourceVector, otherVector, metric: metric)
                candidates.append((otherID, distance))
            }

            candidates.sort { $0.1 < $1.1 }
            let neighborLimit = min(max(1, M), candidates.count)
            adjacency[layerIndex] = Array(candidates.prefix(neighborLimit).map(\.0))
        }

        return SkipLayer(
            nodeToLayerIndex: nodeToLayerIndex,
            layerIndexToNode: layerIndexToNode,
            adjacency: adjacency
        )
    }
}
