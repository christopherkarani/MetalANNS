import Foundation

/// Skip layer in the HNSW hierarchy.
public struct SkipLayer: Sendable, Codable {
    /// Maps graph node ID -> layer-local node index.
    public var nodeToLayerIndex: [UInt32: UInt32]

    /// Maps layer-local index -> graph node ID.
    public var layerIndexToNode: [UInt32]

    /// Layer adjacency lists indexed by layer-local index.
    public var adjacency: [[UInt32]]

    public init(
        nodeToLayerIndex: [UInt32: UInt32] = [:],
        layerIndexToNode: [UInt32] = [],
        adjacency: [[UInt32]] = []
    ) {
        self.nodeToLayerIndex = nodeToLayerIndex
        self.layerIndexToNode = layerIndexToNode
        self.adjacency = adjacency
    }
}

/// Complete HNSW skip-layer structure. Layer 0 uses the base graph.
public struct HNSWLayers: Sendable {
    /// Skip layers where `layers[0]` corresponds to layer 1.
    public let layers: [SkipLayer]

    /// Maximum assigned layer in the hierarchy.
    public let maxLayer: Int

    /// Level multiplier (1 / ln(2)).
    public let mL: Double

    /// Entry point in the highest populated layer.
    public let entryPoint: UInt32

    public init(
        layers: [SkipLayer] = [],
        maxLayer: Int = 0,
        mL: Double = 1.4426950408889634,
        entryPoint: UInt32 = 0
    ) {
        self.layers = layers
        self.maxLayer = maxLayer
        self.mL = mL
        self.entryPoint = entryPoint
    }

    /// Returns neighbor node IDs for `nodeID` at a skip layer.
    public func neighbors(of nodeID: UInt32, at layer: Int) -> [UInt32]? {
        guard layer > 0, layer <= maxLayer else {
            return nil
        }
        let skipLayer = layers[layer - 1]
        guard let layerIndex = skipLayer.nodeToLayerIndex[nodeID] else {
            return nil
        }
        return skipLayer.adjacency[Int(layerIndex)]
    }
}
