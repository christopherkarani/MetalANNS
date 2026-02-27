import Foundation

/// A skip layer where each node's vector is replaced by a PQ code for fast ADC lookup.
/// When `pq` is nil, exact distance is used as fallback.
public struct QuantizedSkipLayer: Sendable, Codable {
    /// The base skip layer with unchanged adjacency and node mappings.
    public var base: SkipLayer

    /// PQ codebook trained on this layer's node vectors.
    /// Nil when the layer has too few nodes for training.
    public var pq: ProductQuantizer?

    /// PQ codes indexed by layer-local index.
    public var codes: [[UInt8]]

    public init(base: SkipLayer, pq: ProductQuantizer?, codes: [[UInt8]]) {
        self.base = base
        self.pq = pq
        self.codes = codes
    }
}

/// Complete quantized HNSW skip-layer structure. Layer 0 still uses full-precision beam search.
public final class QuantizedHNSWLayers: Sendable, Codable {
    /// Quantized skip layers where `quantizedLayers[0]` corresponds to layer 1.
    public let quantizedLayers: [QuantizedSkipLayer]

    /// Maximum assigned layer in the hierarchy.
    public let maxLayer: Int

    /// Level multiplier (1 / ln(2)).
    public let mL: Double

    /// Entry point node ID in the highest populated layer.
    public let entryPoint: UInt32

    public init(
        quantizedLayers: [QuantizedSkipLayer] = [],
        maxLayer: Int = 0,
        mL: Double = 1.4426950408889634,
        entryPoint: UInt32 = 0
    ) {
        self.quantizedLayers = quantizedLayers
        self.maxLayer = maxLayer
        self.mL = mL
        self.entryPoint = entryPoint
    }

    /// Returns the quantized skip layer for `layer` (1-indexed, matching HNSWLayers).
    public func quantizedLayer(at layer: Int) -> QuantizedSkipLayer? {
        guard layer > 0, layer <= maxLayer else {
            return nil
        }
        return quantizedLayers[layer - 1]
    }
}
