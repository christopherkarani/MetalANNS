import Foundation

/// Controls whether and how HNSW skip-layer greedy search uses ADC instead of
/// exact distances.
public struct QuantizedHNSWConfiguration: Sendable, Codable, Equatable {
    /// Base HNSW settings (layer count, M). Forwarded to HNSWBuilder.
    public var base: HNSWConfiguration

    /// When true, skip layers use ADC instead of exact distance.
    public var useQuantizedEdges: Bool

    /// Number of PQ subspaces for skip-layer codebooks.
    public var pqSubspaces: Int

    public static let `default` = QuantizedHNSWConfiguration(
        base: .default,
        useQuantizedEdges: true,
        pqSubspaces: 4
    )

    public init(
        base: HNSWConfiguration = .default,
        useQuantizedEdges: Bool = true,
        pqSubspaces: Int = 4
    ) {
        self.base = base
        self.useQuantizedEdges = useQuantizedEdges
        self.pqSubspaces = max(1, pqSubspaces)
    }

    private enum CodingKeys: String, CodingKey {
        case base
        case useQuantizedEdges
        case pqSubspaces
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        base = try container.decode(HNSWConfiguration.self, forKey: .base)
        useQuantizedEdges = try container.decode(Bool.self, forKey: .useQuantizedEdges)
        let decodedSubspaces = try container.decode(Int.self, forKey: .pqSubspaces)
        pqSubspaces = max(1, decodedSubspaces)
    }

    public static func == (lhs: QuantizedHNSWConfiguration, rhs: QuantizedHNSWConfiguration) -> Bool {
        lhs.base.enabled == rhs.base.enabled &&
            lhs.base.M == rhs.base.M &&
            lhs.base.maxLayers == rhs.base.maxLayers &&
            lhs.useQuantizedEdges == rhs.useQuantizedEdges &&
            lhs.pqSubspaces == rhs.pqSubspaces
    }
}
