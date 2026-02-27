import MetalANNSCore

public struct IndexConfiguration: Sendable, Codable {
    public var degree: Int
    public var metric: Metric
    public var efConstruction: Int
    public var efSearch: Int
    public var maxIterations: Int
    public var useFloat16: Bool
    public var convergenceThreshold: Float
    public var hnswConfiguration: HNSWConfiguration
    public var repairConfiguration: RepairConfiguration
    public var quantizedHNSWConfiguration: QuantizedHNSWConfiguration

    public static let `default` = IndexConfiguration(
        degree: 32,
        metric: .cosine,
        efConstruction: 100,
        efSearch: 64,
        maxIterations: 20,
        useFloat16: false,
        convergenceThreshold: 0.001,
        hnswConfiguration: .default,
        repairConfiguration: .default,
        quantizedHNSWConfiguration: .default
    )

    public init(
        degree: Int = 32,
        metric: Metric = .cosine,
        efConstruction: Int = 100,
        efSearch: Int = 64,
        maxIterations: Int = 20,
        useFloat16: Bool = false,
        convergenceThreshold: Float = 0.001,
        hnswConfiguration: HNSWConfiguration = .default,
        repairConfiguration: RepairConfiguration = .default,
        quantizedHNSWConfiguration: QuantizedHNSWConfiguration = .default
    ) {
        self.degree = degree
        self.metric = metric
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.maxIterations = maxIterations
        self.useFloat16 = useFloat16
        self.convergenceThreshold = convergenceThreshold
        self.hnswConfiguration = hnswConfiguration
        self.repairConfiguration = repairConfiguration
        self.quantizedHNSWConfiguration = quantizedHNSWConfiguration
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        degree = try container.decode(Int.self, forKey: .degree)
        metric = try container.decode(Metric.self, forKey: .metric)
        efConstruction = try container.decode(Int.self, forKey: .efConstruction)
        efSearch = try container.decode(Int.self, forKey: .efSearch)
        maxIterations = try container.decode(Int.self, forKey: .maxIterations)
        useFloat16 = try container.decode(Bool.self, forKey: .useFloat16)
        convergenceThreshold = try container.decode(Float.self, forKey: .convergenceThreshold)
        hnswConfiguration = try container.decodeIfPresent(
            HNSWConfiguration.self,
            forKey: .hnswConfiguration
        ) ?? .default
        repairConfiguration = try container.decodeIfPresent(
            RepairConfiguration.self,
            forKey: .repairConfiguration
        ) ?? .default
        quantizedHNSWConfiguration = try container.decodeIfPresent(
            QuantizedHNSWConfiguration.self,
            forKey: .quantizedHNSWConfiguration
        ) ?? .default
    }
}
