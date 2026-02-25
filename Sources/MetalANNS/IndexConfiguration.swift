import MetalANNSCore

public struct IndexConfiguration: Sendable {
    public var degree: Int
    public var metric: Metric
    public var efConstruction: Int
    public var efSearch: Int
    public var maxIterations: Int
    public var useFloat16: Bool
    public var convergenceThreshold: Float

    public static let `default` = IndexConfiguration(
        degree: 32,
        metric: .cosine,
        efConstruction: 100,
        efSearch: 64,
        maxIterations: 20,
        useFloat16: false,
        convergenceThreshold: 0.001
    )

    public init(
        degree: Int = 32,
        metric: Metric = .cosine,
        efConstruction: Int = 100,
        efSearch: Int = 64,
        maxIterations: Int = 20,
        useFloat16: Bool = false,
        convergenceThreshold: Float = 0.001
    ) {
        self.degree = degree
        self.metric = metric
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.maxIterations = maxIterations
        self.useFloat16 = useFloat16
        self.convergenceThreshold = convergenceThreshold
    }
}
