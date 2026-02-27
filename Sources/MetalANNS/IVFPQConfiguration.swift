import MetalANNSCore

public struct IVFPQConfiguration: Sendable, Codable {
    public var numSubspaces: Int
    public var numCentroids: Int
    public var numCoarseCentroids: Int
    public var nprobe: Int
    public var metric: Metric
    public var trainingIterations: Int

    public init(
        numSubspaces: Int = 8,
        numCentroids: Int = 256,
        numCoarseCentroids: Int = 256,
        nprobe: Int = 8,
        metric: Metric = .l2,
        trainingIterations: Int = 20
    ) {
        self.numSubspaces = max(1, numSubspaces)
        self.numCentroids = numCentroids
        self.numCoarseCentroids = max(1, numCoarseCentroids)
        self.nprobe = max(1, nprobe)
        self.metric = metric
        self.trainingIterations = max(1, trainingIterations)
    }
}
