import Accelerate

public struct AccelerateBackend: ComputeBackend, Sendable {
    public init() {}

    public func computeDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric
    ) async throws -> [Float] {
        fatalError("Not yet implemented")
    }
}
