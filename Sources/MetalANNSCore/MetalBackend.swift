import Metal

public final class MetalBackend: ComputeBackend, @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ANNSError.deviceNotSupported
        }
        guard let queue = device.makeCommandQueue() else {
            throw ANNSError.deviceNotSupported
        }
        self.device = device
        self.commandQueue = queue
    }

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
