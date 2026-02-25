import Foundation
import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "MetalDevice")

public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    public let pipelineCache: PipelineCache

    public init() throws(ANNSError) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ANNSError.deviceNotSupported
        }
        guard let queue = device.makeCommandQueue() else {
            throw ANNSError.deviceNotSupported
        }

        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            throw ANNSError.constructionFailed("Failed to load Metal shader library: \(error)")
        }

        self.device = device
        self.commandQueue = queue
        self.library = library
        self.pipelineCache = PipelineCache(device: device, library: library)

        logger.debug("MetalContext initialized: \(device.name)")
    }

    public func execute(_ encode: (MTLCommandBuffer) throws(ANNSError) -> Void) async throws(ANNSError) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw ANNSError.constructionFailed("Failed to create command buffer")
        }
        try encode(commandBuffer)
        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw ANNSError.constructionFailed("Command buffer failed: \(error)")
        }
    }
}
