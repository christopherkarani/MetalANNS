import Foundation
import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "MetalDevice")

public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let queuePool: CommandQueuePool
    public let library: MTLLibrary
    public let pipelineCache: PipelineCache
    public let searchBufferPool: SearchBufferPool

    public init() throws {
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
        self.queuePool = try CommandQueuePool(
            device: device,
            count: max(1, min(ProcessInfo.processInfo.activeProcessorCount, 16))
        )
        self.library = library
        self.pipelineCache = PipelineCache(device: device, library: library)
        self.searchBufferPool = SearchBufferPool(device: device)

        logger.debug("MetalContext initialized: \(device.name)")
    }

    public func execute(_ encode: (MTLCommandBuffer) throws -> Void) async throws {
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

    /// Like execute(), but picks a queue from the pool for pipelining concurrent calls.
    public func executeOnPool(_ encode: (MTLCommandBuffer) throws -> Void) async throws {
        let queue = await queuePool.next()
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ANNSError.constructionFailed("Failed to create command buffer from pool queue")
        }
        try encode(commandBuffer)
        commandBuffer.commit()
        await commandBuffer.completed()

        if let error = commandBuffer.error {
            throw ANNSError.constructionFailed("Command buffer failed: \(error.localizedDescription)")
        }
    }
}
