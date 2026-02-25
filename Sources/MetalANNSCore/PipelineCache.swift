import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "PipelineCache")

public actor PipelineCache {
    private let device: MTLDevice
    private let library: MTLLibrary
    private var cache: [String: MTLComputePipelineState] = [:]

    public init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        self.library = library
    }

    public func pipeline(for functionName: String) throws(ANNSError) -> MTLComputePipelineState {
        if let cached = cache[functionName] {
            return cached
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw ANNSError.constructionFailed("Metal function '\(functionName)' not found")
        }

        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw ANNSError.constructionFailed("Failed to compile Metal function '\(functionName)': \(error)")
        }
        cache[functionName] = pipeline
        logger.debug("Compiled pipeline: \(functionName)")
        return pipeline
    }
}
