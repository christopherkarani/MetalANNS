import Metal

public final class MetalBackend: ComputeBackend, @unchecked Sendable {
    private let context: MetalContext

    public init(context: MetalContext? = nil) throws {
        if let context {
            self.context = context
        } else {
            self.context = try MetalContext()
        }
    }

    public func computeDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric
    ) async throws -> [Float] {
        guard query.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: query.count)
        }
        let expectedCount = vectorCount * dim
        guard vectors.count == expectedCount else {
            throw ANNSError.dimensionMismatch(expected: expectedCount, got: vectors.count)
        }

        if vectorCount == 0 {
            return []
        }

        let functionName: String
        switch metric {
        case .cosine:
            functionName = "cosine_distance"
        case .l2:
            functionName = "l2_distance"
        case .innerProduct:
            functionName = "inner_product_distance"
        case .hamming:
            throw ANNSError.searchFailed("Metric .hamming is not supported by MetalBackend")
        }

        let pipeline = try await context.pipelineCache.pipeline(for: functionName)

        let scalarSize = MemoryLayout<Float>.stride
        let queryLength = query.count * scalarSize
        let corpusLength = expectedCount * scalarSize
        let outputLength = vectorCount * scalarSize

        guard
            let queryBuffer = context.device.makeBuffer(
                bytes: query,
                length: queryLength,
                options: .storageModeShared
            ),
            let vectorsBase = vectors.baseAddress,
            let corpusBuffer = context.device.makeBuffer(
                bytes: vectorsBase,
                length: corpusLength,
                options: .storageModeShared
            ),
            let outputBuffer = context.device.makeBuffer(
                length: outputLength,
                options: .storageModeShared
            )
        else {
            throw ANNSError.constructionFailed("Failed to allocate Metal buffers")
        }

        var dimU32 = UInt32(dim)
        var nU32 = UInt32(vectorCount)

        try await context.executeOnPool { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(corpusBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
            encoder.setBytes(&dimU32, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&nU32, length: MemoryLayout<UInt32>.stride, index: 4)

            let threadsPerGrid = MTLSize(width: vectorCount, height: 1, depth: 1)
            let threadgroupWidth = max(1, min(vectorCount, pipeline.maxTotalThreadsPerThreadgroup))
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: vectorCount)
        return Array(UnsafeBufferPointer(start: outputPointer, count: vectorCount))
    }
}
