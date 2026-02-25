import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "NNDescentGPU")

public enum NNDescentGPU {
    public static func randomInit(
        context: MetalContext,
        graph: GraphBuffer,
        nodeCount: Int,
        seed: UInt32 = 42
    ) async throws {
        guard nodeCount > 0, nodeCount <= graph.capacity else {
            throw ANNSError.constructionFailed("nodeCount out of bounds for graph capacity")
        }

        let pipeline = try await context.pipelineCache.pipeline(for: "random_init")

        var n = UInt32(nodeCount)
        var degree = UInt32(graph.degree)
        var seedValue = seed

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 0)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBytes(&degree, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&seedValue, length: MemoryLayout<UInt32>.stride, index: 3)

            let threadsPerGrid = MTLSize(width: nodeCount, height: 1, depth: 1)
            let threadgroupWidth = max(1, min(nodeCount, pipeline.maxTotalThreadsPerThreadgroup))
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        logger.debug("random_init complete for \(nodeCount) nodes")
    }

    public static func computeInitialDistances(
        context: MetalContext,
        vectors: VectorBuffer,
        graph: GraphBuffer,
        nodeCount: Int,
        metric: Metric
    ) async throws {
        guard nodeCount > 0, nodeCount <= graph.capacity, nodeCount <= vectors.capacity else {
            throw ANNSError.constructionFailed("nodeCount out of bounds for graph/vector capacity")
        }

        let pipeline = try await context.pipelineCache.pipeline(for: "compute_initial_distances")

        var n = UInt32(nodeCount)
        var degree = UInt32(graph.degree)
        var dim = UInt32(vectors.dim)
        var metricType: UInt32 = switch metric {
        case .cosine: 0
        case .l2: 1
        case .innerProduct: 2
        }

        let totalThreads = nodeCount * graph.degree

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(vectors.buffer, offset: 0, index: 0)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 1)
            encoder.setBuffer(graph.distanceBuffer, offset: 0, index: 2)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&degree, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&metricType, length: MemoryLayout<UInt32>.stride, index: 6)

            let threadsPerGrid = MTLSize(width: totalThreads, height: 1, depth: 1)
            let threadgroupWidth = max(1, min(totalThreads, pipeline.maxTotalThreadsPerThreadgroup))
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
    }
}
