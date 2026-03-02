import Foundation
import Metal

public enum FullGPUSearch {
    private static let maxEF = 256

    public static func search(
        context: MetalContext,
        query: [Float],
        vectors: any VectorStorage,
        graph: GraphBuffer,
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> [SearchResult] {
        let nodeCount = graph.nodeCount > 0
            ? min(graph.nodeCount, vectors.count)
            : min(graph.capacity, vectors.count)

        guard nodeCount > 0 else {
            throw ANNSError.indexEmpty
        }
        guard k > 0 else {
            return []
        }
        guard ef >= k else {
            throw ANNSError.searchFailed("ef must be greater than or equal to k")
        }
        guard k <= maxEF else {
            throw ANNSError.searchFailed("k exceeds FullGPUSearch maximum (\(maxEF)); use CPU/hybrid search")
        }
        guard ef <= maxEF else {
            throw ANNSError.searchFailed("ef exceeds FullGPUSearch maximum (\(maxEF)); use CPU/hybrid search")
        }
        guard query.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
        }
        guard entryPoint >= 0, entryPoint < nodeCount else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }

        let kLimit = min(k, nodeCount)
        let efLimit = min(max(ef, kLimit), nodeCount)

        let kernelName = vectors.isFloat16 ? "beam_search_f16" : "beam_search"
        let pipeline = try await context.pipelineCache.pipeline(for: kernelName)

        let floatSize = MemoryLayout<Float>.stride
        let uintSize = MemoryLayout<UInt32>.stride
        let visitedLength = max(nodeCount * uintSize, uintSize)

        guard
            let queryBuffer = context.device.makeBuffer(
                bytes: query,
                length: query.count * floatSize,
                options: .storageModeShared
            ),
            let outputDistanceBuffer = context.device.makeBuffer(
                length: max(kLimit * floatSize, floatSize),
                options: .storageModeShared
            ),
            let outputIDBuffer = context.device.makeBuffer(
                length: max(kLimit * uintSize, uintSize),
                options: .storageModeShared
            ),
            let visitedGenerationBuffer = context.device.makeBuffer(
                length: visitedLength,
                options: .storageModeShared
            )
        else {
            throw ANNSError.searchFailed("Failed to allocate full GPU search buffers")
        }
        visitedGenerationBuffer.contents().bindMemory(to: UInt32.self, capacity: max(nodeCount, 1)).initialize(
            repeating: 0,
            count: max(nodeCount, 1)
        )

        var nodeCountValue = UInt32(nodeCount)
        var degreeValue = UInt32(graph.degree)
        var dimValue = UInt32(vectors.dim)
        var kValue = UInt32(kLimit)
        var efValue = UInt32(efLimit)
        var entryPointValue = UInt32(entryPoint)
        var visitGenerationValue: UInt32 = 1
        guard metric != .hamming else {
            throw ANNSError.searchFailed("FullGPUSearch does not support metric .hamming")
        }
        var metricType: UInt32 = switch metric {
        case .cosine:
            0
        case .l2:
            1
        case .innerProduct:
            2
        case .hamming:
            0
        }

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.searchFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(vectors.buffer, offset: 0, index: 0)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 1)
            encoder.setBuffer(queryBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputDistanceBuffer, offset: 0, index: 3)
            encoder.setBuffer(outputIDBuffer, offset: 0, index: 4)
            encoder.setBytes(&nodeCountValue, length: uintSize, index: 5)
            encoder.setBytes(&degreeValue, length: uintSize, index: 6)
            encoder.setBytes(&dimValue, length: uintSize, index: 7)
            encoder.setBytes(&kValue, length: uintSize, index: 8)
            encoder.setBytes(&efValue, length: uintSize, index: 9)
            encoder.setBytes(&entryPointValue, length: uintSize, index: 10)
            encoder.setBytes(&metricType, length: uintSize, index: 11)
            encoder.setBuffer(visitedGenerationBuffer, offset: 0, index: 12)
            encoder.setBytes(&visitGenerationValue, length: uintSize, index: 13)

            let threadWidth = max(1, min(pipeline.maxTotalThreadsPerThreadgroup, max(32, min(efLimit, 128))))
            let threadsPerThreadgroup = MTLSize(width: threadWidth, height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        let outputIDPointer = outputIDBuffer.contents().bindMemory(to: UInt32.self, capacity: kLimit)
        let outputDistancePointer = outputDistanceBuffer.contents().bindMemory(to: Float.self, capacity: kLimit)

        var results: [SearchResult] = []
        results.reserveCapacity(kLimit)
        for index in 0..<kLimit {
            let nodeID = outputIDPointer[index]
            if nodeID == UInt32.max {
                continue
            }
            results.append(
                SearchResult(id: "", score: outputDistancePointer[index], internalID: nodeID)
            )
        }
        return results
    }
}
