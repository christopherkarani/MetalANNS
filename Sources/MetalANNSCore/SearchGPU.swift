import Foundation
import Metal

public enum SearchGPU {
    private struct Candidate {
        let nodeID: UInt32
        let distance: Float
    }

    public static func search(
        context: MetalContext,
        query: [Float],
        vectors: VectorBuffer,
        graph: GraphBuffer,
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> [SearchResult] {
        let nodeCount = graph.nodeCount > 0 ? min(graph.nodeCount, vectors.count) : min(graph.capacity, vectors.count)

        guard nodeCount > 0 else {
            throw ANNSError.indexEmpty
        }
        guard k > 0 else {
            return []
        }
        guard ef >= k else {
            throw ANNSError.searchFailed("ef must be greater than or equal to k")
        }
        guard query.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
        }
        guard entryPoint >= 0, entryPoint < nodeCount else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }

        let efLimit = min(ef, nodeCount)
        let entryID = UInt32(entryPoint)
        let entryDistance = distance(query: query, vector: vectors.vector(at: entryPoint), metric: metric)

        var visited: Set<UInt32> = [entryID]
        var candidates: [Candidate] = [Candidate(nodeID: entryID, distance: entryDistance)]
        var results: [Candidate] = [Candidate(nodeID: entryID, distance: entryDistance)]

        while !candidates.isEmpty {
            let current = candidates.removeFirst()
            if results.count >= efLimit, let worst = results.last, current.distance > worst.distance {
                break
            }

            let rawNeighbors = graph.neighborIDs(of: Int(current.nodeID))
            var neighborIDs: [UInt32] = []
            neighborIDs.reserveCapacity(rawNeighbors.count)

            for neighborID in rawNeighbors {
                let neighborIndex = Int(neighborID)
                if neighborID == UInt32.max || neighborIndex < 0 || neighborIndex >= nodeCount {
                    continue
                }
                if visited.contains(neighborID) {
                    continue
                }
                visited.insert(neighborID)
                neighborIDs.append(neighborID)
            }

            if neighborIDs.isEmpty {
                continue
            }

            let neighborDistances = try await computeDistancesOnGPU(
                context: context,
                query: query,
                vectors: vectors,
                neighborIDs: neighborIDs,
                metric: metric
            )

            for (offset, distanceValue) in neighborDistances.enumerated() {
                if results.count < efLimit || distanceValue < results[results.count - 1].distance {
                    let candidate = Candidate(nodeID: neighborIDs[offset], distance: distanceValue)
                    insertSorted(candidate, into: &candidates)
                    insertSorted(candidate, into: &results)
                    if results.count > efLimit {
                        results.removeLast()
                    }
                }
            }
        }

        let topK = min(k, results.count)
        return results.prefix(topK).map { result in
            SearchResult(id: "", score: result.distance, internalID: result.nodeID)
        }
    }

    private static func computeDistancesOnGPU(
        context: MetalContext,
        query: [Float],
        vectors: VectorBuffer,
        neighborIDs: [UInt32],
        metric: Metric
    ) async throws -> [Float] {
        guard !neighborIDs.isEmpty else {
            return []
        }

        let functionName: String = switch metric {
        case .cosine:
            "cosine_distance"
        case .l2:
            "l2_distance"
        case .innerProduct:
            "inner_product_distance"
        }

        let pipeline = try await context.pipelineCache.pipeline(for: functionName)

        var neighborVectors: [Float] = []
        neighborVectors.reserveCapacity(neighborIDs.count * vectors.dim)
        for neighborID in neighborIDs {
            neighborVectors.append(contentsOf: vectors.vector(at: Int(neighborID)))
        }

        let scalarSize = MemoryLayout<Float>.stride
        let queryLength = query.count * scalarSize
        let neighborLength = neighborVectors.count * scalarSize
        let outputLength = neighborIDs.count * scalarSize

        guard
            let queryBuffer = context.device.makeBuffer(
                bytes: query,
                length: queryLength,
                options: .storageModeShared
            ),
            let neighborBuffer = context.device.makeBuffer(
                bytes: neighborVectors,
                length: neighborLength,
                options: .storageModeShared
            ),
            let outputBuffer = context.device.makeBuffer(
                length: outputLength,
                options: .storageModeShared
            )
        else {
            throw ANNSError.searchFailed("Failed to allocate search distance buffers")
        }

        var dim = UInt32(vectors.dim)
        var count = UInt32(neighborIDs.count)

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.searchFailed("Failed to create compute command encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(neighborBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 4)

            let threadsPerGrid = MTLSize(width: neighborIDs.count, height: 1, depth: 1)
            let threadgroupWidth = max(1, min(neighborIDs.count, pipeline.maxTotalThreadsPerThreadgroup))
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: neighborIDs.count)
        return Array(UnsafeBufferPointer(start: outputPointer, count: neighborIDs.count))
    }

    private static func insertSorted(_ candidate: Candidate, into list: inout [Candidate]) {
        var insertionIndex = list.endIndex
        for index in list.indices where candidate.distance < list[index].distance {
            insertionIndex = index
            break
        }
        list.insert(candidate, at: insertionIndex)
    }

    private static func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
        switch metric {
        case .cosine:
            var dot: Float = 0
            var normQ: Float = 0
            var normV: Float = 0
            for d in 0..<query.count {
                let q = query[d]
                let v = vector[d]
                dot += q * v
                normQ += q * q
                normV += v * v
            }
            let denom = sqrt(normQ) * sqrt(normV)
            return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
        case .l2:
            var sum: Float = 0
            for d in 0..<query.count {
                let diff = query[d] - vector[d]
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
            }
            return -dot
        }
    }
}
