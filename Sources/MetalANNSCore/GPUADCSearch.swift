import Foundation
import Metal

public enum GPUADCSearch {
    public static func computeDistances(
        context: MetalContext,
        query: [Float],
        pq: ProductQuantizer,
        codes: [[UInt8]],
        flatCodebooks: [Float]? = nil
    ) async throws -> [Float] {
        let expectedDimension = pq.numSubspaces * pq.subspaceDimension
        guard query.count == expectedDimension else {
            throw ANNSError.dimensionMismatch(expected: expectedDimension, got: query.count)
        }
        guard !codes.isEmpty else {
            return []
        }

        let m = pq.numSubspaces
        let ks = pq.centroidsPerSubspace
        let subspaceDim = pq.subspaceDimension

        var candidateCodes: [UInt8] = []
        candidateCodes.reserveCapacity(codes.count * m)
        for (index, code) in codes.enumerated() {
            guard code.count == m else {
                throw ANNSError.searchFailed("Invalid PQ code size at index \(index)")
            }
            candidateCodes.append(contentsOf: code)
        }

        let flattenedCodebooks = flatCodebooks ?? flattenCodebooks(from: pq)
        let expectedCodebookCount = m * ks * subspaceDim
        guard flattenedCodebooks.count == expectedCodebookCount else {
            throw ANNSError.searchFailed(
                "Invalid flattened codebook size. Expected \(expectedCodebookCount), got \(flattenedCodebooks.count)"
            )
        }

        let tableLengthBytes = m * ks * MemoryLayout<Float>.stride
        let distancesLengthBytes = codes.count * MemoryLayout<Float>.stride
        let codesLengthBytes = candidateCodes.count * MemoryLayout<UInt8>.stride

        guard
            let queryBuffer = context.device.makeBuffer(
                bytes: query,
                length: query.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ),
            let codebookBuffer = context.device.makeBuffer(
                bytes: flattenedCodebooks,
                length: flattenedCodebooks.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ),
            let distanceTableBuffer = context.device.makeBuffer(
                length: tableLengthBytes,
                options: .storageModeShared
            ),
            let codesBuffer = context.device.makeBuffer(
                bytes: candidateCodes,
                length: codesLengthBytes,
                options: .storageModeShared
            ),
            let distancesBuffer = context.device.makeBuffer(
                length: distancesLengthBytes,
                options: .storageModeShared
            )
        else {
            throw ANNSError.constructionFailed("Failed to allocate Metal buffers for GPU ADC")
        }

        let tablePipeline = try await context.pipelineCache.pipeline(for: "pq_compute_distance_table")
        let scanPipeline = try await context.pipelineCache.pipeline(for: "pq_adc_scan")

        var mU32 = UInt32(m)
        var ksU32 = UInt32(ks)
        var subspaceDimU32 = UInt32(subspaceDim)
        var vectorCountU32 = UInt32(codes.count)

        try await context.execute { commandBuffer in
            guard let tableEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create distance-table encoder")
            }

            tableEncoder.setComputePipelineState(tablePipeline)
            tableEncoder.setBuffer(queryBuffer, offset: 0, index: 0)
            tableEncoder.setBuffer(codebookBuffer, offset: 0, index: 1)
            tableEncoder.setBuffer(distanceTableBuffer, offset: 0, index: 2)
            tableEncoder.setBytes(&mU32, length: MemoryLayout<UInt32>.stride, index: 3)
            tableEncoder.setBytes(&ksU32, length: MemoryLayout<UInt32>.stride, index: 4)
            tableEncoder.setBytes(&subspaceDimU32, length: MemoryLayout<UInt32>.stride, index: 5)

            let tableGrid = MTLSize(width: m, height: ks, depth: 1)
            let tableThreads = MTLSize(width: 8, height: 8, depth: 1)
            tableEncoder.dispatchThreads(tableGrid, threadsPerThreadgroup: tableThreads)
            tableEncoder.endEncoding()

            guard let scanEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.constructionFailed("Failed to create ADC scan encoder")
            }

            scanEncoder.setComputePipelineState(scanPipeline)
            scanEncoder.setBuffer(codesBuffer, offset: 0, index: 0)
            scanEncoder.setBuffer(distanceTableBuffer, offset: 0, index: 1)
            scanEncoder.setBuffer(distancesBuffer, offset: 0, index: 2)
            scanEncoder.setBytes(&mU32, length: MemoryLayout<UInt32>.stride, index: 3)
            scanEncoder.setBytes(&ksU32, length: MemoryLayout<UInt32>.stride, index: 4)
            scanEncoder.setBytes(&vectorCountU32, length: MemoryLayout<UInt32>.stride, index: 5)
            scanEncoder.setThreadgroupMemoryLength(tableLengthBytes, index: 0)

            let scanGrid = MTLSize(width: codes.count, height: 1, depth: 1)
            let scanThreadWidth = max(
                1,
                min(codes.count, scanPipeline.maxTotalThreadsPerThreadgroup)
            )
            let scanThreads = MTLSize(width: scanThreadWidth, height: 1, depth: 1)
            scanEncoder.dispatchThreads(scanGrid, threadsPerThreadgroup: scanThreads)
            scanEncoder.endEncoding()
        }

        let base = distancesBuffer.contents().bindMemory(to: Float.self, capacity: codes.count)
        return Array(UnsafeBufferPointer(start: base, count: codes.count))
    }

    public static func search(
        context: MetalContext,
        query: [Float],
        pq: ProductQuantizer,
        codes: [[UInt8]],
        ids: [String],
        k: Int,
        flatCodebooks: [Float]? = nil
    ) async throws -> [SearchResult] {
        throw ANNSError.searchFailed("GPUADCSearch.search is not implemented")
    }

    public static func flattenCodebooks(from pq: ProductQuantizer) -> [Float] {
        var flattened: [Float] = []
        flattened.reserveCapacity(
            pq.numSubspaces * pq.centroidsPerSubspace * pq.subspaceDimension
        )
        for subspace in 0..<pq.numSubspaces {
            for centroid in 0..<pq.centroidsPerSubspace {
                flattened.append(contentsOf: pq.codebooks[subspace][centroid])
            }
        }
        return flattened
    }
}
