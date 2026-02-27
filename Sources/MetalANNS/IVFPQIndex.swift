import Foundation
import Metal
import MetalANNSCore

public actor IVFPQIndex: Sendable {
    private let config: IVFPQConfiguration
    private let capacity: Int
    private let dimension: Int
    private let context: MetalContext?

    private var coarseCentroids: [[Float]] = []
    private var pq: ProductQuantizer?
    private var flattenedCodebooks: [Float] = []
    private var vectorBuffer: PQVectorBuffer?
    private var invertedLists: [[UInt32]] = []
    private var coarseAssignments: [UInt32] = []
    private var idMap = IDMap()
    private var isTrained = false

    public init(capacity: Int, dimension: Int, config: IVFPQConfiguration) throws {
        guard capacity > 0 else {
            throw ANNSError.constructionFailed("IVFPQIndex capacity must be greater than zero")
        }
        guard dimension > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: dimension)
        }
        guard config.numCentroids == 256 else {
            throw ANNSError.constructionFailed("IVFPQIndex requires numCentroids == 256 (UInt8 codes)")
        }

        self.capacity = capacity
        self.dimension = dimension
        self.config = config
        self.context = try? MetalContext()
    }

    public func train(vectors: [[Float]]) async throws {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot train IVFPQIndex with empty vectors")
        }
        guard vectors.count >= config.numCentroids else {
            throw ANNSError.constructionFailed("Need at least \(config.numCentroids) vectors for PQ training")
        }
        guard vectors.count >= config.numCoarseCentroids else {
            throw ANNSError.constructionFailed(
                "Need at least \(config.numCoarseCentroids) vectors for coarse quantizer"
            )
        }

        for vector in vectors where vector.count != dimension {
            throw ANNSError.dimensionMismatch(expected: dimension, got: vector.count)
        }

        let coarseResult = try KMeans.cluster(
            vectors: vectors,
            k: config.numCoarseCentroids,
            maxIterations: config.trainingIterations,
            metric: config.metric,
            seed: 42
        )
        let trainedCoarseCentroids = coarseResult.centroids

        var residuals: [[Float]] = []
        residuals.reserveCapacity(vectors.count)
        for (vectorIndex, vector) in vectors.enumerated() {
            let cluster = coarseResult.assignments[vectorIndex]
            let centroid = trainedCoarseCentroids[cluster]
            residuals.append(residual(vector: vector, centroid: centroid))
        }

        let trainedPQ = try ProductQuantizer.train(
            vectors: residuals,
            numSubspaces: config.numSubspaces,
            centroidsPerSubspace: config.numCentroids,
            maxIterations: config.trainingIterations
        )

        self.coarseCentroids = trainedCoarseCentroids
        self.pq = trainedPQ
        self.flattenedCodebooks = flattenCodebooks(from: trainedPQ)
        self.vectorBuffer = try PQVectorBuffer(capacity: capacity, dim: dimension, pq: trainedPQ)
        self.invertedLists = Array(repeating: [], count: trainedCoarseCentroids.count)
        self.coarseAssignments = []
        self.idMap = IDMap()
        self.isTrained = true
    }

    public func add(vectors: [[Float]], ids: [String]) async throws {
        guard isTrained, let vectorBuffer else {
            throw ANNSError.constructionFailed("IVFPQIndex must be trained before add()")
        }
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }
        guard idMap.count + vectors.count <= capacity else {
            throw ANNSError.constructionFailed("Index capacity exceeded")
        }

        var seen = Set<String>()
        for id in ids {
            if !seen.insert(id).inserted || idMap.internalID(for: id) != nil {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        for vector in vectors where vector.count != dimension {
            throw ANNSError.dimensionMismatch(expected: dimension, got: vector.count)
        }

        for (offset, vector) in vectors.enumerated() {
            guard let internalID = idMap.assign(externalID: ids[offset]) else {
                throw ANNSError.idAlreadyExists(ids[offset])
            }

            let cluster = nearestCoarseCentroid(for: vector)
            let centroid = coarseCentroids[cluster]
            let residualVector = residual(vector: vector, centroid: centroid)

            try vectorBuffer.insert(vector: residualVector, at: Int(internalID))
            invertedLists[cluster].append(internalID)
            coarseAssignments.append(UInt32(cluster))
        }
    }

    public func search(query: [Float], k: Int, nprobe: Int? = nil) async -> [SearchResult] {
        await search(query: query, k: k, nprobe: nprobe, forceGPU: nil)
    }

    func search(query: [Float], k: Int, nprobe: Int? = nil, forceGPU: Bool?) async -> [SearchResult] {
        guard isTrained, let vectorBuffer else {
            return []
        }
        guard k > 0 else {
            return []
        }
        guard query.count == dimension else {
            return []
        }
        guard !coarseCentroids.isEmpty else {
            return []
        }

        let probeCount = max(1, min(nprobe ?? config.nprobe, coarseCentroids.count))
        let centroidScores = coarseCentroids.enumerated().map { (index, centroid) in
            (index, SIMDDistance.distance(query, centroid, metric: config.metric))
        }.sorted { $0.1 < $1.1 }

        var merged: [SearchResult] = []
        merged.reserveCapacity(probeCount * max(k, 16))

        guard let pq else {
            return []
        }

        for (clusterIndex, _) in centroidScores.prefix(probeCount) {
            let queryResidual = residual(vector: query, centroid: coarseCentroids[clusterIndex])
            let candidateIDs = invertedLists[clusterIndex]
            let scores = await distancesForCluster(
                queryResidual: queryResidual,
                candidateIDs: candidateIDs,
                forceGPU: forceGPU,
                pq: pq,
                vectorBuffer: vectorBuffer
            )

            for (offset, internalID) in candidateIDs.enumerated() {
                let score = scores[offset]
                guard let externalID = idMap.externalID(for: internalID) else {
                    continue
                }
                merged.append(SearchResult(id: externalID, score: score, internalID: internalID))
            }
        }

        merged.sort { $0.score < $1.score }
        if merged.count > k {
            merged.removeSubrange(k...)
        }
        return merged
    }

    public var count: Int {
        idMap.count
    }

    public func estimatedVectorCodeBytes() -> Int {
        vectorBuffer?.compressedCodeBytes ?? 0
    }

    public func estimatedMemoryBytes() -> Int {
        let coarseBytes = coarseCentroids.count * dimension * MemoryLayout<Float>.stride
        let codebookBytes: Int = if let pq {
            pq.numSubspaces * pq.centroidsPerSubspace * pq.subspaceDimension * MemoryLayout<Float>.stride
        } else {
            0
        }
        let vectorCodeBytes = vectorBuffer?.compressedCodeBytes ?? 0
        let invertedListBytes = invertedLists.reduce(0) { partial, list in
            partial + (list.count * MemoryLayout<UInt32>.stride)
        }
        let assignmentBytes = coarseAssignments.count * MemoryLayout<UInt32>.stride
        return coarseBytes + codebookBytes + vectorCodeBytes + invertedListBytes + assignmentBytes
    }

    func gpuAvailable() -> Bool {
        context != nil
    }

    func firstNonEmptyCluster() -> Int? {
        invertedLists.firstIndex(where: { !$0.isEmpty })
    }

    func debugClusterDistances(query: [Float], clusterIndex: Int, forceGPU: Bool) async throws -> [Float] {
        guard let pq, let vectorBuffer, isTrained else {
            throw ANNSError.searchFailed("Index must be trained before debug distance queries")
        }
        guard query.count == dimension else {
            throw ANNSError.dimensionMismatch(expected: dimension, got: query.count)
        }
        guard clusterIndex >= 0, clusterIndex < coarseCentroids.count else {
            throw ANNSError.searchFailed("Cluster index out of range")
        }

        let residualQuery = residual(vector: query, centroid: coarseCentroids[clusterIndex])
        let candidateIDs = invertedLists[clusterIndex]
        return await distancesForCluster(
            queryResidual: residualQuery,
            candidateIDs: candidateIDs,
            forceGPU: forceGPU,
            pq: pq,
            vectorBuffer: vectorBuffer
        )
    }

    private func nearestCoarseCentroid(for vector: [Float]) -> Int {
        var bestIndex = 0
        var bestDistance = Float.greatestFiniteMagnitude

        for (index, centroid) in coarseCentroids.enumerated() {
            let distance = SIMDDistance.distance(vector, centroid, metric: config.metric)
            if distance < bestDistance {
                bestDistance = distance
                bestIndex = index
            }
        }
        return bestIndex
    }

    private func residual(vector: [Float], centroid: [Float]) -> [Float] {
        zip(vector, centroid).map { $0 - $1 }
    }

    private func distancesForCluster(
        queryResidual: [Float],
        candidateIDs: [UInt32],
        forceGPU: Bool?,
        pq: ProductQuantizer,
        vectorBuffer: PQVectorBuffer
    ) async -> [Float] {
        guard !candidateIDs.isEmpty else {
            return []
        }

        if forceGPU == false {
            return cpuADCDistances(
                queryResidual: queryResidual,
                candidateIDs: candidateIDs,
                pq: pq,
                vectorBuffer: vectorBuffer
            )
        }

        let shouldTryGPU: Bool
        if forceGPU == true {
            shouldTryGPU = true
        } else {
            shouldTryGPU = context != nil && candidateIDs.count >= 64
        }

        if shouldTryGPU {
            do {
                return try await gpuADCDistances(
                    queryResidual: queryResidual,
                    candidateIDs: candidateIDs,
                    pq: pq,
                    vectorBuffer: vectorBuffer
                )
            } catch {
                // Fall back to CPU path for robustness.
            }
        }

        return cpuADCDistances(
            queryResidual: queryResidual,
            candidateIDs: candidateIDs,
            pq: pq,
            vectorBuffer: vectorBuffer
        )
    }

    private func cpuADCDistances(
        queryResidual: [Float],
        candidateIDs: [UInt32],
        pq: ProductQuantizer,
        vectorBuffer: PQVectorBuffer
    ) -> [Float] {
        guard let table = makeDistanceTable(query: queryResidual, pq: pq, metric: config.metric) else {
            return [Float](repeating: Float.greatestFiniteMagnitude, count: candidateIDs.count)
        }

        var distances: [Float] = []
        distances.reserveCapacity(candidateIDs.count)
        for internalID in candidateIDs {
            let code = vectorBuffer.code(at: Int(internalID))
            guard code.count == pq.numSubspaces else {
                distances.append(Float.greatestFiniteMagnitude)
                continue
            }

            var total: Float = 0
            for subspace in 0..<pq.numSubspaces {
                total += table[subspace][Int(code[subspace])]
            }
            distances.append(total)
        }
        return distances
    }

    private func gpuADCDistances(
        queryResidual: [Float],
        candidateIDs: [UInt32],
        pq: ProductQuantizer,
        vectorBuffer: PQVectorBuffer
    ) async throws -> [Float] {
        guard let context else {
            throw ANNSError.searchFailed("Metal context unavailable for GPU ADC")
        }
        guard queryResidual.count == dimension else {
            throw ANNSError.dimensionMismatch(expected: dimension, got: queryResidual.count)
        }
        guard !candidateIDs.isEmpty else {
            return []
        }

        let m = pq.numSubspaces
        let ks = pq.centroidsPerSubspace
        let subspaceDim = pq.subspaceDimension

        var candidateCodes: [UInt8] = []
        candidateCodes.reserveCapacity(candidateIDs.count * m)
        for internalID in candidateIDs {
            let code = vectorBuffer.code(at: Int(internalID))
            guard code.count == m else {
                throw ANNSError.searchFailed("Invalid PQ code size for internal ID \(internalID)")
            }
            candidateCodes.append(contentsOf: code)
        }

        if flattenedCodebooks.isEmpty {
            flattenedCodebooks = flattenCodebooks(from: pq)
        }

        let tableLengthBytes = m * ks * MemoryLayout<Float>.stride
        let distancesLengthBytes = candidateIDs.count * MemoryLayout<Float>.stride
        let codesLengthBytes = candidateCodes.count * MemoryLayout<UInt8>.stride

        guard
            let queryBuffer = context.device.makeBuffer(
                bytes: queryResidual,
                length: queryResidual.count * MemoryLayout<Float>.stride,
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
        var vectorCountU32 = UInt32(candidateIDs.count)

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

            let scanGrid = MTLSize(width: candidateIDs.count, height: 1, depth: 1)
            let scanThreadWidth = max(
                1,
                min(candidateIDs.count, scanPipeline.maxTotalThreadsPerThreadgroup)
            )
            let scanThreads = MTLSize(width: scanThreadWidth, height: 1, depth: 1)
            scanEncoder.dispatchThreads(scanGrid, threadsPerThreadgroup: scanThreads)
            scanEncoder.endEncoding()
        }

        let base = distancesBuffer.contents().bindMemory(to: Float.self, capacity: candidateIDs.count)
        return Array(UnsafeBufferPointer(start: base, count: candidateIDs.count))
    }

    private func flattenCodebooks(from pq: ProductQuantizer) -> [Float] {
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

    private func makeDistanceTable(
        query: [Float],
        pq: ProductQuantizer,
        metric: Metric
    ) -> [[Float]]? {
        let expectedDimension = pq.numSubspaces * pq.subspaceDimension
        guard query.count == expectedDimension else {
            return nil
        }

        var table = [[Float]](
            repeating: [Float](repeating: 0, count: pq.centroidsPerSubspace),
            count: pq.numSubspaces
        )

        for subspace in 0..<pq.numSubspaces {
            let start = subspace * pq.subspaceDimension
            let end = start + pq.subspaceDimension
            let subQuery = Array(query[start..<end])
            for centroid in 0..<pq.centroidsPerSubspace {
                table[subspace][centroid] = SIMDDistance.distance(
                    subQuery,
                    pq.codebooks[subspace][centroid],
                    metric: metric
                )
            }
        }
        return table
    }
}
