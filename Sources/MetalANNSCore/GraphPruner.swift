import Foundation

public enum GraphPruner {
    /// Prune redundant edges from the graph using path-based diversification.
    /// For each node, removes neighbors that are reachable via shorter paths
    /// through other neighbors.
    ///
    /// FIXME: Major — candidates are iterated in adjacency-slot order, not sorted
    /// by distance. Since pruning is order-dependent, unsorted input may keep a
    /// farther neighbor and prune a closer one. Sort candidates by distance first.
    public static func prune(
        graph: GraphBuffer,
        vectors: any VectorStorage,
        nodeCount: Int,
        metric: Metric
    ) throws {
        guard nodeCount >= 0, nodeCount <= graph.nodeCount, nodeCount <= vectors.count else {
            throw ANNSError.constructionFailed("nodeCount is out of bounds for graph/vector counts")
        }

        let degree = graph.degree
        guard degree > 0 else {
            return
        }

        if let floatVectors = vectors as? VectorBuffer {
            try pruneFloat32(
                graph: graph,
                vectors: floatVectors,
                nodeCount: nodeCount,
                metric: metric
            )
            return
        }

        for nodeID in 0..<nodeCount {
            if nodeID & 1023 == 0 {
                try Task.checkCancellation()
            }
            let neighborIDs = graph.neighborIDs(of: nodeID)
            let neighborDistances = graph.neighborDistances(of: nodeID)

            var candidates: [(id: UInt32, distance: Float, vector: [Float])] = []
            candidates.reserveCapacity(degree)
            for slot in 0..<degree {
                let neighborID = neighborIDs[slot]
                if neighborID == UInt32.max {
                    continue
                }
                let neighborIndex = Int(neighborID)
                if neighborIndex < 0 || neighborIndex >= nodeCount {
                    continue
                }
                candidates.append((neighborID, neighborDistances[slot], vectors.vector(at: neighborIndex)))
            }

            var prunedIDs: [UInt32] = []
            var prunedDistances: [Float] = []
            var prunedVectors: [[Float]] = []
            prunedIDs.reserveCapacity(degree)
            prunedDistances.reserveCapacity(degree)
            prunedVectors.reserveCapacity(degree)

            for candidate in candidates {
                var isRedundant = false

                for selectedVector in prunedVectors {
                    let pathDistance = SIMDDistance.distance(
                        selectedVector,
                        candidate.vector,
                        metric: metric
                    )
                    if pathDistance < candidate.distance {
                        isRedundant = true
                        break
                    }
                }

                if !isRedundant {
                    prunedIDs.append(candidate.id)
                    prunedDistances.append(candidate.distance)
                    prunedVectors.append(candidate.vector)
                }
            }

            if prunedIDs.count < degree {
                prunedIDs.append(contentsOf: Array(repeating: UInt32.max, count: degree - prunedIDs.count))
                prunedDistances.append(
                    contentsOf: Array(
                        repeating: Float.greatestFiniteMagnitude,
                        count: degree - prunedDistances.count
                    ))
            } else if prunedIDs.count > degree {
                prunedIDs = Array(prunedIDs.prefix(degree))
                prunedDistances = Array(prunedDistances.prefix(degree))
            }

            try graph.setNeighbors(of: nodeID, ids: prunedIDs, distances: prunedDistances)
        }
    }

    /// Nodes pruned per parallel pass; cancellation is checked between passes.
    private static let parallelBatchSize = 16_384

    /// Float32 fast path. Each node's pruned list depends only on its own
    /// neighbour list and the (read-only) vectors, so nodes are pruned in
    /// parallel into scratch memory, then written back on the calling thread.
    /// The result is identical to a sequential pass.
    private static func pruneFloat32(
        graph: GraphBuffer,
        vectors: VectorBuffer,
        nodeCount: Int,
        metric: Metric
    ) throws {
        let degree = graph.degree
        let dim = vectors.dim
        guard nodeCount > 0, let vectorBase = vectors.floatPointer.baseAddress else {
            return
        }
        let slotCount = graph.capacity * degree
        let adjacency = UnsafePointer(
            graph.adjacencyBuffer.contents().bindMemory(to: UInt32.self, capacity: slotCount)
        )
        let edgeDistances = UnsafePointer(
            graph.distanceBuffer.contents().bindMemory(to: Float.self, capacity: slotCount)
        )

        let batchSize = min(parallelBatchSize, nodeCount)
        let prunedIDs = UnsafeMutableBufferPointer<UInt32>.allocate(capacity: batchSize * degree)
        let prunedDistances = UnsafeMutableBufferPointer<Float>.allocate(capacity: batchSize * degree)
        defer {
            prunedIDs.deallocate()
            prunedDistances.deallocate()
        }
        let processorCount = ProcessInfo.processInfo.activeProcessorCount
        let work = PruneWork(
            vectors: vectorBase,
            adjacency: adjacency,
            edgeDistances: edgeDistances,
            outIDs: prunedIDs.baseAddress!,
            outDistances: prunedDistances.baseAddress!,
            nodeCount: nodeCount,
            degree: degree,
            dim: dim,
            metric: metric
        )

        var batchStart = 0
        while batchStart < nodeCount {
            try Task.checkCancellation()
            let batchEnd = min(nodeCount, batchStart + batchSize)
            let batchCount = batchEnd - batchStart
            let workers = max(1, min(processorCount, batchCount / 256))
            let chunk = (batchCount + workers - 1) / workers
            let firstNode = batchStart

            DispatchQueue.concurrentPerform(iterations: workers) { worker in
                let start = firstNode + worker * chunk
                let end = min(firstNode + batchCount, start + chunk)
                guard start < end else { return }
                for nodeID in start..<end {
                    work.prune(nodeID, scratchRow: nodeID - firstNode)
                }
            }

            for nodeID in batchStart..<batchEnd {
                let offset = (nodeID - batchStart) * degree
                try graph.setNeighbors(
                    of: nodeID,
                    ids: Array(prunedIDs[offset..<(offset + degree)]),
                    distances: Array(prunedDistances[offset..<(offset + degree)])
                )
            }
            batchStart = batchEnd
        }
    }

    /// Scoped work descriptor for one parallel prune pass.
    ///
    /// `@unchecked Sendable` is sound here (same contract as
    /// `ParallelFlatScan.SliceWork`): the pointers reference memory owned by
    /// `pruneFloat32` that outlives the concurrentPerform join, inputs are only
    /// read, each worker writes only its own scratch rows, and no field mutates
    /// after construction.
    private final class PruneWork: @unchecked Sendable {
        let vectors: UnsafePointer<Float>
        let adjacency: UnsafePointer<UInt32>
        let edgeDistances: UnsafePointer<Float>
        let outIDs: UnsafeMutablePointer<UInt32>
        let outDistances: UnsafeMutablePointer<Float>
        let nodeCount: Int
        let degree: Int
        let dim: Int
        let metric: Metric

        init(
            vectors: UnsafePointer<Float>,
            adjacency: UnsafePointer<UInt32>,
            edgeDistances: UnsafePointer<Float>,
            outIDs: UnsafeMutablePointer<UInt32>,
            outDistances: UnsafeMutablePointer<Float>,
            nodeCount: Int,
            degree: Int,
            dim: Int,
            metric: Metric
        ) {
            self.vectors = vectors
            self.adjacency = adjacency
            self.edgeDistances = edgeDistances
            self.outIDs = outIDs
            self.outDistances = outDistances
            self.nodeCount = nodeCount
            self.degree = degree
            self.dim = dim
            self.metric = metric
        }

        /// Prunes `nodeID`'s list into scratch row `scratchRow` (`degree` slots,
        /// sentinel-padded). Keeps a candidate unless an already-kept neighbour
        /// is closer to it than the node is.
        func prune(_ nodeID: Int, scratchRow: Int) {
            let base = nodeID * degree
            let keptIDs = outIDs + scratchRow * degree
            let keptDistances = outDistances + scratchRow * degree
            var keptCount = 0
            for slot in 0..<degree {
                let candidateID = adjacency[base + slot]
                if candidateID == UInt32.max || Int(candidateID) >= nodeCount {
                    continue
                }
                let candidateDistance = edgeDistances[base + slot]
                let candidateBase = vectors + Int(candidateID) * dim
                var isRedundant = false
                for kept in 0..<keptCount {
                    let pathDistance = SIMDDistance.distance(
                        vectors + Int(keptIDs[kept]) * dim,
                        candidateBase,
                        dim: dim,
                        metric: metric
                    )
                    if pathDistance < candidateDistance {
                        isRedundant = true
                        break
                    }
                }
                if !isRedundant {
                    keptIDs[keptCount] = candidateID
                    keptDistances[keptCount] = candidateDistance
                    keptCount += 1
                }
            }
            for slot in keptCount..<degree {
                keptIDs[slot] = UInt32.max
                keptDistances[slot] = Float.greatestFiniteMagnitude
            }
        }
    }
}
