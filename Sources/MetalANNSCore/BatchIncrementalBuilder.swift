import Foundation

public enum BatchIncrementalBuilder {
    private struct Candidate {
        let nodeID: UInt32
        let distance: Float
    }

    /// Insert multiple vectors at once using:
    /// 1) sequential forward neighbor assignment and
    /// 2) one deferred batched reverse-update pass.
    public static func batchInsert(
        vectors newVectors: [[Float]],
        startingAt startSlot: Int,
        into graph: GraphBuffer,
        vectorStorage: any VectorStorage,
        entryPoint: UInt32,
        metric: Metric,
        degree: Int
    ) throws(ANNSError) {
        guard !newVectors.isEmpty else {
            return
        }
        guard startSlot >= 0 else {
            throw ANNSError.constructionFailed("Start slot must be non-negative")
        }
        guard degree > 0 else {
            throw ANNSError.constructionFailed("Degree must be greater than 0")
        }
        guard startSlot + newVectors.count <= graph.capacity,
              startSlot + newVectors.count <= vectorStorage.capacity else {
            throw ANNSError.constructionFailed("Batch insert exceeds capacity")
        }

        let existingNodeCount = graph.nodeCount
        guard existingNodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot batch insert into empty graph")
        }
        guard entryPoint < graph.capacity, Int(entryPoint) < max(1, existingNodeCount) else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }

        let graphDegree = graph.degree
        var newNodeNeighborLists: [(internalID: UInt32, neighbors: [(nodeID: UInt32, distance: Float)])] = []
        newNodeNeighborLists.reserveCapacity(newVectors.count)

        for (offset, vector) in newVectors.enumerated() {
            guard vector.count == vectorStorage.dim else {
                throw ANNSError.dimensionMismatch(expected: vectorStorage.dim, got: vector.count)
            }

            let internalID = startSlot + offset
            let nearest = nearestNeighbors(
                to: vector,
                graph: graph,
                vectors: vectorStorage,
                entryPoint: Int(entryPoint),
                degree: graphDegree,
                metric: metric
            )

            var neighborIDs = Array(repeating: UInt32.max, count: graphDegree)
            var neighborDistances = Array(repeating: Float.greatestFiniteMagnitude, count: graphDegree)
            for slot in 0..<min(graphDegree, nearest.count) {
                neighborIDs[slot] = nearest[slot].nodeID
                neighborDistances[slot] = nearest[slot].distance
            }

            try graph.setNeighbors(of: internalID, ids: neighborIDs, distances: neighborDistances)
            if graph.nodeCount < internalID + 1 {
                graph.setCount(internalID + 1)
            }

            newNodeNeighborLists.append((UInt32(internalID), nearest))
        }

        for (offset, newNode) in newNodeNeighborLists.enumerated() {
            let newVector = newVectors[offset]
            let newNodeID = newNode.internalID
            var attached = false

            for neighbor in newNode.neighbors {
                let neighborIndex = Int(neighbor.nodeID)
                if neighborIndex < 0 || neighborIndex >= graph.nodeCount {
                    continue
                }

                let existingIDs = graph.neighborIDs(of: neighborIndex)
                var existingDistances = graph.neighborDistances(of: neighborIndex)
                if existingIDs.contains(newNodeID) {
                    attached = true
                    continue
                }

                let candidateDistance = SIMDDistance.distance(
                    newVector,
                    vectorStorage.vector(at: neighborIndex),
                    metric: metric
                )

                if let replaceIndex = worstNeighborIndex(in: existingDistances),
                   candidateDistance < existingDistances[replaceIndex] {
                    var updatedIDs = existingIDs
                    updatedIDs[replaceIndex] = newNodeID
                    existingDistances[replaceIndex] = candidateDistance

                    let sorted = zip(updatedIDs, existingDistances).sorted { lhs, rhs in
                        lhs.1 < rhs.1
                    }
                    try graph.setNeighbors(
                        of: neighborIndex,
                        ids: sorted.map(\.0),
                        distances: sorted.map(\.1)
                    )
                    attached = true
                }
            }

            if !attached {
                let fallbackIndex = Int(entryPoint)
                guard fallbackIndex >= 0 && fallbackIndex < graph.nodeCount else {
                    throw ANNSError.searchFailed("Fallback index out of bounds")
                }

                let existingIDs = graph.neighborIDs(of: fallbackIndex)
                if !existingIDs.contains(newNodeID) {
                    var existingDistances = graph.neighborDistances(of: fallbackIndex)
                    if let replaceIndex = worstNeighborIndex(in: existingDistances) {
                        let fallbackDistance = SIMDDistance.distance(
                            newVector,
                            vectorStorage.vector(at: fallbackIndex),
                            metric: metric
                        )

                        var updatedIDs = existingIDs
                        updatedIDs[replaceIndex] = newNodeID
                        existingDistances[replaceIndex] = fallbackDistance

                        let sorted = zip(updatedIDs, existingDistances).sorted { lhs, rhs in
                            lhs.1 < rhs.1
                        }
                        try graph.setNeighbors(
                            of: fallbackIndex,
                            ids: sorted.map(\.0),
                            distances: sorted.map(\.1)
                        )
                    }
                }
            }
        }

        if graph.nodeCount < startSlot + newVectors.count {
            graph.setCount(startSlot + newVectors.count)
        }
    }

    private static func nearestNeighbors(
        to vector: [Float],
        graph: GraphBuffer,
        vectors: any VectorStorage,
        entryPoint: Int,
        degree: Int,
        metric: Metric
    ) -> [(nodeID: UInt32, distance: Float)] {
        let nodeCount = graph.nodeCount
        guard nodeCount > 0 else {
            return []
        }

        let ef = max(1, degree * 2)
        let efLimit = min(nodeCount, ef)

        let entryDistance = SIMDDistance.distance(
            vector,
            vectors.vector(at: entryPoint),
            metric: metric
        )
        var visited: Set<UInt32> = [UInt32(entryPoint)]
        var candidates: [Candidate] = [Candidate(nodeID: UInt32(entryPoint), distance: entryDistance)]
        var results: [Candidate] = [Candidate(nodeID: UInt32(entryPoint), distance: entryDistance)]

        while !candidates.isEmpty {
            let current = candidates.removeFirst()
            if results.count >= efLimit, let worst = results.last, current.distance > worst.distance {
                break
            }

            let neighborIDs = graph.neighborIDs(of: Int(current.nodeID))
            for neighborID in neighborIDs {
                let neighborIndex = Int(neighborID)
                if neighborID == UInt32.max || neighborIndex < 0 || neighborIndex >= nodeCount {
                    continue
                }
                if visited.contains(neighborID) {
                    continue
                }
                visited.insert(neighborID)

                let distance = SIMDDistance.distance(
                    vector,
                    vectors.vector(at: neighborIndex),
                    metric: metric
                )
                if results.count < efLimit || distance < results[results.count - 1].distance {
                    let candidate = Candidate(nodeID: neighborID, distance: distance)
                    insertSorted(candidate, into: &candidates)
                    insertSorted(candidate, into: &results)
                    if results.count > efLimit {
                        results.removeLast()
                    }
                }
            }
        }

        let selected = results.prefix(degree).filter { $0.nodeID != UInt32.max }
        return selected.map { ($0.nodeID, $0.distance) }
    }

    private static func worstNeighborIndex(in distances: [Float]) -> Int? {
        guard !distances.isEmpty else {
            return nil
        }

        var worstIndex = 0
        var worstDistance = distances[0]
        for index in 1..<distances.count {
            if distances[index] > worstDistance {
                worstDistance = distances[index]
                worstIndex = index
            }
        }
        return worstIndex
    }

    private static func insertSorted(_ candidate: Candidate, into list: inout [Candidate]) {
        var insertionIndex = list.endIndex
        for index in list.indices where candidate.distance < list[index].distance {
            insertionIndex = index
            break
        }
        list.insert(candidate, at: insertionIndex)
    }
}
