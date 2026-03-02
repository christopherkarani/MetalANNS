import Foundation

public enum BeamSearchCPU {
    private struct Candidate {
        let nodeID: UInt32
        let distance: Float
    }

    public static func search(
        query: [Float],
        vectors: any VectorStorage,
        graph: [[(UInt32, Float)]],
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> [SearchResult] {
        guard vectors.count > 0 else {
            throw ANNSError.indexEmpty
        }
        guard graph.count == vectors.count else {
            throw ANNSError.searchFailed("Graph size does not match vector count")
        }
        return try await searchImpl(
            query: query,
            vectorCount: vectors.count,
            dim: vectors.dim,
            graph: graph,
            entryPoint: entryPoint,
            k: k,
            ef: ef,
            metric: metric,
            vectorAt: { vectors.vector(at: $0) }
        )
    }

    public static func search(
        query: [Float],
        vectors: [[Float]],
        graph: [[(UInt32, Float)]],
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> [SearchResult] {
        guard !vectors.isEmpty else {
            throw ANNSError.indexEmpty
        }
        guard graph.count == vectors.count else {
            throw ANNSError.searchFailed("Graph size does not match vector count")
        }

        let dim = vectors[0].count
        for vector in vectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }

        return try await searchImpl(
            query: query,
            vectorCount: vectors.count,
            dim: dim,
            graph: graph,
            entryPoint: entryPoint,
            k: k,
            ef: ef,
            metric: metric,
            vectorAt: { vectors[$0] }
        )
    }

    private static func insertSorted(_ candidate: Candidate, into list: inout [Candidate]) {
        var insertionIndex = list.endIndex
        for index in list.indices where candidate.distance < list[index].distance {
            insertionIndex = index
            break
        }
        list.insert(candidate, at: insertionIndex)
    }

    private static func searchImpl(
        query: [Float],
        vectorCount: Int,
        dim: Int,
        graph: [[(UInt32, Float)]],
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric,
        vectorAt: (Int) -> [Float]
    ) async throws -> [SearchResult] {
        guard k > 0 else {
            return []
        }
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }
        guard ef >= k else {
            throw ANNSError.searchFailed("ef must be greater than or equal to k")
        }
        guard query.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: query.count)
        }
        guard entryPoint >= 0, entryPoint < vectorCount else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }

        let efLimit = min(vectorCount, ef)
        let entryID = UInt32(entryPoint)
        let entryDistance = SIMDDistance.distance(query, vectorAt(entryPoint), metric: metric)

        var visited: Set<UInt32> = [entryID]
        var candidates: [Candidate] = [Candidate(nodeID: entryID, distance: entryDistance)]
        var results: [Candidate] = [Candidate(nodeID: entryID, distance: entryDistance)]

        while !candidates.isEmpty {
            let current = candidates.removeFirst()
            if results.count >= efLimit, let worst = results.last, current.distance > worst.distance {
                break
            }

            for (neighborID, _) in graph[Int(current.nodeID)] {
                let neighborIndex = Int(neighborID)
                if neighborID == UInt32.max || neighborIndex < 0 || neighborIndex >= vectorCount {
                    continue
                }
                if visited.contains(neighborID) {
                    continue
                }
                visited.insert(neighborID)

                let candidateDistance = SIMDDistance.distance(query, vectorAt(neighborIndex), metric: metric)

                if results.count < efLimit || candidateDistance < results[results.count - 1].distance {
                    let candidate = Candidate(nodeID: neighborID, distance: candidateDistance)
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
}
