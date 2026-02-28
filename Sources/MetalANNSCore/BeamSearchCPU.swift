import Foundation

public enum BeamSearchCPU {
    private struct Candidate {
        let nodeID: UInt32
        let distance: Float
    }

    public static func search(
        query: [Float],
        vectors: [[Float]],
        graph: [[(UInt32, Float)]],
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric,
        predicate: (@Sendable (UInt32) -> Bool)? = nil
    ) async throws -> [SearchResult] {
        guard !vectors.isEmpty else {
            throw ANNSError.indexEmpty
        }
        guard k > 0 else {
            return []
        }
        guard ef >= k else {
            throw ANNSError.searchFailed("ef must be greater than or equal to k")
        }
        guard entryPoint >= 0, entryPoint < vectors.count else {
            throw ANNSError.searchFailed("Entry point is out of bounds")
        }
        guard graph.count == vectors.count else {
            throw ANNSError.searchFailed("Graph size does not match vector count")
        }

        let dim = vectors[0].count
        guard dim > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }
        guard query.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: query.count)
        }
        for vector in vectors where vector.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }

        let efLimit = min(vectors.count, ef)
        let entryID = UInt32(entryPoint)
        let entryDistance = SIMDDistance.distance(query, vectors[entryPoint], metric: metric)
        let entryCandidate = Candidate(nodeID: entryID, distance: entryDistance)

        var visited: Set<UInt32> = [entryID]
        var candidates: [Candidate] = [entryCandidate]
        var results: [Candidate] = []
        if predicate?(entryID) ?? true {
            results = [entryCandidate]
        }

        while !candidates.isEmpty {
            let current = candidates.removeFirst()
            if results.count >= efLimit, let worst = results.last, current.distance > worst.distance {
                break
            }

            for (neighborID, _) in graph[Int(current.nodeID)] {
                let neighborIndex = Int(neighborID)
                if neighborID == UInt32.max || neighborIndex < 0 || neighborIndex >= vectors.count {
                    continue
                }
                if visited.contains(neighborID) {
                    continue
                }
                visited.insert(neighborID)

                let candidateDistance = SIMDDistance.distance(
                    query,
                    vectors[neighborIndex],
                    metric: metric
                )

                if results.count < efLimit || candidateDistance < results[results.count - 1].distance {
                    let candidate = Candidate(nodeID: neighborID, distance: candidateDistance)
                    insertSorted(candidate, into: &candidates)
                    if predicate?(neighborID) ?? true {
                        insertSorted(candidate, into: &results)
                        if results.count > efLimit {
                            results.removeLast()
                        }
                    }
                }
            }
        }

        let topK = min(k, results.count)
        return results.prefix(topK).map { result in
            SearchResult(id: "", score: result.distance, internalID: result.nodeID)
        }
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
