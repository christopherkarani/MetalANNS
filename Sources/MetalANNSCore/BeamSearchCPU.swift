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
        metric: Metric
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
        let entryDistance = distance(query: query, vector: vectors[entryPoint], metric: metric)

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
                if neighborID == UInt32.max || neighborIndex < 0 || neighborIndex >= vectors.count {
                    continue
                }
                if visited.contains(neighborID) {
                    continue
                }
                visited.insert(neighborID)

                let candidateDistance = distance(
                    query: query,
                    vector: vectors[neighborIndex],
                    metric: metric
                )

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
