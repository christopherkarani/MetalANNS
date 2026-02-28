import Foundation
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "NNDescentCPU")

public enum NNDescentCPU {
    public static func build(
        vectors: [[Float]],
        degree: Int,
        metric: Metric,
        maxIterations: Int = 20,
        convergenceThreshold: Float = 0.001
    ) async throws -> (graph: [[(UInt32, Float)]], entryPoint: UInt32) {
        let nodeCount = vectors.count
        guard nodeCount > 1 else {
            throw ANNSError.constructionFailed("NNDescentCPU requires at least 2 vectors")
        }
        guard degree > 0, degree < nodeCount else {
            throw ANNSError.constructionFailed("Degree must be in 1..<(nodeCount)")
        }

        let dim = vectors[0].count
        guard dim > 0 else {
            throw ANNSError.constructionFailed("Vector dimension must be greater than 0")
        }
        for vector in vectors {
            if vector.count != dim {
                throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
            }
        }

        func distance(_ lhs: Int, _ rhs: Int) -> Float {
            switch metric {
            case .cosine:
                var dot: Float = 0
                var normL: Float = 0
                var normR: Float = 0
                for d in 0..<dim {
                    let a = vectors[lhs][d]
                    let b = vectors[rhs][d]
                    dot += a * b
                    normL += a * a
                    normR += b * b
                }
                let denom = sqrt(normL) * sqrt(normR)
                return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
            case .l2:
                var sum: Float = 0
                for d in 0..<dim {
                    let diff = vectors[lhs][d] - vectors[rhs][d]
                    sum += diff * diff
                }
                return sum
            case .innerProduct:
                var dot: Float = 0
                for d in 0..<dim {
                    dot += vectors[lhs][d] * vectors[rhs][d]
                }
                return -dot
            case .hamming:
                var mismatches = 0
                for d in 0..<dim where vectors[lhs][d] != vectors[rhs][d] {
                    mismatches += 1
                }
                return Float(mismatches)
            }
        }

        struct LCG {
            var state: UInt64

            mutating func next(upperBound: Int) -> Int {
                state = state &* 6364136223846793005 &+ 1
                return Int(state % UInt64(upperBound))
            }
        }

        typealias Neighbor = (UInt32, Float)
        var graph = [[Neighbor]](repeating: [], count: nodeCount)

        for node in 0..<nodeCount {
            var generator = LCG(state: UInt64(node + 1) &* 0x9E3779B97F4A7C15)
            var neighbors = Set<Int>()
            while neighbors.count < degree {
                let candidate = generator.next(upperBound: nodeCount)
                if candidate != node {
                    neighbors.insert(candidate)
                }
            }

            var list = [Neighbor]()
            list.reserveCapacity(degree)
            for candidate in neighbors {
                list.append((UInt32(candidate), distance(node, candidate)))
            }
            list.sort { $0.1 < $1.1 }
            graph[node] = list
        }

        func tryInsert(_ candidate: Int, into node: Int, dist: Float, graph: inout [[Neighbor]]) -> Bool {
            if candidate == node {
                return false
            }

            var neighbors = graph[node]
            if neighbors.contains(where: { Int($0.0) == candidate }) {
                return false
            }

            var worstIndex = 0
            var worstDistance = neighbors[0].1
            for idx in 1..<neighbors.count {
                if neighbors[idx].1 > worstDistance {
                    worstDistance = neighbors[idx].1
                    worstIndex = idx
                }
            }

            if dist >= worstDistance {
                return false
            }

            neighbors[worstIndex] = (UInt32(candidate), dist)
            neighbors.sort { $0.1 < $1.1 }
            graph[node] = neighbors
            return true
        }

        for iteration in 0..<maxIterations {
            var reverse = [[Int]](repeating: [], count: nodeCount)
            for source in 0..<nodeCount {
                for (targetID, _) in graph[source] {
                    let target = Int(targetID)
                    if target < nodeCount {
                        reverse[target].append(source)
                    }
                }
            }

            var updateCount = 0

            for node in 0..<nodeCount {
                let candidates = Array(Set(graph[node].map { Int($0.0) } + reverse[node]))
                if candidates.count < 2 {
                    continue
                }

                for i in 0..<(candidates.count - 1) {
                    let a = candidates[i]
                    for j in (i + 1)..<candidates.count {
                        let b = candidates[j]
                        if a == b {
                            continue
                        }

                        let dist = distance(a, b)
                        if tryInsert(b, into: a, dist: dist, graph: &graph) {
                            updateCount += 1
                        }
                        if tryInsert(a, into: b, dist: dist, graph: &graph) {
                            updateCount += 1
                        }
                    }
                }
            }

            logger.debug("NNDescentCPU iteration \(iteration): \(updateCount) updates")

            if Float(updateCount) < convergenceThreshold * Float(degree * nodeCount) {
                logger.debug("NNDescentCPU converged at iteration \(iteration + 1)")
                break
            }
        }

        var entryPoint: UInt32 = 0
        var bestMean = Float.greatestFiniteMagnitude

        for node in 0..<nodeCount {
            let mean = graph[node].reduce(Float(0)) { $0 + $1.1 } / Float(degree)
            if mean < bestMean {
                bestMean = mean
                entryPoint = UInt32(node)
            }
        }

        return (graph, entryPoint)
    }
}
