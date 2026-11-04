import Foundation

/// Builds the exact k-nearest-neighbour graph by brute force. Used when the
/// node count does not exceed the graph degree, where NN-Descent cannot run
/// (every node would need more distinct neighbours than exist). Each list holds
/// all other nodes sorted by distance and is padded with `UInt32.max` /
/// `Float.greatestFiniteMagnitude`, the sentinel every traversal skips and every
/// incremental insert replaces first.
package enum ExactGraphBuilder {
    package static func build(
        vectors: [[Float]],
        into graph: GraphBuffer,
        metric: Metric
    ) throws {
        let nodeCount = vectors.count
        let degree = graph.degree
        for node in 0..<nodeCount {
            var neighbors: [(id: UInt32, distance: Float)] = []
            neighbors.reserveCapacity(nodeCount - 1)
            for other in 0..<nodeCount where other != node {
                let distance = SIMDDistance.distance(vectors[node], vectors[other], metric: metric)
                neighbors.append((UInt32(other), distance))
            }
            neighbors.sort { $0.distance < $1.distance }

            var ids = Array(repeating: UInt32.max, count: degree)
            var distances = Array(repeating: Float.greatestFiniteMagnitude, count: degree)
            for slot in 0..<min(degree, neighbors.count) {
                ids[slot] = neighbors[slot].id
                distances[slot] = neighbors[slot].distance
            }
            try graph.setNeighbors(of: node, ids: ids, distances: distances)
        }
        graph.setCount(nodeCount)
    }
}
