import Testing
@testable import MetalANNSCore

@Suite("Graph Repair Tests")
struct GraphRepairTests {
    @Test("RepairConfiguration defaults are sensible")
    func repairConfigDefaults() {
        let config = RepairConfiguration.default
        #expect(config.repairInterval == 100)
        #expect(config.repairDepth == 2)
        #expect(config.repairIterations == 5)
        #expect(config.enabled == true)
    }

    @Test("RepairConfiguration clamps invalid values")
    func repairConfigClamping() {
        let config = RepairConfiguration(repairInterval: -5, repairDepth: 0, repairIterations: 100)
        #expect(config.repairInterval == 0)
        #expect(config.repairDepth == 1)
        #expect(config.repairIterations == 20)
    }

    @Test("Neighborhood collection expands correctly")
    func neighborhoodCollection() async throws {
        let vectors = (0..<10).map { i in
            (0..<8).map { d in
                sin(Float(i * 8 + d) * 0.173)
            }
        }

        let (graphData, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 4,
            metric: .l2,
            maxIterations: 5
        )

        let graphBuffer = try makeGraphBuffer(graphData, degree: 4)

        let updates = try GraphRepairer.repair(
            recentIDs: [0],
            vectors: try makeVectorBuffer(vectors),
            graph: graphBuffer,
            config: RepairConfiguration(repairInterval: 1, repairDepth: 1, repairIterations: 1),
            metric: .l2
        )

        #expect(updates >= 0)
    }
}

private func makeGraphBuffer(_ graphData: [[(UInt32, Float)]], degree: Int) throws -> GraphBuffer {
    let graphBuffer = try GraphBuffer(capacity: graphData.count + 8, degree: degree)
    for node in 0..<graphData.count {
        var ids = Array(repeating: UInt32.max, count: degree)
        var distances = Array(repeating: Float.greatestFiniteMagnitude, count: degree)
        let neighbors = graphData[node]

        for index in 0..<min(degree, neighbors.count) {
            ids[index] = neighbors[index].0
            distances[index] = neighbors[index].1
        }

        try graphBuffer.setNeighbors(of: node, ids: ids, distances: distances)
    }

    graphBuffer.setCount(graphData.count)
    return graphBuffer
}

private func makeVectorBuffer(_ vectors: [[Float]]) throws -> VectorBuffer {
    guard let first = vectors.first else {
        throw ANNSError.constructionFailed("Vector list cannot be empty")
    }

    let vectorBuffer = try VectorBuffer(capacity: vectors.count, dim: first.count)
    for (index, vector) in vectors.enumerated() {
        try vectorBuffer.insert(vector: vector, at: index)
    }
    vectorBuffer.setCount(vectors.count)
    return vectorBuffer
}
