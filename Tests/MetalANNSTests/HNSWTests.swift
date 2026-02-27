import Testing
@testable import MetalANNSCore

@Suite("HNSW Layer Tests")
struct HNSWTests {
    @Test("HNSWLayers stores neighbors")
    func hnswtStorageTest() {
        let layer = SkipLayer(
            nodeToLayerIndex: [0: 0, 1: 1],
            layerIndexToNode: [0, 1],
            adjacency: [[1], [0]]
        )
        let hnsw = HNSWLayers(layers: [layer], maxLayer: 1, entryPoint: 0)

        #expect(layer.nodeToLayerIndex[0] == 0)
        #expect(hnsw.neighbors(of: 0, at: 1) == [1])
    }

    @Test("HNSWBuilder creates layers")
    func hnswtBuildingTest() async throws {
        let vectors = (0..<50).map { i in (0..<8).map { d in Float(i * 8 + d) } }
        let graph = (0..<50).map { i in [(UInt32((i + 1) % 50), Float(0.5))] }

        let hnsw = try HNSWBuilder.buildLayers(
            vectors: try makeVectorBuffer(vectors),
            graph: graph,
            nodeCount: 50,
            metric: .l2
        )

        #expect(hnsw.maxLayer >= 0)
    }

    @Test("HNSWSearchCPU descends layers and searches")
    func hnswtSearchTest() async throws {
        let vectors = (0..<100).map { i in (0..<16).map { d in Float(i * 16 + d) * 0.01 } }
        let (graphData, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 4,
            metric: .l2,
            maxIterations: 5
        )

        let hnsw = try HNSWBuilder.buildLayers(
            vectors: try makeVectorBuffer(vectors),
            graph: graphData,
            nodeCount: vectors.count,
            metric: .l2
        )

        let query = (0..<16).map { d in Float(d) * 0.01 }
        let results = try await HNSWSearchCPU.search(
            query: query,
            vectors: vectors,
            hnsw: hnsw,
            baseGraph: graphData,
            k: 5,
            ef: 32,
            metric: .l2
        )

        #expect(results.count == 5)
    }

    private func makeVectorBuffer(_ vectors: [[Float]]) throws -> VectorBuffer {
        guard let first = vectors.first else {
            throw ANNSError.constructionFailed("Empty vectors")
        }
        let buffer = try VectorBuffer(capacity: vectors.count + 10, dim: first.count)
        for (index, vector) in vectors.enumerated() {
            try buffer.insert(vector: vector, at: index)
        }
        buffer.setCount(vectors.count)
        return buffer
    }
}
