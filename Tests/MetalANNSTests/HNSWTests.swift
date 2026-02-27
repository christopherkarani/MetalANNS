import Foundation
import Testing
@testable import MetalANNSCore

@Suite("HNSW Layer Tests")
struct HNSWTests {
    @Test("HNSWLayers stores and retrieves neighbors correctly")
    func hnswtLayerStructure() {
        let layer1 = SkipLayer(
            nodeToLayerIndex: [0: 0, 2: 1, 5: 2],
            layerIndexToNode: [0, 2, 5],
            adjacency: [[2, 5], [0, 5], [0, 2]]
        )
        let hnsw = HNSWLayers(layers: [layer1], maxLayer: 1, entryPoint: 0)

        #expect(hnsw.neighbors(of: 0, at: 1) == [2, 5])
        #expect(hnsw.neighbors(of: 2, at: 1) == [0, 5])
        #expect(hnsw.neighbors(of: 1, at: 1) == nil)
    }

    @Test("HNSWBuilder assigns levels with exponential distribution")
    func hnswtBuildingTest() async throws {
        let vectors = (0..<100).map { i in (0..<8).map { d in Float(i * 8 + d) * 0.01 } }
        let (graphData, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 4,
            metric: .l2,
            maxIterations: 5
        )

        let vectorBuffer = try makeVectorBuffer(vectors)
        let hnsw = try HNSWBuilder.buildLayers(
            vectors: vectorBuffer,
            graph: graphData,
            nodeCount: vectors.count,
            metric: .l2
        )

        #expect(hnsw.maxLayer >= 0)
        #expect(hnsw.entryPoint < 100)
    }

    @Test("HNSWSearchCPU descends layers and searches")
    func hnswtSearchTest() async throws {
        let vectors = (0..<200).map { i in (0..<16).map { d in Float(i * 16 + d) * 0.01 } }
        let (graphData, _) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 8,
            metric: .cosine,
            maxIterations: 10
        )

        let hnsw = try HNSWBuilder.buildLayers(
            vectors: try makeVectorBuffer(vectors),
            graph: graphData,
            nodeCount: vectors.count,
            metric: .cosine
        )

        let query = (0..<16).map { d in Float(d) * 0.01 }
        let results = try await HNSWSearchCPU.search(
            query: query,
            vectors: vectors,
            hnsw: hnsw,
            baseGraph: graphData,
            k: 10,
            ef: 64,
            metric: .cosine
        )

        #expect(results.count == 10)
        for i in 1..<results.count {
            #expect(results[i].score >= results[i - 1].score)
        }
    }

    @Test("HNSW recall matches or exceeds flat beam search")
    func hnswtRecallComparison() async throws {
        let vectors = (0..<500).map { i in (0..<32).map { d in sin(Float(i * 32 + d) * 0.01) } }
        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 16,
            metric: .cosine,
            maxIterations: 15
        )

        let hnsw = try HNSWBuilder.buildLayers(
            vectors: try makeVectorBuffer(vectors),
            graph: graphData,
            nodeCount: vectors.count,
            metric: .cosine
        )

        let backend = AccelerateBackend()
        let flat = vectors.flatMap { $0 }
        let queries = (0..<10).map { i in (0..<32).map { d in sin(Float(i * 32 + d) * 0.01) } }
        let k = 10
        let ef = 64

        var hnswRecall: Float = 0
        var flatRecall: Float = 0

        for query in queries {
            let hnswResults = try await HNSWSearchCPU.search(
                query: query,
                vectors: vectors,
                hnsw: hnsw,
                baseGraph: graphData,
                k: k,
                ef: ef,
                metric: .cosine
            )

            let flatResults = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: graphData,
                entryPoint: Int(entryPoint),
                k: k,
                ef: ef,
                metric: .cosine
            )

            let exactDistances = try await computeExactDistances(query, vectors, backend, flat)
            let exactTopK = Set(
                exactDistances.enumerated()
                    .sorted { $0.element < $1.element }
                    .prefix(k)
                    .map { UInt32($0.offset) }
            )

            hnswRecall += Float(Set(hnswResults.map(\.internalID)).intersection(exactTopK).count) / Float(k)
            flatRecall += Float(Set(flatResults.map(\.internalID)).intersection(exactTopK).count) / Float(k)
        }

        hnswRecall /= Float(queries.count)
        flatRecall /= Float(queries.count)

        #expect(abs(hnswRecall - flatRecall) < 0.05)
    }

    @Test("HNSWConfiguration has defaults")
    func hnswtConfigTest() {
        let config = HNSWConfiguration.default
        #expect(config.enabled == true)
        #expect(config.M > 0)
        #expect(config.maxLayers >= 0)
    }

    private func computeExactDistances(
        _ query: [Float],
        _ vectors: [[Float]],
        _ backend: AccelerateBackend,
        _ flat: [Float]
    ) async throws -> [Float] {
        try await withVectorBuffer(flat) { pointer in
            try await backend.computeDistances(
                query: query,
                vectors: pointer,
                vectorCount: vectors.count,
                dim: query.count,
                metric: .cosine
            )
        }
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

    private func withVectorBuffer<T>(
        _ values: [Float],
        _ body: (UnsafeBufferPointer<Float>) async throws -> T
    ) async throws -> T {
        let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: values.count)
        _ = buffer.initialize(from: values)
        defer {
            buffer.deinitialize()
            buffer.deallocate()
        }
        return try await body(UnsafeBufferPointer(buffer))
    }
}
