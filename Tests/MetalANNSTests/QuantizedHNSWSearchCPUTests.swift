import Foundation
import Testing
@testable import MetalANNSCore

@Suite("QuantizedHNSWSearchCPU Tests")
struct QuantizedHNSWSearchCPUTests {
    @Test("Search returns results")
    func searchReturnsResults() async throws {
        let fixture = try await makeFixture(vectorCount: 600, dim: 128)
        let query = fixture.vectors[42]

        let results = try await QuantizedHNSWSearchCPU.search(
            query: query,
            vectors: fixture.vectors,
            hnsw: fixture.quantized,
            baseGraph: fixture.graph,
            k: 10,
            ef: 64,
            metric: .cosine
        )

        #expect(results.count == 10)
    }

    @Test("Recall vs exact")
    func recallVsExact() async throws {
        let fixture = try await makeFixture(vectorCount: 600, dim: 128)
        let queries = Array(fixture.vectors.prefix(20))
        let k = 10

        var recallSum: Float = 0
        for query in queries {
            let quantized = try await QuantizedHNSWSearchCPU.search(
                query: query,
                vectors: fixture.vectors,
                hnsw: fixture.quantized,
                baseGraph: fixture.graph,
                k: k,
                ef: 64,
                metric: .cosine
            )

            let exact = try await HNSWSearchCPU.search(
                query: query,
                vectors: fixture.vectors,
                hnsw: fixture.hnsw,
                baseGraph: fixture.graph,
                k: k,
                ef: 64,
                metric: .cosine
            )

            let qIDs = Set(quantized.map(\.internalID))
            let eIDs = Set(exact.map(\.internalID))
            let overlap = qIDs.intersection(eIDs).count
            recallSum += Float(overlap) / Float(k)
        }

        let recall = recallSum / Float(queries.count)
        #expect(recall > 0.80)
    }

    @Test("Falls back for nil PQ")
    func fallsBackForNilPQ() async throws {
        let fixture = try await makeFixture(vectorCount: 500, dim: 128)
        let nilLayers = fixture.hnsw.layers.map { QuantizedSkipLayer(base: $0, pq: nil, codes: []) }
        let q = QuantizedHNSWLayers(
            quantizedLayers: nilLayers,
            maxLayer: fixture.hnsw.maxLayer,
            mL: fixture.hnsw.mL,
            entryPoint: fixture.hnsw.entryPoint
        )

        let results = try await QuantizedHNSWSearchCPU.search(
            query: fixture.vectors[3],
            vectors: fixture.vectors,
            hnsw: q,
            baseGraph: fixture.graph,
            k: 10,
            ef: 64,
            metric: .cosine
        )

        #expect(results.count == 10)
    }

    @Test("Empty query throws")
    func emptyQueryThrows() async {
        do {
            _ = try await QuantizedHNSWSearchCPU.search(
                query: [],
                vectors: [],
                hnsw: QuantizedHNSWLayers(),
                baseGraph: [],
                k: 10,
                ef: 64,
                metric: .cosine
            )
            #expect(Bool(false), "Expected indexEmpty")
        } catch let error as ANNSError {
            if case .indexEmpty = error { }
            else {
                #expect(Bool(false), "Expected indexEmpty, got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.indexEmpty, got \(error)")
        }
    }

    private struct Fixture {
        let vectors: [[Float]]
        let graph: [[(UInt32, Float)]]
        let hnsw: HNSWLayers
        let quantized: QuantizedHNSWLayers
    }

    private func makeFixture(vectorCount: Int, dim: Int) async throws -> Fixture {
        let vectors = makeVectors(count: vectorCount, dim: dim)
        let graphBuild = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 32,
            metric: .cosine,
            maxIterations: 8,
            convergenceThreshold: 0.001
        )
        let vectorBuffer = try VectorBuffer(capacity: vectors.count, dim: dim)
        for (index, vector) in vectors.enumerated() {
            try vectorBuffer.insert(vector: vector, at: index)
        }
        vectorBuffer.setCount(vectors.count)

        let hnsw = try HNSWBuilder.buildLayers(
            vectors: vectorBuffer,
            graph: graphBuild.graph,
            nodeCount: vectors.count,
            metric: .cosine,
            config: HNSWConfiguration(enabled: true, M: 8, maxLayers: 6)
        )
        let quantized = try QuantizedHNSWBuilder.build(
            from: hnsw,
            vectors: vectors,
            config: QuantizedHNSWConfiguration(pqSubspaces: 4),
            metric: .cosine
        )

        return Fixture(
            vectors: vectors,
            graph: graphBuild.graph,
            hnsw: hnsw,
            quantized: quantized
        )
    }

    private func makeVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let x = Float(row * dim + col)
                return sin(x * 0.017) + cos(x * 0.031)
            }
        }
    }
}
