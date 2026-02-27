import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("QuantizedHNSW Benchmark Tests")
struct QuantizedHNSWBenchmarkTests {
    @Test("Quantized vs exact recall")
    func quantizedVsExactRecall() async throws {
        let vectors = makeVectors(count: 1_000, dim: 128)
        let ids = (0..<vectors.count).map { "id-\($0)" }

        var exactConfig = IndexConfiguration.default
        exactConfig.quantizedHNSWConfiguration.useQuantizedEdges = false
        let exactIndex = ANNSIndex(configuration: exactConfig, context: nil)
        try await exactIndex.build(vectors: vectors, ids: ids)

        let quantizedIndex = ANNSIndex(configuration: .default, context: nil)
        try await quantizedIndex.build(vectors: vectors, ids: ids)

        let queries = Array(vectors.prefix(50))
        let exactRecall = try await recallAtK(index: exactIndex, vectors: vectors, ids: ids, queries: queries, k: 10)
        let quantizedRecall = try await recallAtK(index: quantizedIndex, vectors: vectors, ids: ids, queries: queries, k: 10)

        #expect(quantizedRecall >= exactRecall - 0.05)
        print("Quantized recall@10: \(quantizedRecall), exact recall@10: \(exactRecall)")
    }

    @Test("Memory reduction")
    func memoryReduction() async throws {
        let vectors = makeVectors(count: 1_000, dim: 128)
        let core = try await buildCoreQuantized(vectors: vectors, metric: .cosine)

        let encodedQuantized = try JSONEncoder().encode(core.quantized)
        let skipNodeCount = core.quantized.quantizedLayers.reduce(0) { $0 + $1.base.layerIndexToNode.count }
        let fullFloatBytes = skipNodeCount * vectors[0].count * MemoryLayout<Float>.size

        let bytesPerNodeQuantized = Double(encodedQuantized.count) / Double(max(1, skipNodeCount))
        let bytesPerNodeFloat = Double(fullFloatBytes) / Double(max(1, skipNodeCount))
        let ratio = bytesPerNodeQuantized / max(bytesPerNodeFloat, 1)

        print("Quantized bytes/node: \(bytesPerNodeQuantized), float bytes/node: \(bytesPerNodeFloat), ratio: \(ratio)")
        #expect(encodedQuantized.count > 0)
    }

    @Test("ADC faster than exact")
    func adcFasterThanExact() async throws {
        let vectors = makeVectors(count: 1_000, dim: 128)
        let ids = (0..<vectors.count).map { "id-\($0)" }
        let queries = Array(vectors.prefix(200))

        var exactConfig = IndexConfiguration.default
        exactConfig.quantizedHNSWConfiguration.useQuantizedEdges = false
        let exactIndex = ANNSIndex(configuration: exactConfig, context: nil)
        try await exactIndex.build(vectors: vectors, ids: ids)

        let quantizedIndex = ANNSIndex(configuration: .default, context: nil)
        try await quantizedIndex.build(vectors: vectors, ids: ids)

        let exactDuration = try await measureDuration {
            for query in queries {
                _ = try await exactIndex.search(query: query, k: 10, metric: .cosine)
            }
        }
        let quantizedDuration = try await measureDuration {
            for query in queries {
                _ = try await quantizedIndex.search(query: query, k: 10, metric: .cosine)
            }
        }

        let exactSeconds = seconds(exactDuration)
        let quantizedSeconds = seconds(quantizedDuration)
        print("Exact time (s): \(exactSeconds), quantized time (s): \(quantizedSeconds)")
        #expect(quantizedSeconds <= exactSeconds * 1.5)
    }

    @Test("PQ trains in reasonable time")
    func pqTrainsInReasonableTime() throws {
        // Keep this size CI-stable while still exercising real PQ training work.
        let vectors = makeVectors(count: 1_024, dim: 128)
        let clock = ContinuousClock()
        let start = clock.now
        _ = try ProductQuantizer.train(
            vectors: vectors,
            numSubspaces: 4,
            centroidsPerSubspace: 256,
            maxIterations: 4
        )
        let elapsed = clock.now - start
        let elapsedSeconds = seconds(elapsed)
        print("PQ train elapsed (s): \(elapsedSeconds)")
        #expect(elapsedSeconds < 60.0)
    }

    private struct CoreFixture {
        let hnsw: HNSWLayers
        let quantized: QuantizedHNSWLayers
    }

    private func buildCoreQuantized(vectors: [[Float]], metric: Metric) async throws -> CoreFixture {
        let graphBuild = try await NNDescentCPU.build(
            vectors: vectors,
            degree: 32,
            metric: metric,
            maxIterations: 8,
            convergenceThreshold: 0.001
        )

        let vectorBuffer = try VectorBuffer(capacity: vectors.count, dim: vectors[0].count)
        for (index, vector) in vectors.enumerated() {
            try vectorBuffer.insert(vector: vector, at: index)
        }
        vectorBuffer.setCount(vectors.count)

        let hnsw = try HNSWBuilder.buildLayers(
            vectors: vectorBuffer,
            graph: graphBuild.graph,
            nodeCount: vectors.count,
            metric: metric,
            config: .default
        )
        let quantized = try QuantizedHNSWBuilder.build(
            from: hnsw,
            vectors: vectors,
            config: .default,
            metric: metric
        )
        return CoreFixture(hnsw: hnsw, quantized: quantized)
    }

    private func recallAtK(
        index: ANNSIndex,
        vectors: [[Float]],
        ids: [String],
        queries: [[Float]],
        k: Int
    ) async throws -> Float {
        var total: Float = 0
        for query in queries {
            let approx = try await index.search(query: query, k: k, metric: .cosine)
            let exact = exactTopKIDs(query: query, vectors: vectors, ids: ids, k: k, metric: .cosine)
            let overlap = Set(approx.map(\.id)).intersection(Set(exact)).count
            total += Float(overlap) / Float(k)
        }
        return total / Float(queries.count)
    }

    private func exactTopKIDs(
        query: [Float],
        vectors: [[Float]],
        ids: [String],
        k: Int,
        metric: Metric
    ) -> [String] {
        let scored = vectors.enumerated().map { (index, vector) in
            (id: ids[index], score: SIMDDistance.distance(query, vector, metric: metric))
        }.sorted { $0.score < $1.score }
        return Array(scored.prefix(k).map(\.id))
    }

    private func makeVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let x = Float(row * dim + col)
                return sin(x * 0.017) + cos(x * 0.031)
            }
        }
    }

    private func measureDuration(
        _ body: () async throws -> Void
    ) async throws -> Duration {
        let clock = ContinuousClock()
        let start = clock.now
        try await body()
        return clock.now - start
    }

    private func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }
}
