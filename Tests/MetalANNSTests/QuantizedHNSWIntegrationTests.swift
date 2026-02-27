import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("QuantizedHNSW Integration Tests")
struct QuantizedHNSWIntegrationTests {
    @Test("Default config uses quantized")
    func defaultConfigUsesQuantized() async {
        let index = ANNSIndex()
        let config = await index.configurationForTesting()
        #expect(config.quantizedHNSWConfiguration.useQuantizedEdges)
    }

    @Test("Search with quantized enabled")
    func searchWithQuantizedEnabled() async throws {
        let vectors = makeVectors(count: 500, dim: 128)
        let ids = (0..<vectors.count).map { "id-\($0)" }
        let index = ANNSIndex(configuration: .default, context: nil)
        try await index.build(vectors: vectors, ids: ids)

        for query in vectors.prefix(10) {
            let results = try await index.search(query: query, k: 10)
            #expect(results.count == 10)
        }
    }

    @Test("Search with quantized disabled")
    func searchWithQuantizedDisabled() async throws {
        let vectors = makeVectors(count: 300, dim: 128)
        let ids = (0..<vectors.count).map { "id-\($0)" }

        var enabledConfig = IndexConfiguration.default
        enabledConfig.quantizedHNSWConfiguration.useQuantizedEdges = true

        var disabledConfig = IndexConfiguration.default
        disabledConfig.quantizedHNSWConfiguration.useQuantizedEdges = false

        let enabled = ANNSIndex(configuration: enabledConfig, context: nil)
        let disabled = ANNSIndex(configuration: disabledConfig, context: nil)
        try await enabled.build(vectors: vectors, ids: ids)
        try await disabled.build(vectors: vectors, ids: ids)

        for query in vectors.prefix(5) {
            let enabledResults = try await enabled.search(query: query, k: 10)
            let disabledResults = try await disabled.search(query: query, k: 10)
            #expect(enabledResults.map(\.id) == disabledResults.map(\.id))
        }
    }

    @Test("Quantized recall")
    func quantizedRecall() async throws {
        let vectors = makeVectors(count: 500, dim: 128)
        let ids = (0..<vectors.count).map { "id-\($0)" }
        let index = ANNSIndex(configuration: .default, context: nil)
        try await index.build(vectors: vectors, ids: ids)

        let k = 10
        var recallSum: Float = 0
        let queries = Array(vectors.prefix(20))
        for query in queries {
            let results = try await index.search(query: query, k: k, metric: .cosine)
            let exact = exactTopKIDs(query: query, vectors: vectors, ids: ids, k: k, metric: .cosine)
            let overlap = Set(results.map(\.id)).intersection(Set(exact)).count
            recallSum += Float(overlap) / Float(k)
        }

        let recall = recallSum / Float(queries.count)
        #expect(recall > 0.80)
    }

    private func makeVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let x = Float(row * dim + col)
                return sin(x * 0.017) + cos(x * 0.031)
            }
        }
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
}
