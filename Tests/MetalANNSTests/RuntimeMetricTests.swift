import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Runtime Metric Selection Tests")
struct RuntimeMetricTests {
    @Test("Search with different metric returns valid results")
    func searchWithDifferentMetric() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 200)
        let ids = (0..<100).map { "v\($0)" }
        let query = vectors[3]

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.search(query: query, k: 10, metric: .l2)

        #expect(!results.isEmpty)
        #expect(results.allSatisfy { $0.score >= 0 })

        if results.count > 1 {
            for i in 1..<results.count {
                #expect(results[i - 1].score <= results[i].score + 1e-6)
            }
        }

        for result in results {
            guard let indexInCorpus = parseVectorIndex(from: result.id) else {
                #expect(false)
                continue
            }
            let expectedDistance = SIMDDistance.l2(query, vectors[indexInCorpus])
            #expect(abs(expectedDistance - result.score) < 1e-3)
        }
    }

    @Test("Default metric matches build metric")
    func defaultMetricMatchesBuildMetric() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let ids = (0..<100).map { "v\($0)" }
        let query = vectors[12]

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let defaultMetricResults = try await index.search(query: query, k: 10)
        let explicitMetricResults = try await index.search(query: query, k: 10, metric: .cosine)

        #expect(defaultMetricResults.map(\.id) == explicitMetricResults.map(\.id))
        #expect(defaultMetricResults.map(\.internalID) == explicitMetricResults.map(\.internalID))
        #expect(defaultMetricResults.count == explicitMetricResults.count)
        for i in 0..<defaultMetricResults.count {
            #expect(abs(defaultMetricResults[i].score - explicitMetricResults[i].score) < 1e-6)
        }
    }

    private func parseVectorIndex(from id: String) -> Int? {
        guard id.hasPrefix("v") else {
            return nil
        }
        return Int(id.dropFirst())
    }

    private func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }
}
