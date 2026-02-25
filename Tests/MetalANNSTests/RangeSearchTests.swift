import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Range Search Tests")
struct RangeSearchTests {
    @Test("Range search returns only results within threshold")
    func rangeSearchReturnsWithinThreshold() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let ids = (0..<100).map { "v\($0)" }

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.rangeSearch(
            query: vectors[0],
            maxDistance: 0.5,
            limit: 100
        )

        #expect(!results.isEmpty)
        #expect(results.allSatisfy { $0.score <= 0.5 + 1e-6 })
    }

    @Test("Range search finds exact match with tight threshold")
    func rangeSearchFindsExactMatch() async throws {
        let dim = 16
        let vectors = makeVectors(count: 50, dim: dim, seedOffset: 500)
        let ids = (0..<50).map { "v\($0)" }
        let targetID = "v17"

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.rangeSearch(
            query: vectors[17],
            maxDistance: 0.01,
            limit: 50
        )

        #expect(results.contains { $0.id == targetID })
    }

    @Test("Range search accepts zero threshold for exact L2 matches")
    func rangeSearchAllowsZeroThreshold() async throws {
        let vectors: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]
        let ids = ["a", "b", "c"]

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 2, metric: .l2))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.rangeSearch(
            query: vectors[0],
            maxDistance: 0,
            limit: 10
        )

        #expect(results.contains { $0.id == "a" })
        #expect(results.allSatisfy { $0.score <= 0 + 1e-6 })
    }

    @Test("Range search accepts negative thresholds for inner product")
    func rangeSearchAllowsNegativeThresholdInnerProduct() async throws {
        let target = Array(repeating: Float(1), count: 8)
        let vectors: [[Float]] = [
            target,
            Array(repeating: Float(0), count: 8),
            [1, -1, 1, -1, 1, -1, 1, -1]
        ]
        let ids = ["target", "zero", "mixed"]

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 2, metric: .innerProduct))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.rangeSearch(
            query: target,
            maxDistance: -1.0,
            limit: 10
        )

        #expect(results.contains { $0.id == "target" })
    }

    @Test("Range search over-fetches when filter is selective")
    func rangeSearchOverFetchesWithFilter() async throws {
        var vectors: [[Float]] = []
        var ids: [String] = []

        for i in 0..<12 {
            vectors.append([Float(i) * 0.00001, 0, 0, 0])
            ids.append("drop_\(i)")
        }
        for i in 0..<12 {
            vectors.append([0.010 + Float(i) * 0.00001, 0, 0, 0])
            ids.append("keep_\(i)")
        }
        for i in 0..<6 {
            vectors.append([2.0 + Float(i) * 0.01, 0, 0, 0])
            ids.append("far_\(i)")
        }

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 16, metric: .l2))
        try await index.build(vectors: vectors, ids: ids)

        for id in ids {
            if id.hasPrefix("keep_") {
                try await index.setMetadata("bucket", value: "keep", for: id)
            } else {
                try await index.setMetadata("bucket", value: "drop", for: id)
            }
        }

        let results = try await index.rangeSearch(
            query: [0, 0, 0, 0],
            maxDistance: 0.2,
            limit: 10,
            filter: .equals(column: "bucket", value: "keep")
        )

        #expect(results.count == 10)
        #expect(results.allSatisfy { $0.id.hasPrefix("keep_") })
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
