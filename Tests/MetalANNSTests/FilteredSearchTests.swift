import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Filtered Search Tests")
struct FilteredSearchTests {
    @Test("Filtered search returns only matching results")
    func filteredSearchReturnsOnlyMatching() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let ids = (0..<100).map { "v\($0)" }

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<100 {
            let category = i < 50 ? "A" : "B"
            try await index.setMetadata("category", value: category, for: "v\(i)")
        }

        let results = try await index.search(
            query: vectors[10],
            k: 20,
            filter: .equals(column: "category", value: "A")
        )

        #expect(!results.isEmpty)
        let validIDs = Set((0..<50).map { "v\($0)" })
        #expect(results.allSatisfy { validIDs.contains($0.id) })
    }

    @Test("Filtered search maintains reasonable recall")
    func filteredSearchRecall() async throws {
        let dim = 32
        let vectors = makeVectors(count: 200, dim: dim, seedOffset: 100)
        let ids = (0..<200).map { "v\($0)" }

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<200 {
            let category = i.isMultiple(of: 2) ? "even" : "odd"
            try await index.setMetadata("category", value: category, for: "v\(i)")
        }

        var hits = 0
        var queryCount = 0
        for i in stride(from: 0, to: 80, by: 2) {
            let results = try await index.search(
                query: vectors[i],
                k: 5,
                filter: .equals(column: "category", value: "even")
            )
            if results.contains(where: { $0.id == "v\(i)" }) {
                hits += 1
            }
            queryCount += 1
        }

        let recall = Float(hits) / Float(max(1, queryCount))
        #expect(recall >= 0.50)
    }

    @Test("Compound filters work correctly")
    func compoundFilterWorks() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 500)
        let ids = (0..<100).map { "v\($0)" }

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<100 {
            let category = i < 50 ? "A" : "B"
            let score: Float = (25..<50).contains(i) || (75..<100).contains(i) ? 5.0 : 1.0
            try await index.setMetadata("category", value: category, for: "v\(i)")
            try await index.setMetadata("score", value: score, for: "v\(i)")
        }

        let andFilter: SearchFilter = .and([
            .equals(column: "category", value: "A"),
            .greaterThan(column: "score", value: 3.0)
        ])
        let andResults = try await index.search(query: vectors[25], k: 100, filter: andFilter)
        let andAllowed = Set((25..<50).map { "v\($0)" })
        #expect(!andResults.isEmpty)
        #expect(andResults.allSatisfy { andAllowed.contains($0.id) })

        let orFilter: SearchFilter = .or([
            .equals(column: "category", value: "A"),
            .greaterThan(column: "score", value: 3.0)
        ])
        let orResults = try await index.search(query: vectors[25], k: 100, filter: orFilter)
        let orAllowed = Set((0..<50).map { "v\($0)" } + (75..<100).map { "v\($0)" })
        #expect(!orResults.isEmpty)
        #expect(orResults.allSatisfy { orAllowed.contains($0.id) })

        let notFilter: SearchFilter = .not(.equals(column: "category", value: "A"))
        let notResults = try await index.search(query: vectors[75], k: 100, filter: notFilter)
        let notAllowed = Set((50..<100).map { "v\($0)" })
        #expect(!notResults.isEmpty)
        #expect(notResults.allSatisfy { notAllowed.contains($0.id) })
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
