import Foundation
import Testing
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("In-Graph Filtering Tests")
struct InGraphFilteringTests {
    @Test("Predicate gates results but not traversal")
    func predicateGatesResultsNotTraversal() async throws {
        let count = 20
        let dim = 8
        let vectors = makeVectors(count: count, dim: dim)
        let graph = makeLinearGraph(count: count)
        let query = vectors[0]
        let k = 12

        let results = try await BeamSearchCPU.search(
            query: query,
            vectors: vectors,
            graph: graph,
            entryPoint: 1,
            k: k,
            ef: count,
            metric: .l2,
            predicate: { $0.isMultiple(of: 2) }
        )

        #expect(results.allSatisfy { $0.internalID.isMultiple(of: 2) })
        let evenCount = (0..<count).filter { $0.isMultiple(of: 2) }.count
        #expect(results.count == min(k, evenCount))
    }

    @Test("Nil predicate remains backward compatible")
    func nilPredicateIsBackwardCompatible() async throws {
        let count = 30
        let dim = 8
        let vectors = makeVectors(count: count, dim: dim)
        let graph = makeLinearGraph(count: count)
        let query = vectors[7]

        let implicit = try await BeamSearchCPU.search(
            query: query,
            vectors: vectors,
            graph: graph,
            entryPoint: 3,
            k: 10,
            ef: 20,
            metric: .l2
        )

        let explicitNil = try await BeamSearchCPU.search(
            query: query,
            vectors: vectors,
            graph: graph,
            entryPoint: 3,
            k: 10,
            ef: 20,
            metric: .l2,
            predicate: nil
        )

        #expect(implicit.count == explicitNil.count)
        for (lhs, rhs) in zip(implicit, explicitNil) {
            #expect(lhs.internalID == rhs.internalID)
            #expect(abs(lhs.score - rhs.score) < 1e-6)
        }
    }

    @Test("Filtered search returns only category A")
    func filteredSearchCorrectness() async throws {
        let dim = 32
        let count = 200
        let vectors = makeVectors(count: count, dim: dim)
        let ids = (0..<count).map { "v\($0)" }

        let index = ANNSIndex(
            configuration: IndexConfiguration(degree: 8, metric: .l2, efSearch: 96),
            context: nil
        )
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<count {
            try await index.setMetadata("category", value: i < 100 ? "A" : "B", for: "v\(i)")
        }

        let results = try await index.search(
            query: vectors[21],
            k: 10,
            filter: .equals(column: "category", value: "A")
        )

        #expect(results.count == 10)
        #expect(results.allSatisfy { extractIndex(from: $0.id) < 100 })
    }

    @Test("Filtered search recall@10 is >= 0.70")
    func filteredSearchRecall() async throws {
        let dim = 32
        let count = 500
        let vectors = makeVectors(count: count, dim: dim)
        let ids = (0..<count).map { "v\($0)" }

        let index = ANNSIndex(
            configuration: IndexConfiguration(degree: 12, metric: .l2, efSearch: 128),
            context: nil
        )
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<count {
            let category = i.isMultiple(of: 2) ? "even" : "odd"
            try await index.setMetadata("category", value: category, for: "v\(i)")
        }

        let queryIndices = stride(from: 0, to: 40, by: 2).map { $0 }
        var recallSum = 0.0

        for idx in queryIndices {
            let query = vectors[idx]
            let results = try await index.search(
                query: query,
                k: 10,
                filter: .equals(column: "category", value: "even")
            )
            let truth = bruteForceFilteredTopK(
                query: query,
                vectors: vectors,
                ids: ids,
                predicate: { extractIndex(from: $0).isMultiple(of: 2) },
                k: 10
            )
            recallSum += recall(results: results, groundTruth: truth)
        }

        let averageRecall = recallSum / Double(queryIndices.count)
        #expect(averageRecall >= 0.70)
    }

    @Test("Soft-deleted IDs never appear")
    func softDeletedExcludedFromResults() async throws {
        let dim = 16
        let count = 100
        let vectors = makeVectors(count: count, dim: dim)
        let ids = (0..<count).map { "v\($0)" }

        let index = ANNSIndex(
            configuration: IndexConfiguration(degree: 8, metric: .l2, efSearch: 96),
            context: nil
        )
        try await index.build(vectors: vectors, ids: ids)

        try await index.delete(id: "v0")
        try await index.delete(id: "v1")
        try await index.delete(id: "v2")

        for i in 0..<10 {
            let results = try await index.search(query: vectors[i], k: 20)
            #expect(!results.contains { $0.id == "v0" })
            #expect(!results.contains { $0.id == "v1" })
            #expect(!results.contains { $0.id == "v2" })
        }
    }

    @Test("Filtered search returns precisely k")
    func filteredSearchReturnsPreciselyK() async throws {
        let dim = 32
        let count = 200
        let vectors = makeVectors(count: count, dim: dim)
        let ids = (0..<count).map { "v\($0)" }

        let index = ANNSIndex(
            configuration: IndexConfiguration(degree: 8, metric: .l2, efSearch: 96),
            context: nil
        )
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<count {
            try await index.setMetadata("category", value: i < 100 ? "A" : "B", for: "v\(i)")
        }

        let results = try await index.search(
            query: vectors[12],
            k: 5,
            filter: .equals(column: "category", value: "A")
        )

        #expect(results.count == 5)
        #expect(results.allSatisfy { extractIndex(from: $0.id) < 100 })
    }

    @Test("Range search with filter returns only matching IDs")
    func rangeSearchFilterCorrectness() async throws {
        let dim = 32
        let count = 180
        let vectors = makeVectors(count: count, dim: dim)
        let ids = (0..<count).map { "v\($0)" }

        let index = ANNSIndex(
            configuration: IndexConfiguration(degree: 8, metric: .l2, efSearch: 96),
            context: nil
        )
        try await index.build(vectors: vectors, ids: ids)

        for i in 0..<count {
            try await index.setMetadata("category", value: i < 90 ? "A" : "B", for: "v\(i)")
        }

        let results = try await index.rangeSearch(
            query: vectors[11],
            maxDistance: 10_000,
            limit: 20,
            filter: .equals(column: "category", value: "A")
        )

        #expect(results.count == 20)
        #expect(results.allSatisfy { extractIndex(from: $0.id) < 90 })
    }

    @Test("Range search excludes soft-deleted IDs")
    func rangeSearchExcludesSoftDeleted() async throws {
        let dim = 16
        let count = 100
        let vectors = makeVectors(count: count, dim: dim)
        let ids = (0..<count).map { "v\($0)" }

        let index = ANNSIndex(
            configuration: IndexConfiguration(degree: 8, metric: .l2, efSearch: 96),
            context: nil
        )
        try await index.build(vectors: vectors, ids: ids)

        try await index.delete(id: "v0")
        try await index.delete(id: "v1")
        try await index.delete(id: "v2")

        let results = try await index.rangeSearch(
            query: vectors[4],
            maxDistance: 10_000,
            limit: 80
        )

        #expect(!results.contains { $0.id == "v0" })
        #expect(!results.contains { $0.id == "v1" })
        #expect(!results.contains { $0.id == "v2" })
    }

    private func makeVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float(row * dim + col)
                return sin(i * 0.091) + cos(i * 0.047)
            }
        }
    }

    private func makeLinearGraph(count: Int) -> [[(UInt32, Float)]] {
        (0..<count).map { i in
            var neighbors: [(UInt32, Float)] = []
            if i > 0 {
                neighbors.append((UInt32(i - 1), 1.0))
            }
            if i < count - 1 {
                neighbors.append((UInt32(i + 1), 1.0))
            }
            return neighbors
        }
    }

    private func bruteForceFilteredTopK(
        query: [Float],
        vectors: [[Float]],
        ids: [String],
        predicate: (String) -> Bool,
        k: Int
    ) -> Set<String> {
        Set(
            zip(ids, vectors)
                .filter { predicate($0.0) }
                .map { ($0.0, SIMDDistance.distance(query, $0.1, metric: .l2)) }
                .sorted { $0.1 < $1.1 }
                .prefix(k)
                .map { $0.0 }
        )
    }

    private func recall(results: [SearchResult], groundTruth: Set<String>) -> Double {
        guard !groundTruth.isEmpty else {
            return 0
        }
        let hitCount = results.filter { groundTruth.contains($0.id) }.count
        return Double(hitCount) / Double(groundTruth.count)
    }

    private func extractIndex(from id: String) -> Int {
        Int(id.dropFirst()) ?? -1
    }
}
