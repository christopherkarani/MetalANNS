import Foundation
import Testing
@testable import MetalANNS

@Suite("ANNSIndex Public API Tests")
struct ANNSIndexTests {
    @Test("Build and search returns mapped external IDs")
    func buildAndSearch() async throws {
        let dim = 16
        let vectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let ids = (0..<100).map { "vec_\($0)" }

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let results = try await index.search(query: vectors[0], k: 5)

        #expect(results.count == 5)
        #expect(results[0].id == "vec_0")
        #expect(abs(results[0].score) < 1e-4)
        #expect(results.allSatisfy { !$0.id.isEmpty })
    }

    @Test("Insert then search finds inserted vectors")
    func insertAndSearch() async throws {
        let dim = 16
        let baseVectors = makeVectors(count: 50, dim: dim, seedOffset: 0)
        let baseIDs = (0..<50).map { "vec_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)

        let insertedVectors = makeVectors(count: 5, dim: dim, seedOffset: 10_000)
        for i in 0..<5 {
            try await index.insert(insertedVectors[i], id: "new_\(i)")
        }

        for i in 0..<5 {
            let results = try await index.search(query: insertedVectors[i], k: 1)
            #expect(results.count == 1)
            #expect(results[0].id == "new_\(i)")
        }

        #expect(await index.count == 55)
    }

    @Test("Delete excludes soft-deleted IDs from search")
    func deleteAndSearch() async throws {
        let dim = 16
        let vectors = makeVectors(count: 50, dim: dim, seedOffset: 0)
        let ids = (0..<50).map { "vec_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        try await index.delete(id: "vec_0")
        try await index.delete(id: "vec_5")

        let results = try await index.search(query: vectors[1], k: 50)
        let resultIDs = Set(results.map(\.id))
        #expect(!resultIDs.contains("vec_0"))
        #expect(!resultIDs.contains("vec_5"))
        #expect(await index.count == 48)
    }

    @Test("Save and load round-trips lifecycle state")
    func saveAndLoadLifecycle() async throws {
        let dim = 16
        let baseVectors = makeVectors(count: 100, dim: dim, seedOffset: 0)
        let baseIDs = (0..<100).map { "vec_\($0)" }
        let insertedVectors = makeVectors(count: 5, dim: dim, seedOffset: 20_000)

        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: baseVectors, ids: baseIDs)
        for i in 0..<5 {
            try await index.insert(insertedVectors[i], id: "new_\(i)")
        }
        try await index.delete(id: "vec_0")
        try await index.delete(id: "vec_10")

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-public-api-\(UUID().uuidString)")
            .appendingPathExtension("mann")
        let tempMetaURL = URL(fileURLWithPath: tempURL.path + ".meta.json")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
            try? FileManager.default.removeItem(at: tempMetaURL)
        }

        let before = try await index.search(query: baseVectors[12], k: 10)
        try await index.save(to: tempURL)

        let loaded = try await ANNSIndex.load(from: tempURL)
        let after = try await loaded.search(query: baseVectors[12], k: 10)

        #expect(before.map(\.id) == after.map(\.id))
        #expect(before.map(\.internalID) == after.map(\.internalID))
        let originalCount = await index.count
        let loadedCount = await loaded.count
        #expect(originalCount == loadedCount)
    }

    @Test("Batch search returns one result list per query")
    func batchSearchReturnsCorrectShape() async throws {
        let dim = 16
        let vectors = makeVectors(count: 50, dim: dim, seedOffset: 0)
        let ids = (0..<50).map { "vec_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        let queries = [vectors[1], vectors[2], vectors[3]]
        let results = try await index.batchSearch(queries: queries, k: 5)

        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.count == 5 })
    }

    @Test("Build rejects invalid node count and degree combinations")
    func buildValidationRejectsInvalidDegree() async throws {
        let singleVector = [[Float]](repeating: [0, 1, 2, 3], count: 1)
        let singleIDs = ["only"]
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 1, metric: .cosine))
        do {
            try await index.build(vectors: singleVector, ids: singleIDs)
            #expect(Bool(false), "Expected construction failure for single-vector build")
        } catch let error as ANNSError {
            guard case .constructionFailed = error else {
                #expect(Bool(false), "Expected constructionFailed, got \(error)")
                return
            }
        }

        let vectors = makeVectors(count: 5, dim: 4, seedOffset: 1000)
        let ids = (0..<5).map { "bad-\($0)" }
        let invalidDegreeIndex = ANNSIndex(configuration: IndexConfiguration(degree: 5, metric: .cosine))
        do {
            try await invalidDegreeIndex.build(vectors: vectors, ids: ids)
            #expect(Bool(false), "Expected construction failure for degree >= node count")
        } catch let error as ANNSError {
            guard case .constructionFailed = error else {
                #expect(Bool(false), "Expected constructionFailed, got \(error)")
                return
            }
        }
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
