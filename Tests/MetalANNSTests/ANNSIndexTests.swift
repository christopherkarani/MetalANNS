import Foundation
import Testing
@testable import MetalANNS
import MetalANNSCore

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
        let tempDBURL = URL(fileURLWithPath:
            tempURL.deletingPathExtension().appendingPathExtension("db").path)
        defer {
            try? FileManager.default.removeItem(at: tempURL)
            try? FileManager.default.removeItem(at: tempMetaURL)
            try? FileManager.default.removeItem(at: tempDBURL)
        }

        let before = try await index.search(query: baseVectors[12], k: 10)
        try await index.save(to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempDBURL.path))

        let loaded = try await ANNSIndex.load(from: tempURL)
        let after = try await loaded.search(query: baseVectors[12], k: 10)

        #expect(before.map(\.id) == after.map(\.id))
        #expect(before.map(\.internalID) == after.map(\.internalID))
        let originalCount = await index.count
        let loadedCount = await loaded.count
        #expect(originalCount == loadedCount)
    }

    @Test("Save creates SQLite sidecar")
    func saveCreatesDBFile() async throws {
        let index = ANNSIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
            ids.append("node-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
        try await index.save(to: url)

        #expect(FileManager.default.fileExists(atPath: url.path))

        let dbPath = URL(fileURLWithPath: url.deletingPathExtension().path)
            .appendingPathExtension("db").path
        #expect(FileManager.default.fileExists(atPath: dbPath))

        #expect(!FileManager.default.fileExists(atPath: url.path + ".meta.json"))
    }

    @Test("Load roundtrip prefers SQLite when fresh")
    func loadFromSQLiteRoundtrip() async throws {
        let index = ANNSIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
            ids.append("node-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
        try await index.save(to: url)

        let metaJSON = URL(fileURLWithPath: url.path + ".meta.json")
        try? FileManager.default.removeItem(at: metaJSON)

        let loaded = try await ANNSIndex.load(from: url)
        #expect(await loaded.count == 50)

        let query = vectors[0]
        let results = try await loaded.search(query: query, k: 5)
        #expect(results.first?.id == "node-0")
    }

    @Test("Load falls back to JSON when DB missing")
    func loadFallsBackToJSONSidecar() async throws {
        let index = ANNSIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
            ids.append("legacy-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("legacy.anns")
        try await index.save(to: url)

        let dbPath = url.deletingPathExtension().appendingPathExtension("db").path
        try? FileManager.default.removeItem(atPath: dbPath)

        struct LegacyPersistedMetadata: Encodable {
            let configuration: IndexConfiguration
            let softDeletion: SoftDeletion
            let metadataStore: MetadataStore?
        }
        let legacyMeta = LegacyPersistedMetadata(
            configuration: .default,
            softDeletion: SoftDeletion(),
            metadataStore: nil
        )
        let metaJSON = try JSONEncoder().encode(legacyMeta)
        let metaURL = URL(fileURLWithPath: url.path + ".meta.json")
        try metaJSON.write(to: metaURL, options: Data.WritingOptions.atomic)

        let loaded = try await ANNSIndex.load(from: url)
        #expect(await loaded.count == 50)
    }

    @Test("Load falls back when DB is stale or unreadable")
    func loadFallsBackWhenDBInvalid() async throws {
        let index = ANNSIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
            ids.append("corrupt-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("corrupt.anns")
        try await index.save(to: url)

        let dbURL = URL(fileURLWithPath: url.deletingPathExtension().appendingPathExtension("db").path)
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: dbURL, options: .atomic)

        struct LegacyMeta: Encodable {
            let configuration: IndexConfiguration
            let softDeletion: SoftDeletion
            let metadataStore: MetadataStore?
        }
        let fallbackJSON = try JSONEncoder().encode(
            LegacyMeta(
                configuration: .default,
                softDeletion: SoftDeletion(),
                metadataStore: nil
            )
        )
        try fallbackJSON.write(
            to: URL(fileURLWithPath: url.path + ".meta.json"),
            options: Data.WritingOptions.atomic
        )

        let loaded = try await ANNSIndex.load(from: url)
        #expect(await loaded.count == 50)
    }

    @Test("Load falls back to JSON when legacy meta.db is unreadable")
    func loadFallsBackWhenLegacyMetaDBCorrupt() async throws {
        let index = ANNSIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
            ids.append("legacy-corrupt-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("legacy-corrupt.anns")
        try await index.save(to: url)

        let dbPath = url.deletingPathExtension().appendingPathExtension("db").path
        try? FileManager.default.removeItem(atPath: dbPath)

        struct LegacyMeta: Encodable {
            let configuration: IndexConfiguration
            let softDeletion: SoftDeletion
            let metadataStore: MetadataStore?
        }
        let fallbackJSON = try JSONEncoder().encode(
            LegacyMeta(
                configuration: .default,
                softDeletion: SoftDeletion(),
                metadataStore: nil
            )
        )
        try fallbackJSON.write(
            to: URL(fileURLWithPath: url.path + ".meta.json"),
            options: .atomic
        )
        try Data([0x00, 0x01, 0x02, 0x03]).write(
            to: URL(fileURLWithPath: url.path + ".meta.db"),
            options: .atomic
        )

        let loaded = try await ANNSIndex.load(from: url)
        #expect(await loaded.count == 50)
    }

    @Test("Save removes stale legacy metadata sidecars")
    func saveRemovesLegacyMetadataSidecars() async throws {
        let index = ANNSIndex(configuration: .default)
        var vectors: [[Float]] = []
        var ids: [String] = []
        for i in 0..<50 {
            vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
            ids.append("cleanup-\(i)")
        }
        try await index.build(vectors: vectors, ids: ids)

        let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let url = URL(fileURLWithPath: dir).appendingPathComponent("cleanup.anns")
        let metaJSONURL = URL(fileURLWithPath: url.path + ".meta.json")
        let metaDBURL = URL(fileURLWithPath: url.path + ".meta.db")

        try Data("stale".utf8).write(to: metaJSONURL, options: .atomic)
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: metaDBURL, options: .atomic)
        #expect(FileManager.default.fileExists(atPath: metaJSONURL.path))
        #expect(FileManager.default.fileExists(atPath: metaDBURL.path))

        try await index.save(to: url)

        #expect(!FileManager.default.fileExists(atPath: metaJSONURL.path))
        #expect(!FileManager.default.fileExists(atPath: metaDBURL.path))
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
