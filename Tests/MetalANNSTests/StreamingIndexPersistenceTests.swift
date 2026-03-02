import Foundation
import Testing
@testable import MetalANNS

@Suite("StreamingIndex Persistence Tests")
struct StreamingIndexPersistenceTests {
    @Test("Save and load roundtrip count")
    func saveAndLoadEmpty() async throws {
        let index = StreamingIndex(config: StreamingConfiguration(
            deltaCapacity: 20,
            mergeStrategy: .blocking
        ))

        let vectors = (0..<5).map { makeVector(row: $0, dim: 16) }
        let ids = (0..<5).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("streaming.db").path))
        let loaded = try await StreamingIndex.load(from: dir)

        #expect(await loaded.count == 5)
    }

    @Test("Search after load")
    func searchAfterLoad() async throws {
        let index = StreamingIndex(config: StreamingConfiguration(
            deltaCapacity: 20,
            mergeStrategy: .blocking
        ))

        let vectors = (0..<20).map { makeVector(row: 100 + $0, dim: 16) }
        let ids = (0..<20).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        let loaded = try await StreamingIndex.load(from: dir)

        let results = try await loaded.search(query: vectors[7], k: 3)
        #expect(results.contains(where: { $0.id == "v7" }))
    }

    @Test("Save auto-flushes pending delta")
    func saveRequiresFlush() async throws {
        let index = StreamingIndex(config: StreamingConfiguration(
            deltaCapacity: 100,
            mergeStrategy: .blocking
        ))

        let vectors = (0..<10).map { makeVector(row: 1_000 + $0, dim: 16) }
        let ids = (0..<10).map { "v\($0)" }
        try await index.batchInsert(vectors, ids: ids)

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        let loaded = try await StreamingIndex.load(from: dir)
        #expect(await loaded.count == 10)
    }

    @Test("Save creates streaming.db and removes JSON sidecar")
    func saveCreatesStreamingDB() async throws {
        let config = StreamingConfiguration(deltaCapacity: 100, mergeStrategy: .blocking)
        let index = StreamingIndex(config: config)

        for i in 0..<20 {
            let vector = makeVector(row: i, dim: 8)
            try await index.insert(vector, id: "vec-\(i)")
        }
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)

        let dbPath = dir.appendingPathComponent("streaming.db").path
        #expect(
            FileManager.default.fileExists(atPath: dbPath),
            "Expected streaming.db to be created"
        )

        let jsonPath = dir.appendingPathComponent("streaming.meta.json").path
        #expect(
            !FileManager.default.fileExists(atPath: jsonPath),
            "Should no longer create streaming.meta.json"
        )
    }

    @Test("Save/load roundtrip via SQLite")
    func saveLoadRoundtripViaSQLite() async throws {
        let config = StreamingConfiguration(deltaCapacity: 100, mergeStrategy: .blocking)
        let index = StreamingIndex(config: config)

        for i in 0..<30 {
            let vector = makeVector(row: i, dim: 8)
            try await index.insert(vector, id: "vec-\(i)")
        }
        try await index.flush()

        let dir = tempDirectoryURL()
        defer { try? FileManager.default.removeItem(at: dir) }

        try await index.save(to: dir)
        let loaded = try await StreamingIndex.load(from: dir)
        let count = await loaded.count
        #expect(count == 30)
    }

    private func tempDirectoryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-streaming-\(UUID().uuidString)")
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.149) + cos(i * 0.063)
        }
    }
}
