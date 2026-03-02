# GRDB Phase 3: StreamingIndex Integration

### Mission

Create `StreamingDatabase` and wire it into `StreamingIndex.save()` and
`StreamingIndex.load()`, replacing the legacy `SQLiteStructuredStore` (`.meta.db`)
and JSON sidecar (`.meta.json`) with a GRDB-backed `streaming.db` file.

After this phase:
- `save()` writes `base.anns` (binary) + `streaming.db` (GRDB). No more JSON.
- `load()` reads `streaming.db` first (if fresh). Falls back to `.meta.db` → `.meta.json`.
- Vectors are stored as per-row BLOBs (not a giant JSON array).
- `SQLiteStructuredStore` remains for fallback reads — removal is Phase 4.

**Track every completed step in `tasks/grdb-phase3-todo.md`.**
Mark each checkbox `[x]` immediately after the step passes verification.

---

### Constraints (read before writing a single line of code)

1. **TDD is mandatory.** Write failing tests first, confirm failure, then implement.
2. **Swift 6 strict concurrency.** All new types must be `Sendable`. No `nonisolated(unsafe)`.
3. **Do NOT modify `IndexDatabase.swift`** — Phase 1 output is frozen.
4. **Do NOT modify `ANNSIndex.swift`** — Phase 2 output is frozen.
5. **Do NOT delete `SQLiteStructuredStore.swift`** — still needed for `.meta.db` fallback.
6. **Do NOT change `PersistedMeta` or `MetadataValue` visibility** — they stay `private`.
   Load code lives inside `StreamingIndex` so it already has access.
7. **Do NOT touch** any Metal shader or `MetalANNSCore` types.
8. **Commit after each task.** Use exact commit messages from the todo.

---

### Verified Codebase Facts

Read each file before touching it.

| Fact | Source |
|------|--------|
| `StreamingIndex` is a `public actor` | `StreamingIndex.swift:10` |
| `MetadataValue` is a `private enum` with `.string`, `.float`, `.int64` — `Sendable, Codable` | `StreamingIndex.swift:11-15` |
| `PersistedMeta` is a `private struct` with `config`, `vectorDimension`, `allVectorData`, `allIDsList`, `deletedIDs`, `metadataByID` | `StreamingIndex.swift:17-76` |
| **`allVectorData` is `[Float]` (FLAT)** — not `[[Float]]`. Vectors are packed by dimension. | `StreamingIndex.swift:20,88` |
| `vectorDimension: Int?` determines the stride for slicing `allVectorData` | `StreamingIndex.swift:96` |
| `vector(atLogicalIndex:)` slices `allVectorData[start..<end]` by dimension | `StreamingIndex.swift:898-908` |
| `PersistedMeta.init(from:)` has legacy support: decodes `allVectorsList: [[Float]]` → flattens to `[Float]` | `StreamingIndex.swift:59-65` |
| `metadataByID: [String: [String: MetadataValue]]` — external string IDs, not UInt32 | `StreamingIndex.swift:23,95` |
| `deletedIDs` is `Set<String>` on the actor, `[String]` in PersistedMeta | `StreamingIndex.swift:91,22` |
| Current `save(to:)` writes: binary `base.anns` → JSON `streaming.meta.json` → SQLite `streaming.meta.db` | `StreamingIndex.swift:392-434` |
| Current `load(from:)` tries `streaming.meta.db` first, then `streaming.meta.json` | `StreamingIndex.swift:437-452` |
| `applyLoadedState(base:meta:)` rebuilds all actor state from `PersistedMeta` | `StreamingIndex.swift:455-471` |
| `validateLoadedMeta(_:)` checks ID uniqueness, deleted subset, metadata keys, dimension consistency | `StreamingIndex.swift:871-896` |
| `replaceDirectory(at:with:)` does atomic dir swap with backup | `StreamingIndex.swift:918+` |
| `StreamingConfiguration` is `Codable, Equatable` with `deltaCapacity`, `mergeStrategy`, `indexConfiguration` | `StreamingConfiguration.swift:4-58` |
| `config` property is `let` (immutable after init) | `StreamingIndex.swift:98` |
| Existing test `saveAndLoadEmpty` checks for `streaming.meta.db`, removes `streaming.meta.json` | `StreamingIndexPersistenceTests.swift:23-24` |
| Existing test `searchAfterLoad` removes `streaming.meta.json` before load | `StreamingIndexPersistenceTests.swift:46` |
| Existing test `saveRequiresFlush` removes `streaming.meta.json` before load | `StreamingIndexPersistenceTests.swift:68` |
| `SQLiteStructuredStore` is a raw SQLite3 KV blob store (not GRDB) | `Storage/SQLiteStructuredStore.swift:5` |
| `Sources/MetalANNS/Storage/` already exists (has `IndexDatabase.swift`, `SQLiteStructuredStore.swift`) | directory |
| `Tests/MetalANNSTests/Storage/` already exists (has `IndexDatabaseTests.swift`) | directory |

---

### Critical Data Shape: Flat Vectors

**The plan references `allVectorsList` — this property does NOT exist.**
The real property is `allVectorData: [Float]` (flat, dimension-packed).

Save path must:
1. Read `allVectorData: [Float]` and `vectorDimension: Int?`
2. Slice into `[[Float]]` by stride: `allVectorData[i*dim ..< (i+1)*dim]`
3. Pass to `StreamingDatabase.insertVectors(_:ids:)`

Load path must:
1. Call `StreamingDatabase.loadAllVectors()` → `([[Float]], [String])`
2. Flatten to `[Float]`: `vectors.flatMap { $0 }`
3. Set `allVectorData` on reconstructed `PersistedMeta`

---

### TDD Implementation Order

Work strictly in this order. Never jump ahead.

**Round 1 — StreamingDatabase (Task 7)**
Write 5 failing tests for the `StreamingDatabase` type. Implement the type
with schema, vector BLOB storage, deletion, config, and metadata. Tests pass.

**Round 2 — StreamingIndex Integration (Task 8)**
Write 2 failing tests for streaming save/load via SQLite. Rewrite
`StreamingIndex.save()` and `load()`. Update 3 existing tests. All pass.

Run after every step:
```
swift test --filter "StreamingDatabaseTests|StreamingIndexPersistenceTests" 2>&1 | tail -15
```

---

### Step-by-Step Instructions

---

#### Task 7 — Create StreamingDatabase

**Files:**
- Create: `Sources/MetalANNS/Storage/StreamingDatabase.swift`
- Create: `Tests/MetalANNSTests/Storage/StreamingDatabaseTests.swift`

**Step 7.1: Write the failing tests**

```swift
// Tests/MetalANNSTests/Storage/StreamingDatabaseTests.swift
import XCTest
import GRDB
@testable import MetalANNS

final class StreamingDatabaseTests: XCTestCase {

    private func tempDBPath() -> String {
        NSTemporaryDirectory() + "streaming-test-\(UUID().uuidString).db"
    }

    func testInsertAndFetchVectors() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        let vectors: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        let ids = ["a", "b", "c"]

        try db.insertVectors(vectors, ids: ids)

        let (loadedVectors, loadedIDs) = try db.loadAllVectors()
        XCTAssertEqual(loadedIDs, ids)
        XCTAssertEqual(loadedVectors.count, 3)
        XCTAssertEqual(loadedVectors[0], [1.0, 2.0, 3.0])
    }

    func testIncrementalInsert() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        try db.insertVectors([[1.0, 2.0]], ids: ["first"])
        try db.insertVectors([[3.0, 4.0]], ids: ["second"])

        let (vectors, ids) = try db.loadAllVectors()
        XCTAssertEqual(ids.count, 2)
        XCTAssertEqual(vectors.count, 2)
    }

    func testMarkDeleted() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        try db.insertVectors([[1.0], [2.0], [3.0]], ids: ["a", "b", "c"])
        try db.markDeleted(ids: ["b"])

        let deletedIDs = try db.loadDeletedIDs()
        XCTAssertEqual(deletedIDs, Set(["b"]))
    }

    func testSaveAndLoadConfig() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        let config = StreamingConfiguration(
            deltaCapacity: 500,
            mergeStrategy: .blocking
        )
        try db.saveConfig(config)

        let loaded = try db.loadConfig()
        XCTAssertEqual(loaded?.deltaCapacity, 500)
    }

    func testSaveAndLoadPerVectorMetadata() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let db = try StreamingDatabase(path: path)

        try db.insertVectors([[1.0]], ids: ["vec-1"])
        try db.saveVectorMetadata(id: "vec-1", metadata: ["color": "red", "score": "0.95"])

        let loaded = try db.loadVectorMetadata(id: "vec-1")
        XCTAssertEqual(loaded?["color"], "red")
        XCTAssertEqual(loaded?["score"], "0.95")
    }
}
```

**Step 7.2: Confirm tests fail**

```bash
swift test --filter StreamingDatabaseTests 2>&1 | tail -10
```

Expected: FAIL — `StreamingDatabase` not found.

**Step 7.3: Implement StreamingDatabase**

```swift
// Sources/MetalANNS/Storage/StreamingDatabase.swift
import Foundation
import GRDB

/// SQLite-backed persistence for StreamingIndex state.
///
/// Replaces the JSON-based PersistedMeta that serialized all vectors
/// as `[[Float]]` — now each vector is a row with BLOB storage.
///
/// Per-vector metadata is stored as string key-value pairs here because
/// StreamingIndex's metadata uses external string IDs (unlike MetadataStore
/// in ANNSIndex which uses UInt32 internal IDs). The two metadata systems
/// are intentionally separate.
public final class StreamingDatabase: Sendable {
    let pool: DatabasePool

    public init(path: String) throws {
        var config = Configuration()
        config.prepareDatabase { db in
            try db.execute(sql: "PRAGMA mmap_size = 268435456")
        }
        pool = try DatabasePool(path: path, configuration: config)
        try migrate()
    }

    private func migrate() throws {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1-streaming") { db in
            // Vectors table — one row per vector
            try db.create(table: "vectors") { t in
                t.autoIncrementedPrimaryKey("rowID")
                t.column("externalID", .text).notNull().unique()
                t.column("data", .blob).notNull()
            }

            // Deleted IDs
            try db.create(table: "deleted") { t in
                t.column("externalID", .text).notNull().primaryKey()
            }

            // Per-vector metadata (string key-value pairs keyed by external ID)
            try db.create(table: "vector_metadata") { t in
                t.column("externalID", .text).notNull()
                t.column("key", .text).notNull()
                t.column("value", .text).notNull()
                t.primaryKey(["externalID", "key"])
            }

            // Streaming config
            try db.create(table: "config") { t in
                t.column("key", .text).notNull().primaryKey()
                t.column("value", .text).notNull()
            }

            // Dimension tracking
            try db.create(table: "state") { t in
                t.column("key", .text).notNull().primaryKey()
                t.column("intValue", .integer)
            }
        }

        try migrator.migrate(pool)
    }

    // MARK: - Vectors

    /// Insert vectors as BLOBs (Float array → raw bytes).
    public func insertVectors(_ vectors: [[Float]], ids: [String]) throws {
        precondition(vectors.count == ids.count)
        try pool.write { db in
            let stmt = try db.makeStatement(sql: """
                INSERT OR REPLACE INTO vectors (externalID, data) VALUES (?, ?)
                """)
            for (vector, id) in zip(vectors, ids) {
                let data = vector.withUnsafeBytes { Data($0) }
                try stmt.execute(arguments: [id, data])
            }
        }
    }

    /// Load all vectors and IDs in insertion order.
    public func loadAllVectors() throws -> (vectors: [[Float]], ids: [String]) {
        try pool.read { db in
            let rows = try Row.fetchAll(db, sql: "SELECT externalID, data FROM vectors ORDER BY rowID")
            var vectors: [[Float]] = []
            var ids: [String] = []
            for row in rows {
                let id: String = row["externalID"]
                let data: Data = row["data"]
                let vector = try decodeVectorBlob(data)
                ids.append(id)
                vectors.append(vector)
            }
            return (vectors, ids)
        }
    }

    // MARK: - Deletion

    public func markDeleted(ids: Set<String>) throws {
        try pool.write { db in
            let stmt = try db.makeStatement(sql: "INSERT OR IGNORE INTO deleted (externalID) VALUES (?)")
            for id in ids {
                try stmt.execute(arguments: [id])
            }
        }
    }

    public func loadDeletedIDs() throws -> Set<String> {
        try pool.read { db in
            let ids = try String.fetchAll(db, sql: "SELECT externalID FROM deleted")
            return Set(ids)
        }
    }

    // MARK: - Config

    public func saveConfig(_ config: StreamingConfiguration) throws {
        let data = try JSONEncoder().encode(config)
        let json = String(data: data, encoding: .utf8)!
        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["streaming.config", json]
            )
        }
    }

    public func loadConfig() throws -> StreamingConfiguration? {
        try pool.read { db in
            guard let json = try String.fetchOne(
                db, sql: "SELECT value FROM config WHERE key = ?",
                arguments: ["streaming.config"]
            ) else { return nil }
            return try JSONDecoder().decode(StreamingConfiguration.self, from: Data(json.utf8))
        }
    }

    // MARK: - Per-Vector Metadata

    public func saveVectorMetadata(id: String, metadata: [String: String]) throws {
        try pool.write { db in
            try db.execute(
                sql: "DELETE FROM vector_metadata WHERE externalID = ?",
                arguments: [id]
            )
            let stmt = try db.makeStatement(sql: """
                INSERT INTO vector_metadata (externalID, key, value) VALUES (?, ?, ?)
                """)
            for (key, value) in metadata {
                try stmt.execute(arguments: [id, key, value])
            }
        }
    }

    public func loadVectorMetadata(id: String) throws -> [String: String]? {
        try pool.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: "SELECT key, value FROM vector_metadata WHERE externalID = ?",
                arguments: [id]
            )
            guard !rows.isEmpty else { return nil }
            var result: [String: String] = [:]
            for row in rows {
                result[row["key"]] = row["value"]
            }
            return result
        }
    }

    /// Batch-save all per-vector metadata (used during full save).
    public func saveAllVectorMetadata(_ metadataByID: [String: [String: String]]) throws {
        try pool.write { db in
            try db.execute(sql: "DELETE FROM vector_metadata")
            let stmt = try db.makeStatement(sql: """
                INSERT INTO vector_metadata (externalID, key, value) VALUES (?, ?, ?)
                """)
            for (id, entries) in metadataByID {
                for (key, value) in entries {
                    try stmt.execute(arguments: [id, key, value])
                }
            }
        }
    }

    /// Load all per-vector metadata.
    public func loadAllVectorMetadata() throws -> [String: [String: String]] {
        try pool.read { db in
            let rows = try Row.fetchAll(db, sql: "SELECT externalID, key, value FROM vector_metadata")
            var result: [String: [String: String]] = [:]
            for row in rows {
                let id: String = row["externalID"]
                let key: String = row["key"]
                let value: String = row["value"]
                result[id, default: [:]][key] = value
            }
            return result
        }
    }

    // MARK: - State

    public func saveVectorDimension(_ dimension: Int) throws {
        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO state (key, intValue) VALUES (?, ?)",
                arguments: ["vectorDimension", dimension]
            )
        }
    }

    public func loadVectorDimension() throws -> Int? {
        try pool.read { db in
            try Int.fetchOne(db, sql: "SELECT intValue FROM state WHERE key = 'vectorDimension'")
        }
    }

    // MARK: - Internal

    private func decodeVectorBlob(_ data: Data) throws -> [Float] {
        let scalarSize = MemoryLayout<Float>.size
        guard data.count % scalarSize == 0 else {
            throw ANNSError.corruptFile("Invalid streaming vector blob size: \(data.count)")
        }
        let floatCount = data.count / scalarSize
        var vector = Array(repeating: Float.zero, count: floatCount)
        _ = vector.withUnsafeMutableBytes { buffer in
            data.copyBytes(to: buffer)
        }
        return vector
    }
}
```

**Step 7.4: Verify tests pass**

```bash
swift test --filter StreamingDatabaseTests 2>&1 | tail -10
```

Expected: PASS — all 5 tests green.

**Step 7.5: Commit + update todo**

```bash
git add Sources/MetalANNS/Storage/StreamingDatabase.swift \
        Tests/MetalANNSTests/Storage/StreamingDatabaseTests.swift
git commit -m "feat: add StreamingDatabase for SQLite-backed streaming state"
```

---

#### Task 8 — Wire StreamingDatabase into StreamingIndex Save/Load

**Files:**
- Modify: `Sources/MetalANNS/StreamingIndex.swift`
- Modify: `Tests/MetalANNSTests/StreamingIndexPersistenceTests.swift`

**Step 8.1: Write the failing tests**

Add to `StreamingIndexPersistenceTests.swift`:

```swift
@Test("Save creates streaming.db and removes JSON sidecar")
func saveCreatesStreamingDB() async throws {
    let config = StreamingConfiguration(deltaCapacity: 100, mergeStrategy: .blocking)
    let index = StreamingIndex(config: config)

    for i in 0..<20 {
        let vector = makeVector(row: i, dim: 8)
        try await index.insert(vector: vector, id: "vec-\(i)")
    }
    try await index.flush()

    let dir = tempDirectoryURL()
    defer { try? FileManager.default.removeItem(at: dir) }

    try await index.save(to: dir)

    // streaming.db should exist
    let dbPath = dir.appendingPathComponent("streaming.db").path
    #expect(
        FileManager.default.fileExists(atPath: dbPath),
        "Expected streaming.db to be created"
    )

    // Old streaming.meta.json should NOT exist
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
        try await index.insert(vector: vector, id: "vec-\(i)")
    }
    try await index.flush()

    let dir = tempDirectoryURL()
    defer { try? FileManager.default.removeItem(at: dir) }

    try await index.save(to: dir)
    let loaded = try await StreamingIndex.load(from: dir)
    let count = await loaded.count
    #expect(count == 30)
}
```

**Step 8.2: Confirm tests fail**

```bash
swift test --filter "saveCreatesStreamingDB|saveLoadRoundtripViaSQLite" 2>&1 | tail -10
```

Expected: FAIL — `streaming.db` not created by current save path.

**Step 8.3: Rewrite `StreamingIndex.save(to:)`**

Replace the current `save(to:)` method (lines 392-434) in `StreamingIndex.swift`:

```swift
public func save(to url: URL) async throws {
    try checkBackgroundMergeError()
    try await flush()

    guard let base else {
        throw ANNSError.constructionFailed("Nothing to save — index is empty")
    }

    // Snapshot all state before any cross-actor await points.
    let configSnapshot = config
    let vectorDimensionSnapshot = vectorDimension
    let allVectorDataSnapshot = allVectorData
    let allIDsListSnapshot = allIDsList
    let deletedIDsSnapshot = deletedIDs
    let metadataSnapshot = metadataByID

    // Validate before writing.
    let meta = PersistedMeta(
        config: configSnapshot,
        vectorDimension: vectorDimensionSnapshot,
        allVectorData: allVectorDataSnapshot,
        allIDsList: allIDsListSnapshot,
        deletedIDs: Array(deletedIDsSnapshot),
        metadataByID: metadataSnapshot
    )
    try Self.validateLoadedMeta(meta)

    let fileManager = FileManager.default
    let parentURL = url.deletingLastPathComponent()
    let tempURL = parentURL.appendingPathComponent(
        ".\(url.lastPathComponent).tmp-\(UUID().uuidString)"
    )
    let tempBaseURL = tempURL.appendingPathComponent("base.anns")

    try fileManager.createDirectory(at: tempURL, withIntermediateDirectories: true)
    do {
        // Save binary base index (unchanged)
        try await base.save(to: tempBaseURL)

        // Save structured data to SQLite
        let dbPath = tempURL.appendingPathComponent("streaming.db").path
        let db = try StreamingDatabase(path: dbPath)

        // Slice flat allVectorData → [[Float]] by dimension
        var slicedVectors: [[Float]] = []
        if let dim = vectorDimensionSnapshot, dim > 0 {
            for i in 0 ..< allIDsListSnapshot.count {
                let start = i * dim
                let end = start + dim
                slicedVectors.append(Array(allVectorDataSnapshot[start..<end]))
            }
        }
        try db.insertVectors(slicedVectors, ids: allIDsListSnapshot)
        try db.saveConfig(configSnapshot)
        try db.markDeleted(ids: deletedIDsSnapshot)

        if let dim = vectorDimensionSnapshot {
            try db.saveVectorDimension(dim)
        }

        // Convert MetadataValue → JSON strings for storage.
        // MetadataValue is private to StreamingIndex, accessible here.
        let encoder = JSONEncoder()
        var stringMetadata: [String: [String: String]] = [:]
        for (id, entries) in metadataSnapshot {
            var converted: [String: String] = [:]
            for (key, value) in entries {
                let data = try encoder.encode(value)
                converted[key] = String(data: data, encoding: .utf8)
            }
            stringMetadata[id] = converted
        }
        try db.saveAllVectorMetadata(stringMetadata)

        try Self.replaceDirectory(at: url, with: tempURL)
    } catch {
        try? fileManager.removeItem(at: tempURL)
        throw error
    }
}
```

**Step 8.4: Verify save test passes**

```bash
swift test --filter saveCreatesStreamingDB 2>&1 | tail -10
```

Expected: PASS.

**Step 8.5: Add freshness check helper**

Add to `StreamingIndex.swift`:

```swift
private nonisolated static func hasFreshStreamingDatabase(
    at dbPath: String,
    forBaseANNS baseANNSPath: String
) -> Bool {
    let fm = FileManager.default
    guard
        fm.fileExists(atPath: dbPath),
        fm.fileExists(atPath: baseANNSPath),
        let dbAttrs = try? fm.attributesOfItem(atPath: dbPath),
        let baseAttrs = try? fm.attributesOfItem(atPath: baseANNSPath),
        let dbDate = dbAttrs[.modificationDate] as? Date,
        let baseDate = baseAttrs[.modificationDate] as? Date
    else {
        return false
    }
    return dbDate >= baseDate
}
```

**Step 8.6: Rewrite `StreamingIndex.load(from:)`**

Replace the current `load(from:)` method (lines 437-452):

```swift
public static func load(from url: URL) async throws -> StreamingIndex {
    let dbURL = url.appendingPathComponent("streaming.db")
    let baseANNSPath = url.appendingPathComponent("base.anns").path
    let hasFreshDB = hasFreshStreamingDatabase(at: dbURL.path, forBaseANNS: baseANNSPath)

    let meta: PersistedMeta

    if hasFreshDB {
        do {
            let db = try StreamingDatabase(path: dbURL.path)
            guard let config = try db.loadConfig() else {
                throw ANNSError.corruptFile("Missing streaming config in database")
            }

            let (vectors, ids) = try db.loadAllVectors()
            let deletedIDs = try db.loadDeletedIDs()
            let allStringMetadata = try db.loadAllVectorMetadata()
            let dimension = try db.loadVectorDimension()

            // Convert string metadata back to MetadataValue.
            let decoder = JSONDecoder()
            var metadataByID: [String: [String: MetadataValue]] = [:]
            for (id, entries) in allStringMetadata {
                var converted: [String: MetadataValue] = [:]
                for (key, jsonStr) in entries {
                    converted[key] = try decoder.decode(
                        MetadataValue.self, from: Data(jsonStr.utf8)
                    )
                }
                metadataByID[id] = converted
            }

            // Flatten [[Float]] → [Float] for PersistedMeta
            let flatVectorData = vectors.flatMap { $0 }

            meta = PersistedMeta(
                config: config,
                vectorDimension: dimension,
                allVectorData: flatVectorData,
                allIDsList: ids,
                deletedIDs: Array(deletedIDs),
                metadataByID: metadataByID
            )
        } catch {
            // Corrupt DB — try legacy paths
            meta = try loadLegacyMeta(from: url)
        }
    } else {
        // No fresh streaming.db — backward compatibility
        meta = try loadLegacyMeta(from: url)
    }

    try validateLoadedMeta(meta)

    let loadedBase = try await ANNSIndex.load(
        from: url.appendingPathComponent("base.anns")
    )
    let streaming = StreamingIndex(config: meta.config)
    await streaming.applyLoadedState(base: loadedBase, meta: meta)
    return streaming
}

/// Fallback load from legacy .meta.db or .meta.json sidecars.
private static func loadLegacyMeta(from url: URL) throws -> PersistedMeta {
    // Try SQLiteStructuredStore (.meta.db) first
    let sqliteMetaURL = url.appendingPathComponent("streaming.meta.db")
    if let sqliteMeta = try SQLiteStructuredStore.load(PersistedMeta.self, from: sqliteMetaURL) {
        return sqliteMeta
    }

    // Fall back to JSON (.meta.json)
    let jsonURL = url.appendingPathComponent("streaming.meta.json")
    guard FileManager.default.fileExists(atPath: jsonURL.path) else {
        throw ANNSError.corruptFile("Missing both streaming.db and streaming.meta.json")
    }
    let data = try Data(contentsOf: jsonURL)
    return try JSONDecoder().decode(PersistedMeta.self, from: data)
}
```

**Step 8.7: Verify roundtrip test passes**

```bash
swift test --filter saveLoadRoundtripViaSQLite 2>&1 | tail -10
```

Expected: PASS.

**Step 8.8: Update existing streaming persistence tests**

In `StreamingIndexPersistenceTests.swift`, update the three existing tests:

**`saveAndLoadEmpty`** — change line 23:
```swift
// Old:
#expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("streaming.meta.db").path))
// New:
#expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("streaming.db").path))
```

Remove line 24 (no longer need to delete `streaming.meta.json` — it's not created):
```swift
// Old:
try? FileManager.default.removeItem(at: dir.appendingPathComponent("streaming.meta.json"))
// New: (delete this line entirely)
```

**`searchAfterLoad`** — remove line 46:
```swift
// Old:
try? FileManager.default.removeItem(at: dir.appendingPathComponent("streaming.meta.json"))
// New: (delete this line entirely)
```

**`saveRequiresFlush`** — remove line 68:
```swift
// Old:
try? FileManager.default.removeItem(at: dir.appendingPathComponent("streaming.meta.json"))
// New: (delete this line entirely)
```

**Step 8.9: Verify all streaming persistence tests pass**

```bash
swift test --filter StreamingIndexPersistenceTests 2>&1 | tail -15
```

Expected: PASS — all 5 tests (3 existing + 2 new) green.

**Step 8.10: Run full regression suite**

```bash
swift test 2>&1 | grep -E "passed|failed|error:" | tail -5
```

Expected: Zero regressions.

**Step 8.11: Commit + update todo**

```bash
git add Sources/MetalANNS/StreamingIndex.swift \
        Tests/MetalANNSTests/StreamingIndexPersistenceTests.swift
git commit -m "feat: StreamingIndex save/load uses SQLite, JSON fallback for backward compat"
```

---

### Definition of Done

Phase 3 is complete when ALL of the following are true:

- [ ] `swift test --filter StreamingDatabaseTests` → all PASS
- [ ] `swift test --filter StreamingIndexPersistenceTests` → all PASS (5 tests)
- [ ] `swift test --filter ANNSIndexTests` → all PASS (no regression from Phase 2)
- [ ] `swift test` → zero regressions
- [ ] `tasks/grdb-phase3-todo.md` → all checkboxes marked `[x]`
- [ ] Two commits in git log with exact messages above

---

### Anti-patterns (do NOT do these)

| Anti-pattern | Why |
|---|---|
| Referencing `allVectorsList` property | Does NOT exist. The real property is `allVectorData: [Float]` (flat). Slice by `vectorDimension`. |
| Storing vectors as a single JSON blob in config table | Defeats the purpose — use per-row BLOBs in `vectors` table |
| Changing `MetadataValue` visibility to `public` | Must stay `private`. Encode/decode within `StreamingIndex` where it's accessible. |
| Changing `PersistedMeta` visibility | Must stay `private`. All reconstruction happens inside `StreamingIndex`. |
| Deleting `SQLiteStructuredStore.swift` | Still needed for `.meta.db` fallback reads of legacy files |
| Skipping `validateLoadedMeta` on the SQLite load path | Validation catches corrupt data regardless of source |
| Modifying `ANNSIndex.swift` or `IndexDatabase.swift` | Phase 1/2 outputs are frozen |
| Forgetting `vectorDimension` in `StreamingDatabase` save | Without dimension, flat vector data can't be reconstructed on load |

---

### What Comes Next (Phase 4)

Phase 4 is the final cleanup phase:
- Task 9: Full integration tests + backward compatibility verification
- Task 10: Remove dead code (`SQLiteStructuredStore.swift`, JSON sidecar writes,
  `metadataURL`/`metadataDBURL` helpers)

Prompt: `docs/prompts/grdb-phase4-cleanup.md` (to be written after Phase 3 review).
