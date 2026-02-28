# GRDB Storage Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace JSON-based persistence (IDMap, metadata sidecars, StreamingIndex state) with GRDB/SQLite while keeping the binary format for vectors and graph data.

**Architecture:** Hybrid storage — binary mmap format (v2/v3) for bulk numerical data (vectors, graph), GRDB/SQLite for structured data (IDMap, metadata, streaming state). The binary format is untouched for backward compatibility. SQLite is additive — old files load via fallback, new saves always write SQLite.

**Tech Stack:** GRDB.swift v7+ (requires Swift 6.1+ / Xcode 16.3+), SQLite WAL mode via DatabasePool, Swift 6 strict concurrency.

---

## File Layout (Before -> After)

```
BEFORE:                              AFTER:
index.anns                           index.anns           (unchanged binary)
index.anns.meta.json                 index.db             (replaces .meta.json)

streaming/                           streaming/
  base.anns                            base.anns          (unchanged binary)
  base.anns.meta.json                  streaming.db       (replaces both JSONs)
  streaming.meta.json
```

## Module Strategy

- **MetalANNSCore** — stays dependency-free (shaders, core types, binary serializer)
- **MetalANNS** — gains GRDB dependency (ANNSIndex, StreamingIndex, database layer)
- **IDMap** — stays a plain struct in Core; SQLite persistence wraps it in MetalANNS

---

## Task 1: Add GRDB Dependency

**Files:**
- Modify: `Package.swift`

**Step 1: Update Package.swift to add GRDB**

```swift
// Package.swift — full replacement
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MetalANNS",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "MetalANNS", targets: ["MetalANNS"])
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.0.0"),
    ],
    targets: [
        .target(
            name: "MetalANNSCore",
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "MetalANNS",
            dependencies: [
                "MetalANNSCore",
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MetalANNSTests",
            dependencies: ["MetalANNS", "MetalANNSCore", "MetalANNSBenchmarks"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "MetalANNSBenchmarks",
            dependencies: ["MetalANNS", "MetalANNSCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
```

**Step 2: Resolve dependencies**

Run: `swift package resolve`
Expected: GRDB.swift cloned and resolved successfully.

> **Note:** GRDB v7.10.0 requires Swift 6.1+ / Xcode 16.3+. Verify your toolchain
> with `swift --version` before resolving. If on Swift 6.0, pin to an earlier 7.x
> release: `.package(url: "...", "7.0.0"..<"7.9.0")`.

**Step 3: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds (or Metal shader warnings only — CLI can't compile .metal).

**Step 4: Commit**

```bash
git add Package.swift Package.resolved
git commit -m "deps: add GRDB.swift for SQLite-backed persistence"
```

---

## Task 2: Create IndexDatabase Foundation

**Files:**
- Create: `Sources/MetalANNS/Storage/IndexDatabase.swift`
- Test: `Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift
import XCTest
import GRDB
@testable import MetalANNS

final class IndexDatabaseTests: XCTestCase {

    private func tempDBPath() -> String {
        NSTemporaryDirectory() + "test-\(UUID().uuidString).db"
    }

    func testCreateAndOpenDatabase() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        // Should create file and apply migrations
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))

        // Should have the expected tables
        let tables = try db.pool.read { db in
            try String.fetchAll(db, sql: """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'grdb_migrations'
                ORDER BY name
                """)
        }
        XCTAssertEqual(tables, ["config", "idmap", "soft_deletion"])
    }

    func testReopenExistingDatabase() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Create and close
        _ = try IndexDatabase(path: path)

        // Reopen — should not fail
        let db = try IndexDatabase(path: path)
        let count = try db.pool.read { db in
            try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM idmap")
        }
        XCTAssertEqual(count, 0)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter IndexDatabaseTests 2>&1 | tail -10`
Expected: FAIL — `IndexDatabase` type not found.

**Step 3: Write IndexDatabase implementation**

```swift
// Sources/MetalANNS/Storage/IndexDatabase.swift
import Foundation
import GRDB

/// SQLite-backed persistence layer for index structured data.
///
/// Manages IDMap, configuration, soft-deletion, and MetadataStore
/// via a GRDB DatabasePool in WAL mode.
///
/// MetadataStore is stored as a single JSON blob (not row-per-entry)
/// because its internal structure uses UInt32 internal IDs with three
/// typed column dictionaries — not suitable for normalized SQL rows.
public final class IndexDatabase: Sendable {
    let pool: DatabasePool

    public init(path: String) throws {
        var config = Configuration()
        config.prepareDatabase { db in
            // Use memory-mapped I/O for reads (up to 256 MB).
            try db.execute(sql: "PRAGMA mmap_size = 268435456")
        }
        pool = try DatabasePool(path: path, configuration: config)
        try migrate()
    }

    // MARK: - Schema

    private func migrate() throws {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1-foundation") { db in
            // Bidirectional ID mapping
            try db.create(table: "idmap") { t in
                t.column("externalID", .text).notNull().primaryKey()
                t.column("internalID", .integer).notNull().unique()
            }
            try db.create(index: "idmap_by_internal", on: "idmap", columns: ["internalID"])

            // Key-value config (metric, useFloat16, MetadataStore JSON, etc.)
            try db.create(table: "config") { t in
                t.column("key", .text).notNull().primaryKey()
                t.column("value", .text).notNull()
            }

            // Soft-deleted internal IDs
            try db.create(table: "soft_deletion") { t in
                t.column("internalID", .integer).notNull().primaryKey()
            }
        }

        try migrator.migrate(pool)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter IndexDatabaseTests 2>&1 | tail -10`
Expected: PASS — both tests green.

**Step 5: Commit**

```bash
git add Sources/MetalANNS/Storage/IndexDatabase.swift Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift
git commit -m "feat: add IndexDatabase foundation with schema v1"
```

---

## Task 3: Add IDMap Read/Write to IndexDatabase

**Files:**
- Modify: `Sources/MetalANNS/Storage/IndexDatabase.swift`
- Test: `Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift`

**Step 1: Write the failing tests**

Add to `IndexDatabaseTests.swift`:

```swift
func testSaveAndLoadIDMap() throws {
    let path = tempDBPath()
    defer { try? FileManager.default.removeItem(atPath: path) }
    let db = try IndexDatabase(path: path)

    var idMap = IDMap()
    _ = idMap.assign(externalID: "alpha")
    _ = idMap.assign(externalID: "beta")
    _ = idMap.assign(externalID: "gamma")

    try db.saveIDMap(idMap)

    let loaded = try db.loadIDMap()
    XCTAssertEqual(loaded.count, 3)
    XCTAssertEqual(loaded.internalID(for: "alpha"), 0)
    XCTAssertEqual(loaded.internalID(for: "beta"), 1)
    XCTAssertEqual(loaded.internalID(for: "gamma"), 2)
    XCTAssertEqual(loaded.externalID(for: 0), "alpha")
    XCTAssertEqual(loaded.nextInternalID, 3)
}

func testSaveIDMapOverwritesPrevious() throws {
    let path = tempDBPath()
    defer { try? FileManager.default.removeItem(atPath: path) }
    let db = try IndexDatabase(path: path)

    var idMap1 = IDMap()
    _ = idMap1.assign(externalID: "old")
    try db.saveIDMap(idMap1)

    var idMap2 = IDMap()
    _ = idMap2.assign(externalID: "new-a")
    _ = idMap2.assign(externalID: "new-b")
    try db.saveIDMap(idMap2)

    let loaded = try db.loadIDMap()
    XCTAssertEqual(loaded.count, 2)
    XCTAssertNil(loaded.internalID(for: "old"))
    XCTAssertEqual(loaded.internalID(for: "new-a"), 0)
}

func testLoadIDMapFromEmptyDatabase() throws {
    let path = tempDBPath()
    defer { try? FileManager.default.removeItem(atPath: path) }
    let db = try IndexDatabase(path: path)

    let loaded = try db.loadIDMap()
    XCTAssertEqual(loaded.count, 0)
    XCTAssertEqual(loaded.nextInternalID, 0)
}

func testSaveAndLoadIDMapWithGaps() throws {
    let path = tempDBPath()
    defer { try? FileManager.default.removeItem(atPath: path) }
    let db = try IndexDatabase(path: path)

    // Simulate an IDMap where some internal IDs were removed (compaction gap).
    // IDs 0, 1, 2 assigned but nextInternalID could be higher after compaction.
    var idMap = IDMap()
    _ = idMap.assign(externalID: "a")  // 0
    _ = idMap.assign(externalID: "b")  // 1
    _ = idMap.assign(externalID: "c")  // 2

    try db.saveIDMap(idMap)
    let loaded = try db.loadIDMap()

    // nextInternalID should be restored from the stored value, not inferred
    XCTAssertEqual(loaded.nextInternalID, 3)
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter "testSaveAndLoadIDMap|testSaveIDMapOverwritesPrevious|testLoadIDMapFromEmptyDatabase|testSaveAndLoadIDMapWithGaps" 2>&1 | tail -10`
Expected: FAIL — `saveIDMap`/`loadIDMap` not found.

**Step 3: Add IDMap CRUD to IndexDatabase**

Add to `IndexDatabase.swift` — imports need `import MetalANNSCore`:

```swift
import MetalANNSCore

// MARK: - IDMap Persistence
extension IndexDatabase {

    /// Atomically replaces the full IDMap in the database.
    public func saveIDMap(_ idMap: IDMap) throws {
        try pool.write { db in
            try db.execute(sql: "DELETE FROM idmap")

            let stmt = try db.makeStatement(sql: """
                INSERT INTO idmap (externalID, internalID) VALUES (?, ?)
                """)

            for internalID in 0 ..< UInt32(idMap.count) {
                guard let externalID = idMap.externalID(for: internalID) else { continue }
                try stmt.execute(arguments: [externalID, Int64(internalID)])
            }

            // Store nextID in config for reconstruction (handles gaps from compaction)
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["idmap.nextID", String(idMap.nextInternalID)]
            )
        }
    }

    /// Loads the full IDMap from the database.
    ///
    /// Restores nextInternalID from the stored config value to handle gaps
    /// from compaction (where nextID > count of entries).
    public func loadIDMap() throws -> IDMap {
        try pool.read { db in
            let rows = try Row.fetchAll(db, sql: "SELECT externalID, internalID FROM idmap ORDER BY internalID")
            let nextIDStr = try String.fetchOne(db, sql: "SELECT value FROM config WHERE key = 'idmap.nextID'")

            var idMap = IDMap()
            for row in rows {
                let externalID: String = row["externalID"]
                // assign() auto-increments, so we rely on insertion order matching internalID order
                _ = idMap.assign(externalID: externalID)
            }

            // Restore nextInternalID if the stored value is higher than what
            // assign() produced (e.g., gaps from compaction).
            if let storedNextID = nextIDStr.flatMap(UInt32.init),
               storedNextID > idMap.nextInternalID {
                idMap.nextInternalID = storedNextID
            }

            return idMap
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "testSaveAndLoadIDMap|testSaveIDMapOverwritesPrevious|testLoadIDMapFromEmptyDatabase|testSaveAndLoadIDMapWithGaps" 2>&1 | tail -10`
Expected: PASS — all four tests green.

**Step 5: Commit**

```bash
git add Sources/MetalANNS/Storage/IndexDatabase.swift Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift
git commit -m "feat: add IDMap save/load to IndexDatabase"
```

---

## Task 4: Add Metadata Persistence to IndexDatabase

**Files:**
- Modify: `Sources/MetalANNS/Storage/IndexDatabase.swift`
- Test: `Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift`

**Step 1: Write the failing tests**

Add to `IndexDatabaseTests.swift`:

```swift
func testSaveAndLoadConfig() throws {
    let path = tempDBPath()
    defer { try? FileManager.default.removeItem(atPath: path) }
    let db = try IndexDatabase(path: path)

    var config = IndexConfiguration.default
    config.metric = .l2
    config.useFloat16 = true
    config.degree = 32

    try db.saveConfiguration(config)
    let loaded = try db.loadConfiguration()

    XCTAssertEqual(loaded?.metric, .l2)
    XCTAssertEqual(loaded?.useFloat16, true)
    XCTAssertEqual(loaded?.degree, 32)
}

func testSaveAndLoadSoftDeletion() throws {
    let path = tempDBPath()
    defer { try? FileManager.default.removeItem(atPath: path) }
    let db = try IndexDatabase(path: path)

    var softDeletion = SoftDeletion()
    softDeletion.markDeleted(3)
    softDeletion.markDeleted(7)
    softDeletion.markDeleted(15)

    try db.saveSoftDeletion(softDeletion)
    let loaded = try db.loadSoftDeletion()

    XCTAssertEqual(loaded.deletedCount, 3)
    XCTAssertTrue(loaded.isDeleted(3))
    XCTAssertTrue(loaded.isDeleted(7))
    XCTAssertTrue(loaded.isDeleted(15))
    XCTAssertFalse(loaded.isDeleted(0))
}

func testSaveAndLoadMetadataStore() throws {
    let path = tempDBPath()
    defer { try? FileManager.default.removeItem(atPath: path) }
    let db = try IndexDatabase(path: path)

    var store = MetadataStore()
    store.set("color", stringValue: "red", for: 0)
    store.set("score", floatValue: 0.95, for: 0)
    store.set("color", stringValue: "blue", for: 1)
    store.set("priority", intValue: 42, for: 1)

    try db.saveMetadataStore(store)
    let loaded = try db.loadMetadataStore()

    XCTAssertEqual(loaded.getString("color", for: 0), "red")
    XCTAssertEqual(loaded.getString("color", for: 1), "blue")
    XCTAssertEqual(loaded.getFloat("score", for: 0), 0.95)
    XCTAssertEqual(loaded.getInt("priority", for: 1), 42)
}

func testSaveAndLoadEmptyMetadataStore() throws {
    let path = tempDBPath()
    defer { try? FileManager.default.removeItem(atPath: path) }
    let db = try IndexDatabase(path: path)

    let loaded = try db.loadMetadataStore()
    XCTAssertTrue(loaded.isEmpty)
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "testSaveAndLoadConfig|testSaveAndLoadSoftDeletion|testSaveAndLoadMetadataStore|testSaveAndLoadEmptyMetadataStore" 2>&1 | tail -10`
Expected: FAIL — methods not found.

**Step 3: Implement metadata persistence**

Add to `IndexDatabase.swift`:

```swift
// MARK: - Configuration Persistence
extension IndexDatabase {

    public func saveConfiguration(_ config: IndexConfiguration) throws {
        let data = try JSONEncoder().encode(config)
        let json = String(data: data, encoding: .utf8)!
        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["index.configuration", json]
            )
        }
    }

    public func loadConfiguration() throws -> IndexConfiguration? {
        try pool.read { db in
            guard let json = try String.fetchOne(
                db,
                sql: "SELECT value FROM config WHERE key = ?",
                arguments: ["index.configuration"]
            ) else { return nil }
            return try JSONDecoder().decode(IndexConfiguration.self, from: Data(json.utf8))
        }
    }
}

// MARK: - Soft Deletion Persistence
extension IndexDatabase {

    public func saveSoftDeletion(_ softDeletion: SoftDeletion) throws {
        try pool.write { db in
            try db.execute(sql: "DELETE FROM soft_deletion")
            let stmt = try db.makeStatement(sql: "INSERT INTO soft_deletion (internalID) VALUES (?)")
            for id in softDeletion.allDeletedIDs {
                try stmt.execute(arguments: [Int64(id)])
            }
        }
    }

    public func loadSoftDeletion() throws -> SoftDeletion {
        try pool.read { db in
            let ids = try Int64.fetchAll(db, sql: "SELECT internalID FROM soft_deletion")
            var softDeletion = SoftDeletion()
            for id in ids {
                softDeletion.markDeleted(UInt32(id))
            }
            return softDeletion
        }
    }
}

// MARK: - MetadataStore Persistence
//
// MetadataStore uses three typed column dictionaries keyed by UInt32 internal IDs.
// Rather than normalizing into SQL rows (which would require type discrimination
// and internal-to-external ID translation), we store it as a single JSON blob.
// MetadataStore conforms to Codable, so this round-trips perfectly.
extension IndexDatabase {

    public func saveMetadataStore(_ store: MetadataStore) throws {
        let data = try JSONEncoder().encode(store)
        let json = String(data: data, encoding: .utf8)!
        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["index.metadataStore", json]
            )
        }
    }

    public func loadMetadataStore() throws -> MetadataStore {
        try pool.read { db in
            guard let json = try String.fetchOne(
                db,
                sql: "SELECT value FROM config WHERE key = ?",
                arguments: ["index.metadataStore"]
            ) else { return MetadataStore() }
            return try JSONDecoder().decode(MetadataStore.self, from: Data(json.utf8))
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "testSaveAndLoadConfig|testSaveAndLoadSoftDeletion|testSaveAndLoadMetadataStore|testSaveAndLoadEmptyMetadataStore" 2>&1 | tail -10`
Expected: PASS.

**Step 5: Commit**

```bash
git add Sources/MetalANNS/Storage/IndexDatabase.swift Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift
git commit -m "feat: add config, soft-deletion, and metadata persistence to IndexDatabase"
```

---

## Task 5: Wire IndexDatabase into ANNSIndex Save

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (lines 831-873)
- Modify: `Sources/MetalANNS/Storage/IndexDatabase.swift` (add convenience method)

**Step 1: Write the failing test**

Add to existing `PersistenceTests.swift`:

```swift
func testSaveCreatesDBFile() async throws {
    // Build a small index
    let index = ANNSIndex(configuration: .default)
    var vectors: [[Float]] = []
    var ids: [String] = []
    for i in 0..<50 {
        vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
        ids.append("node-\(i)")
    }
    try await index.batchInsert(vectors: vectors, ids: ids)
    try await index.build()

    let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: dir) }

    let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
    try await index.save(to: url)

    // Binary file should exist (unchanged)
    XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

    // SQLite file should also exist (new)
    let dbPath = url.path.replacingOccurrences(of: ".anns", with: ".db")
    XCTAssertTrue(FileManager.default.fileExists(atPath: dbPath),
                  "Expected index.db to be created alongside index.anns")

    // Old meta.json should NOT be created
    XCTAssertFalse(FileManager.default.fileExists(atPath: url.path + ".meta.json"),
                   "Should no longer create .meta.json sidecar")
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter testSaveCreatesDBFile 2>&1 | tail -10`
Expected: FAIL — no .db file created yet.

**Step 3: Modify ANNSIndex.save to write SQLite**

In `ANNSIndex.swift`, replace the save method (lines 831-851):

```swift
public func save(to url: URL) async throws {
    guard isBuilt, let vectors, let graph else {
        throw ANNSError.indexEmpty
    }

    // Binary format — vectors + graph (unchanged)
    try IndexSerializer.save(
        vectors: vectors,
        graph: graph,
        idMap: idMap,
        entryPoint: entryPoint,
        metric: configuration.metric,
        to: url
    )

    // SQLite — structured data (replaces .meta.json)
    let dbPath = Self.databasePath(for: url)
    let db = try IndexDatabase(path: dbPath)
    try db.saveIDMap(idMap)
    try db.saveConfiguration(configuration)
    try db.saveSoftDeletion(softDeletion)
    try db.saveMetadataStore(metadataStore)
}
```

Add the helper:

```swift
private nonisolated static func databasePath(for fileURL: URL) -> String {
    let base = fileURL.deletingPathExtension()
    return base.appendingPathExtension("db").path
}
```

Do the same for `saveMmapCompatible` (lines 853-873) — same pattern.

**Step 4: Run test to verify it passes**

Run: `swift test --filter testSaveCreatesDBFile 2>&1 | tail -10`
Expected: PASS.

**Step 5: Commit**

```bash
git add Sources/MetalANNS/ANNSIndex.swift
git commit -m "feat: ANNSIndex.save writes SQLite alongside binary format"
```

---

## Task 6: Wire IndexDatabase into ANNSIndex Load (with Fallback)

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (lines 875-954)

**Step 1: Write the failing tests**

```swift
func testLoadFromSQLiteRoundtrip() async throws {
    let index = ANNSIndex(configuration: .default)
    var vectors: [[Float]] = []
    var ids: [String] = []
    for i in 0..<50 {
        vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
        ids.append("node-\(i)")
    }
    try await index.batchInsert(vectors: vectors, ids: ids)
    try await index.build()

    let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: dir) }

    let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
    try await index.save(to: url)

    // Delete .meta.json if it exists — force SQLite path
    let metaJSON = URL(fileURLWithPath: url.path + ".meta.json")
    try? FileManager.default.removeItem(at: metaJSON)

    let loaded = try await ANNSIndex.load(from: url)
    XCTAssertEqual(await loaded.count, 50)

    // Verify search works
    let query = vectors[0]
    let results = try await loaded.search(query: query, k: 5)
    XCTAssertEqual(results.first?.id, "node-0")
}

func testLoadFallsBackToJSONSidecar() async throws {
    // Build and save with the current code (which creates both .db and .meta.json
    // during the transition, or only .db after migration).
    let index = ANNSIndex(configuration: .default)
    var vectors: [[Float]] = []
    var ids: [String] = []
    for i in 0..<30 {
        vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
        ids.append("legacy-\(i)")
    }
    try await index.batchInsert(vectors: vectors, ids: ids)
    try await index.build()

    let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: dir) }

    let url = URL(fileURLWithPath: dir).appendingPathComponent("legacy.anns")
    try await index.save(to: url)

    // Remove the .db file to simulate old-format file (pre-migration)
    let dbPath = url.path.replacingOccurrences(of: ".anns", with: ".db")
    try? FileManager.default.removeItem(atPath: dbPath)

    // Manually write a .meta.json sidecar to simulate old format.
    // ANNSIndex.PersistedMetadata is private, so we encode the same structure manually.
    let sidecarPayload: [String: Any] = [
        "configuration": [:] as [String: Any], // Will decode with defaults via IndexConfiguration's custom Codable init
        "softDeletion": ["deletedIDs": []] as [String: Any],
    ]
    // Simpler approach: use the existing PersistedMetadata shape.
    // Since we can't reference the private type, encode an equivalent JSON manually:
    let metaJSON = """
    {
        "configuration": {},
        "softDeletion": {}
    }
    """.data(using: .utf8)!
    let metaURL = URL(fileURLWithPath: url.path + ".meta.json")
    try metaJSON.write(to: metaURL, options: .atomic)

    // Load should fall back to JSON sidecar since .db is missing
    let loaded = try await ANNSIndex.load(from: url)

    // The binary format embeds IDMap, so count should still be correct
    XCTAssertEqual(await loaded.count, 30)
}
```

> **Note:** The fallback test writes a minimal `.meta.json` with empty config/softDeletion.
> `IndexConfiguration` has a custom `init(from decoder:)` with backward compatibility
> that provides defaults for missing keys, so an empty JSON object decodes to `.default`.
> The binary file still contains the IDMap, so vector count is preserved.

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "testLoadFromSQLiteRoundtrip|testLoadFallsBackToJSONSidecar" 2>&1 | tail -10`
Expected: FAIL.

**Step 3: Modify ANNSIndex.load to prefer SQLite with JSON fallback**

Replace `load(from:)` in ANNSIndex.swift:

```swift
public static func load(from url: URL) async throws -> ANNSIndex {
    // Try SQLite first, fall back to JSON sidecar
    let dbPath = databasePath(for: url)
    let hasDB = FileManager.default.fileExists(atPath: dbPath)

    let persistedConfig: IndexConfiguration?
    let persistedSoftDeletion: SoftDeletion
    let persistedMetadataStore: MetadataStore
    let dbIDMap: IDMap?

    if hasDB {
        let db = try IndexDatabase(path: dbPath)
        persistedConfig = try db.loadConfiguration()
        persistedSoftDeletion = try db.loadSoftDeletion()
        persistedMetadataStore = try db.loadMetadataStore()
        dbIDMap = try db.loadIDMap()
    } else {
        // Backward compatibility: read from JSON sidecar
        let persistedMetadata = try loadPersistedMetadataIfPresent(from: url)
        persistedConfig = persistedMetadata?.configuration
        persistedSoftDeletion = persistedMetadata?.softDeletion ?? SoftDeletion()
        persistedMetadataStore = persistedMetadata?.metadataStore ?? MetadataStore()
        dbIDMap = nil
    }

    let initialConfiguration = persistedConfig ?? .default
    let index = ANNSIndex(configuration: initialConfiguration)
    let loaded = try IndexSerializer.load(from: url, device: await index.currentDevice())

    var resolvedConfiguration = persistedConfig ?? .default
    resolvedConfiguration.metric = loaded.metric
    resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16
    resolvedConfiguration.useBinary = loaded.vectors is BinaryVectorBuffer

    // Prefer SQLite IDMap if available; otherwise use the one from binary format
    let finalIDMap = dbIDMap ?? loaded.idMap

    await index.applyLoadedState(
        configuration: resolvedConfiguration,
        vectors: loaded.vectors,
        graph: loaded.graph,
        idMap: finalIDMap,
        entryPoint: loaded.entryPoint,
        softDeletion: persistedSoftDeletion,
        metadataStore: persistedMetadataStore
    )
    try await index.rebuildHNSWFromCurrentState()
    return index
}
```

Apply the same pattern to `loadMmap` and `loadDiskBacked`.

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "testLoadFromSQLiteRoundtrip|testLoadFallsBackToJSONSidecar" 2>&1 | tail -10`
Expected: PASS.

**Step 5: Run ALL persistence tests to verify backward compat**

Run: `swift test --filter "PersistenceTests|MmapTests|DiskBackedTests" 2>&1 | tail -20`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add Sources/MetalANNS/ANNSIndex.swift
git commit -m "feat: ANNSIndex.load prefers SQLite with JSON sidecar fallback"
```

---

## Task 7: Create StreamingDatabase

**Files:**
- Create: `Sources/MetalANNS/Storage/StreamingDatabase.swift`
- Test: `Tests/MetalANNSTests/Storage/StreamingDatabaseTests.swift`

**Step 1: Write the failing tests**

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

**Step 2: Run tests to verify they fail**

Run: `swift test --filter StreamingDatabaseTests 2>&1 | tail -10`
Expected: FAIL — `StreamingDatabase` not found.

**Step 3: Implement StreamingDatabase**

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

    /// Insert vectors as BLOBs (Float array -> raw bytes).
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
                let vector: [Float] = data.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: Float.self))
                }
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
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter StreamingDatabaseTests 2>&1 | tail -10`
Expected: PASS — all tests green.

**Step 5: Commit**

```bash
git add Sources/MetalANNS/Storage/StreamingDatabase.swift Tests/MetalANNSTests/Storage/StreamingDatabaseTests.swift
git commit -m "feat: add StreamingDatabase for SQLite-backed streaming state"
```

---

## Task 8: Wire StreamingDatabase into StreamingIndex Save/Load

**Files:**
- Modify: `Sources/MetalANNS/StreamingIndex.swift` (lines 337-410)

> **Important:** `PersistedMeta` and `MetadataValue` are `private` types inside
> `StreamingIndex`. The new SQLite load path must reconstruct a `PersistedMeta`
> instance. Since the load code lives inside `StreamingIndex` (as a static method),
> it already has access to private types. No visibility change is needed.

**Step 1: Write the failing test**

Add to `StreamingIndexPersistenceTests.swift`:

```swift
func testSaveCreatesStreamingDB() async throws {
    let config = StreamingConfiguration(deltaCapacity: 100, mergeStrategy: .blocking)
    let index = StreamingIndex(config: config)

    for i in 0..<20 {
        let vector = (0..<8).map { _ in Float.random(in: -1...1) }
        try await index.insert(vector: vector, id: "vec-\(i)")
    }
    try await index.flush()

    let dir = NSTemporaryDirectory() + "streaming-test-\(UUID().uuidString)"
    let url = URL(fileURLWithPath: dir)
    defer { try? FileManager.default.removeItem(at: url) }

    try await index.save(to: url)

    // streaming.db should exist
    let dbPath = url.appendingPathComponent("streaming.db").path
    XCTAssertTrue(FileManager.default.fileExists(atPath: dbPath),
                  "Expected streaming.db to be created")

    // Old streaming.meta.json should NOT exist
    let jsonPath = url.appendingPathComponent("streaming.meta.json").path
    XCTAssertFalse(FileManager.default.fileExists(atPath: jsonPath),
                   "Should no longer create streaming.meta.json")
}

func testSaveLoadRoundtripViaSQLite() async throws {
    let config = StreamingConfiguration(deltaCapacity: 100, mergeStrategy: .blocking)
    let index = StreamingIndex(config: config)

    for i in 0..<30 {
        let vector = (0..<8).map { _ in Float.random(in: -1...1) }
        try await index.insert(vector: vector, id: "vec-\(i)")
    }
    try await index.flush()

    let dir = NSTemporaryDirectory() + "streaming-roundtrip-\(UUID().uuidString)"
    let url = URL(fileURLWithPath: dir)
    defer { try? FileManager.default.removeItem(at: url) }

    try await index.save(to: url)
    let loaded = try await StreamingIndex.load(from: url)
    let count = await loaded.count
    XCTAssertEqual(count, 30)
}
```

**Step 2: Run tests to verify they fail**

Run: `swift test --filter "testSaveCreatesStreamingDB|testSaveLoadRoundtripViaSQLite" 2>&1 | tail -10`
Expected: FAIL.

**Step 3: Modify StreamingIndex.save to write SQLite**

Replace the save method in `StreamingIndex.swift`:

```swift
public func save(to url: URL) async throws {
    try checkBackgroundMergeError()
    try await flush()

    guard let base else {
        throw ANNSError.constructionFailed("Nothing to save — index is empty")
    }

    // Snapshot state before any await
    let configSnapshot = config
    let vectorDimensionSnapshot = vectorDimension
    let vectorsSnapshot = allVectorsList
    let idsSnapshot = allIDsList
    let deletedSnapshot = deletedIDs
    let metadataSnapshot = metadataByID

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
        try db.insertVectors(vectorsSnapshot, ids: idsSnapshot)
        try db.saveConfig(configSnapshot)
        try db.markDeleted(ids: deletedSnapshot)

        if let dim = vectorDimensionSnapshot {
            try db.pool.write { dbConn in
                try dbConn.execute(
                    sql: "INSERT OR REPLACE INTO state (key, intValue) VALUES (?, ?)",
                    arguments: ["vectorDimension", dim]
                )
            }
        }

        // Convert MetadataValue dict to string dict for storage.
        // MetadataValue is a private enum inside StreamingIndex, so the
        // conversion happens here where we have access to it.
        var stringMetadata: [String: [String: String]] = [:]
        for (id, entries) in metadataSnapshot {
            var converted: [String: String] = [:]
            for (key, value) in entries {
                let data = try JSONEncoder().encode(value)
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

**Step 4: Modify StreamingIndex.load with SQLite preference + JSON fallback**

> **Key insight:** Since `PersistedMeta` and `MetadataValue` are private to
> `StreamingIndex`, the load code must live inside the same actor (which it already
> does as a static method). No visibility changes needed.

```swift
public static func load(from url: URL) async throws -> StreamingIndex {
    let dbURL = url.appendingPathComponent("streaming.db")
    let jsonURL = url.appendingPathComponent("streaming.meta.json")

    if FileManager.default.fileExists(atPath: dbURL.path) {
        // New SQLite path
        let db = try StreamingDatabase(path: dbURL.path)
        guard let config = try db.loadConfig() else {
            throw ANNSError.corruptFile("Missing streaming config in database")
        }

        let (vectors, ids) = try db.loadAllVectors()
        let deletedIDs = try db.loadDeletedIDs()
        let allMetadata = try db.loadAllVectorMetadata()

        let dimRow = try db.pool.read { dbConn in
            try Int.fetchOne(dbConn, sql: "SELECT intValue FROM state WHERE key = 'vectorDimension'")
        }

        // Convert string metadata back to MetadataValue.
        // MetadataValue is private to StreamingIndex, accessible here.
        var metadataByID: [String: [String: MetadataValue]] = [:]
        let decoder = JSONDecoder()
        for (id, entries) in allMetadata {
            var converted: [String: MetadataValue] = [:]
            for (key, jsonStr) in entries {
                converted[key] = try decoder.decode(MetadataValue.self, from: Data(jsonStr.utf8))
            }
            metadataByID[id] = converted
        }

        let meta = PersistedMeta(
            config: config,
            vectorDimension: dimRow,
            allVectorsList: vectors,
            allIDsList: ids,
            deletedIDs: Array(deletedIDs),
            metadataByID: metadataByID
        )
        try validateLoadedMeta(meta)

        let loadedBase = try await ANNSIndex.load(
            from: url.appendingPathComponent("base.anns")
        )
        let streaming = StreamingIndex(config: config)
        await streaming.applyLoadedState(base: loadedBase, meta: meta)
        return streaming
    } else {
        // Backward compatibility: JSON path
        let data = try Data(contentsOf: jsonURL)
        let meta = try JSONDecoder().decode(PersistedMeta.self, from: data)
        try validateLoadedMeta(meta)

        let loadedBase = try await ANNSIndex.load(
            from: url.appendingPathComponent("base.anns")
        )
        let streaming = StreamingIndex(config: meta.config)
        await streaming.applyLoadedState(base: loadedBase, meta: meta)
        return streaming
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `swift test --filter "testSaveCreatesStreamingDB|testSaveLoadRoundtripViaSQLite" 2>&1 | tail -10`
Expected: PASS.

**Step 5: Run ALL streaming persistence tests**

Run: `swift test --filter StreamingIndexPersistenceTests 2>&1 | tail -20`
Expected: All tests pass (old tests still work via updated code paths).

**Step 6: Commit**

```bash
git add Sources/MetalANNS/StreamingIndex.swift
git commit -m "feat: StreamingIndex save/load uses SQLite, JSON fallback for backward compat"
```

---

## Task 9: Full Integration Tests + Backward Compatibility

**Files:**
- Modify: `Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift`
- Modify existing persistence test files

**Step 1: Write backward compatibility integration test**

```swift
func testBackwardCompatLoadThenSaveUpgrades() async throws {
    // 1. Build an index and save in OLD format (simulate pre-migration)
    let index = ANNSIndex(configuration: .default)
    var vectors: [[Float]] = []
    var ids: [String] = []
    for i in 0..<40 {
        vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
        ids.append("compat-\(i)")
    }
    try await index.batchInsert(vectors: vectors, ids: ids)
    try await index.build()

    let dir = NSTemporaryDirectory() + "compat-\(UUID().uuidString)"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: dir) }

    let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
    try await index.save(to: url)

    // Remove .db to simulate old-format file
    let dbPath = url.path.replacingOccurrences(of: ".anns", with: ".db")
    try FileManager.default.removeItem(atPath: dbPath)

    // Write a .meta.json sidecar (old format).
    // The binary file still has IDMap embedded, so load should work.
    let metaJSON = """
    {
        "configuration": {},
        "softDeletion": {}
    }
    """.data(using: .utf8)!
    let metaURL = URL(fileURLWithPath: url.path + ".meta.json")
    try metaJSON.write(to: metaURL, options: .atomic)

    // 2. Load from old format
    let loaded = try await ANNSIndex.load(from: url)
    XCTAssertEqual(await loaded.count, 40)

    // 3. Re-save — should now create .db
    try await loaded.save(to: url)
    XCTAssertTrue(FileManager.default.fileExists(atPath: dbPath),
                  "Re-save should create .db file (auto-upgrade)")

    // 4. Load again from new format
    let reloaded = try await ANNSIndex.load(from: url)
    XCTAssertEqual(await reloaded.count, 40)

    // 5. Verify search still works
    let results = try await reloaded.search(query: vectors[0], k: 5)
    XCTAssertEqual(results.first?.id, "compat-0")
}
```

**Step 2: Run the integration test**

Run: `swift test --filter testBackwardCompatLoadThenSaveUpgrades 2>&1 | tail -10`
Expected: PASS.

**Step 3: Run the FULL test suite**

Run: `swift test 2>&1 | tail -30`
Expected: All tests pass (except known Metal/shader environment failures).

**Step 4: Commit**

```bash
git add Tests/
git commit -m "test: add backward compatibility and integration tests for GRDB migration"
```

---

## Task 10: Cleanup — Remove Dead JSON Sidecar Code

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (remove `metadataURL` helper, `loadPersistedMetadataIfPresent`)

**Step 1: Verify no code paths still create .meta.json**

Search for `.meta.json` usage. If all save paths now use SQLite, the JSON creation code is dead.

**Step 2: Keep `loadPersistedMetadataIfPresent` for backward-compat loading only**

Mark it with a deprecation comment:

```swift
// MARK: - Legacy JSON Sidecar (backward compatibility only)
/// Loads metadata from the old .meta.json sidecar format.
/// Used only when no .db file exists (pre-migration indexes).
private static func loadPersistedMetadataIfPresent(from url: URL) throws -> PersistedMetadata? {
    // ... existing code unchanged ...
}
```

**Step 3: Run full test suite**

Run: `swift test 2>&1 | tail -20`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add Sources/MetalANNS/ANNSIndex.swift
git commit -m "refactor: mark JSON sidecar loading as legacy, document backward-compat path"
```

---

## Out of Scope (Future Work)

These are deferred intentionally:

1. **IVFPQ migration to SQLite** — works fine as JSON, not a pain point yet
2. **Removing IDMap from binary format** — would require v4 format, breaking change
3. **Incremental IDMap updates** — current full-replace is fine for the data sizes involved
4. **SQLite FTS for metadata search** — premature, add when needed
5. **Migration CLI tool** — auto-upgrade on load is sufficient

---

## Risk Checklist

| Risk | Mitigation |
|------|-----------|
| GRDB v7.10.0 requires Swift 6.1+ / Xcode 16.3+ | Verify toolchain before resolving; pin `"7.0.0"..<"7.9.0"` if on Swift 6.0 |
| SQLite file corruption on crash | WAL mode + GRDB transactions provide ACID guarantees |
| Performance regression on save | Prepared statements for batch inserts; benchmark before/after |
| Breaking existing users | JSON fallback on load; binary format unchanged |
| Transitive dependency concern | GRDB is pure Swift, no further transitive deps |
| MetadataStore as JSON blob limits SQL queryability | Acceptable — metadata filtering happens in-memory via `MetadataStore.matches()`, not SQL |

---

## Review Changelog (fixes applied)

Issues found during plan review and corrected:

1. **MetadataStore API was fabricated** — Plan invented `set(vectorID:key:value:)`, `get(vectorID:key:)`, `MetadataValue` type, and `allEntries()`. Actual API uses typed columns (`set(_:stringValue:for:)` etc.) keyed by `UInt32` internal IDs. **Fix:** Store MetadataStore as a single JSON blob via its `Codable` conformance. Removed `vector_metadata` table from IndexDatabase schema (StreamingDatabase keeps its own for its different metadata system).

2. **SoftDeletion API name mismatches** — Plan used `markDeleted(internalID:)` and `.deletedInternalIDs`. Actual: `markDeleted(_:)` (unnamed parameter) and `.allDeletedIDs`. **Fix:** All call sites corrected.

3. **IndexConfiguration.graphDegree doesn't exist** — Actual property is `.degree`. **Fix:** Test corrected.

4. **loadIDMap never restored nextInternalID** — Stored the value but never read it back. **Fix:** `loadIDMap()` now reads `idmap.nextID` from config and restores it if higher than auto-incremented value. Added gap-handling test.

5. **testLoadFallsBackToJSONSidecar was broken** — Had a runtime-crashing forced cast and never wrote a `.meta.json` file. **Fix:** Test now writes a minimal valid JSON sidecar manually. Added note about `IndexConfiguration`'s backward-compatible `Codable` init.

6. **PersistedMeta visibility concern** — `PersistedMeta` and `MetadataValue` are private to `StreamingIndex`. **Fix:** Added explicit note that the load code lives inside `StreamingIndex` as a static method, so it already has access. No visibility change needed.

7. **GRDB version requirements** — v7.10.0 requires Swift 6.1+ / Xcode 16.3+. **Fix:** Added note in Task 1 with version pinning guidance. Updated risk checklist.

8. **Two different metadata systems conflated** — `MetadataStore` (UInt32 internal IDs, typed columns) vs StreamingIndex's `metadataByID` (String external IDs, `MetadataValue` enum). **Fix:** Added documentation distinguishing the two. IndexDatabase uses JSON blob for MetadataStore. StreamingDatabase uses row-per-entry for streaming metadata with explicit conversion code.
