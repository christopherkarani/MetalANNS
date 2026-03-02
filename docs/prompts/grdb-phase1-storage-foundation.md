# GRDB Phase 1: Storage Foundation

### Mission

Introduce GRDB/SQLite as the structured-data persistence layer for MetalANNS.
This phase is **storage-only** — no changes to `ANNSIndex`, `StreamingIndex`, or any
search/insert logic. The goal is a clean, tested `IndexDatabase` type that can
save and load `IDMap`, `IndexConfiguration`, `SoftDeletion`, and `MetadataStore`
via a SQLite database file (`index.db`).

**Track every completed step in `tasks/grdb-phase1-todo.md`.**
Mark each checkbox `[x]` immediately after the step passes verification.

---

### Constraints (read before writing a single line of code)

1. **TDD is mandatory.** Write the failing test first, confirm it fails, implement,
   confirm it passes. Never write implementation code before the test.
2. **Swift 6 strict concurrency.** All new types must be `Sendable`. No `nonisolated(unsafe)`.
3. **`MetalANNSCore` stays dependency-free.** `IndexDatabase` lives in `Sources/MetalANNS/`,
   not `Sources/MetalANNSCore/`. The only Core change is adding `IDMap.makeForPersistence`.
4. **Do not change any existing public API.** `IDMap`, `IndexConfiguration`, `SoftDeletion`,
   and `MetadataStore` keep their current interfaces.
5. **Commit after every task.** Use the exact commit messages listed below.
6. **Do not touch** `ANNSIndex.swift`, `StreamingIndex.swift`, or any Metal shader file.

---

### Verified Codebase Facts

Read each file before touching it.

| Fact | Source |
|------|--------|
| `IDMap` is a `public struct` in `Sources/MetalANNSCore/IDMap.swift:4` | `IDMap.swift:4` |
| `IDMap.assign(externalID:)` auto-increments from `nextID` starting at 0 | `IDMap.swift:27` |
| `IDMap.nextInternalID` returns the next-to-be-assigned UInt32 | `IDMap.swift:12` |
| `IDMap.externalID(for:)` / `internalID(for:)` are O(1) dict lookups | `IDMap.swift:42,46` |
| `IDMap` does NOT have a `makeForPersistence` factory yet — you must add it | `IDMap.swift` |
| `SoftDeletion` is a `public struct` in `Sources/MetalANNSCore/SoftDeletion.swift:3` | `SoftDeletion.swift:3` |
| `SoftDeletion.markDeleted(_:)` inserts into an internal `Set<UInt32>` | `SoftDeletion.swift:8` |
| `SoftDeletion.allDeletedIDs` exposes the full set for persistence | `SoftDeletion.swift:25` |
| `MetadataStore` is a `public struct` in `Sources/MetalANNSCore/MetadataStore.swift:3` | `MetadataStore.swift:3` |
| `MetadataStore` has three internal dicts: `stringColumns`, `floatColumns`, `intColumns` | `MetadataStore.swift:4-6` |
| `MetadataStore` is `Codable` — serialize as a single JSON blob in the `config` table | `MetadataStore.swift:3` |
| `IndexConfiguration` is in `Sources/MetalANNS/IndexConfiguration.swift:3` | `IndexConfiguration.swift:3` |
| `IndexConfiguration` is `Codable` — serialize as a single JSON blob | `IndexConfiguration.swift:3` |
| `Package.swift` currently has NO external dependencies | `Package.swift` |
| `Sources/MetalANNS/Storage/` directory does not exist yet — create it | directory structure |
| `Tests/MetalANNSTests/Storage/` directory does not exist yet — create it | directory structure |
| GRDB v7.10.0 requires Swift 6.1+ / Xcode 16.3+; verify with `swift --version` first | GRDB docs |
| If on Swift 6.0, pin to `"7.0.0"..<"7.9.0"` instead of `from: "7.0.0"` | GRDB docs |

---

### TDD Implementation Order

Work strictly in this order. Never jump ahead.

**Round 1 — GRDB Dependency**
Add GRDB to `Package.swift`. Resolve and build. No tests yet.

**Round 2 — IndexDatabase Foundation**
Write two failing tests: database creation + table verification. Implement `IndexDatabase`
with WAL config and `v1-foundation` migration. Tests pass.

**Round 3 — IDMap Persistence**
Write three failing tests for IDMap save/load. Add `IDMap.makeForPersistence` to Core.
Add `saveIDMap`/`loadIDMap` to `IndexDatabase`. Tests pass.

**Round 4 — Metadata Persistence**
Write four failing tests for config, soft-deletion, and metadata. Implement three
extension pairs on `IndexDatabase`. Tests pass. Full suite clean.

Run after every round:
```
swift test --filter IndexDatabaseTests 2>&1 | tail -15
```

---

### Step-by-Step Instructions

---

#### Task 1 — Add GRDB Dependency

**File**: `Package.swift`

**Step 1.1: Check Swift version**

```bash
swift --version
```

If Swift 6.1+, use `from: "7.0.0"`. If Swift 6.0, use `"7.0.0"..<"7.9.0"`.

**Step 1.2: Update Package.swift**

Replace the `dependencies` and affected `targets` sections:

```swift
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
            dependencies: [
                "MetalANNS",
                "MetalANNSCore",
                "MetalANNSBenchmarks",
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
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

**Step 1.3: Resolve and build**

```bash
swift package resolve
swift build 2>&1 | tail -5
```

Expected: GRDB resolves; build succeeds (Metal shader warnings are OK — CLI can't compile .metal).

**Step 1.4: Commit + update todo**

```bash
git add Package.swift Package.resolved
git commit -m "deps: add GRDB.swift for SQLite-backed persistence"
```

Mark Task 1 checkboxes in `tasks/grdb-phase1-todo.md`.

---

#### Task 2 — IndexDatabase Foundation

**Files:**
- Create: `Sources/MetalANNS/Storage/IndexDatabase.swift`
- Create: `Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift`

**Step 2.1: Write the failing tests**

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
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))

        let tables = try db.pool.read { db in
            try String.fetchAll(db, sql: """
                SELECT name FROM sqlite_master
                WHERE type='table'
                  AND name NOT LIKE 'sqlite_%'
                  AND name != 'grdb_migrations'
                ORDER BY name
                """)
        }
        XCTAssertEqual(tables, ["config", "idmap", "soft_deletion"])
    }

    func testReopenExistingDatabase() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        _ = try IndexDatabase(path: path)

        let db = try IndexDatabase(path: path)
        let count = try db.pool.read { db in
            try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM idmap")
        }
        XCTAssertEqual(count, 0)
    }
}
```

**Step 2.2: Confirm tests fail**

```bash
swift test --filter IndexDatabaseTests 2>&1 | tail -10
```

Expected: FAIL — `IndexDatabase` type not found.

**Step 2.3: Implement IndexDatabase**

```swift
// Sources/MetalANNS/Storage/IndexDatabase.swift
import Foundation
import GRDB

/// SQLite-backed persistence layer for MetalANNS structured data.
///
/// Uses GRDB DatabasePool in WAL mode. Manages IDMap, configuration,
/// soft-deletion, and MetadataStore.
///
/// MetadataStore is persisted as a single JSON blob in the `config` table
/// (not row-per-entry) because its structure uses UInt32 internal IDs with
/// three typed column dictionaries — not suitable for normalized rows.
public final class IndexDatabase: Sendable {
    let pool: DatabasePool

    public init(path: String) throws {
        var config = Configuration()
        config.prepareDatabase { db in
            try db.execute(sql: "PRAGMA mmap_size = 268435456")
        }
        pool = try DatabasePool(path: path, configuration: config)
        try migrate()
    }

    // MARK: - Schema

    private func migrate() throws {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1-foundation") { db in
            try db.create(table: "idmap") { t in
                t.column("externalID", .text).notNull().primaryKey()
                t.column("internalID", .integer).notNull().unique()
            }
            try db.create(index: "idmap_by_internal", on: "idmap", columns: ["internalID"])

            try db.create(table: "config") { t in
                t.column("key", .text).notNull().primaryKey()
                t.column("value", .text).notNull()
            }

            try db.create(table: "soft_deletion") { t in
                t.column("internalID", .integer).notNull().primaryKey()
            }
        }

        try migrator.migrate(pool)
    }

    /// Flush WAL so the .db file can be safely moved or backed up.
    public func prepareForFileMove() throws {
        try pool.writeWithoutTransaction { db in
            try db.execute(sql: "PRAGMA wal_checkpoint(TRUNCATE)")
        }
    }
}
```

**Step 2.4: Verify tests pass**

```bash
swift test --filter IndexDatabaseTests 2>&1 | tail -10
```

Expected: PASS — both tests green.

**Step 2.5: Commit + update todo**

```bash
git add Sources/MetalANNS/Storage/IndexDatabase.swift \
        Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift
git commit -m "feat: add IndexDatabase foundation with schema v1"
```

---

#### Task 3 — IDMap Read/Write

**Files:**
- Modify: `Sources/MetalANNSCore/IDMap.swift`
- Modify: `Sources/MetalANNS/Storage/IndexDatabase.swift`
- Modify: `Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift`

**Step 3.1: Write the failing tests**

Append to `IndexDatabaseTests.swift` (inside the class body):

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
```

**Step 3.2: Confirm tests fail**

```bash
swift test --filter "testSaveAndLoadIDMap|testSaveIDMapOverwritesPrevious|testLoadIDMapFromEmptyDatabase" 2>&1 | tail -10
```

Expected: FAIL — `saveIDMap`/`loadIDMap` not found.

**Step 3.3: Add IDMap persistence factory to IDMap.swift**

Add this internal factory at the bottom of `IDMap.swift`:

```swift
// MARK: - Persistence Reconstruction (internal — used only by IndexDatabase)
extension IDMap {
    /// Rebuilds an IDMap from persisted rows without calling `assign()`.
    /// Preserves exact internal IDs and the saved nextID value
    /// (which may be higher than rows.count if gaps existed).
    static func makeForPersistence(rows: [(String, UInt32)], nextID: UInt32) -> IDMap {
        var map = IDMap()
        for (externalID, internalID) in rows {
            map.externalToInternal[externalID] = internalID
            map.internalToExternal[internalID] = externalID
        }
        map.nextID = nextID
        return map
    }
}
```

> **Note:** `externalToInternal` and `internalToExternal` are `private` in IDMap —
> change them to `fileprivate` to allow access from the same file, or move this
> extension into `IDMap.swift` itself (preferred since it's the same module).

**Step 3.4: Add IDMap persistence to IndexDatabase.swift**

Add a new extension at the bottom of `IndexDatabase.swift`:

```swift
// MARK: - IDMap Persistence
import MetalANNSCore

extension IndexDatabase {

    /// Atomically replaces the full IDMap in the database.
    public func saveIDMap(_ idMap: IDMap) throws {
        try pool.write { db in
            try db.execute(sql: "DELETE FROM idmap")

            let stmt = try db.makeStatement(sql: """
                INSERT INTO idmap (externalID, internalID) VALUES (?, ?)
                """)

            for internalID: UInt32 in 0 ..< idMap.nextInternalID {
                guard let externalID = idMap.externalID(for: internalID) else { continue }
                try stmt.execute(arguments: [externalID, Int64(internalID)])
            }

            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["idmap.nextID", String(idMap.nextInternalID)]
            )
        }
    }

    /// Loads the full IDMap from the database. Returns an empty IDMap if
    /// the table is empty.
    public func loadIDMap() throws -> IDMap {
        try pool.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: "SELECT externalID, internalID FROM idmap ORDER BY internalID"
            )

            let pairs: [(String, UInt32)] = rows.map { row in
                let ext: String = row["externalID"]
                let int64: Int64 = row["internalID"]
                return (ext, UInt32(int64))
            }

            let nextIDString = try String.fetchOne(
                db,
                sql: "SELECT value FROM config WHERE key = ?",
                arguments: ["idmap.nextID"]
            )
            let nextID = nextIDString.flatMap(UInt32.init) ?? UInt32(pairs.count)

            return IDMap.makeForPersistence(rows: pairs, nextID: nextID)
        }
    }
}
```

**Step 3.5: Verify tests pass**

```bash
swift test --filter "testSaveAndLoadIDMap|testSaveIDMapOverwritesPrevious|testLoadIDMapFromEmptyDatabase" 2>&1 | tail -10
```

Expected: PASS — all three green.

**Step 3.6: Commit + update todo**

```bash
git add Sources/MetalANNSCore/IDMap.swift \
        Sources/MetalANNS/Storage/IndexDatabase.swift \
        Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift
git commit -m "feat: add IDMap save/load to IndexDatabase"
```

---

#### Task 4 — Metadata Persistence

**Files:**
- Modify: `Sources/MetalANNS/Storage/IndexDatabase.swift`
- Modify: `Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift`

**Step 4.1: Write the failing tests**

Append to `IndexDatabaseTests.swift`:

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
    XCTAssertEqual(loaded.getFloat("score", for: 0), 0.95, accuracy: 1e-6)
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

**Step 4.2: Confirm tests fail**

```bash
swift test --filter "testSaveAndLoadConfig|testSaveAndLoadSoftDeletion|testSaveAndLoadMetadataStore|testSaveAndLoadEmptyMetadataStore" 2>&1 | tail -10
```

Expected: FAIL — methods not found.

**Step 4.3: Implement metadata persistence extensions**

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
            ), let data = json.data(using: .utf8) else {
                return nil
            }
            return try JSONDecoder().decode(IndexConfiguration.self, from: data)
        }
    }
}

// MARK: - SoftDeletion Persistence
extension IndexDatabase {

    public func saveSoftDeletion(_ softDeletion: SoftDeletion) throws {
        try pool.write { db in
            try db.execute(sql: "DELETE FROM soft_deletion")
            for internalID in softDeletion.allDeletedIDs {
                try db.execute(
                    sql: "INSERT INTO soft_deletion (internalID) VALUES (?)",
                    arguments: [Int64(internalID)]
                )
            }
        }
    }

    public func loadSoftDeletion() throws -> SoftDeletion {
        try pool.read { db in
            let ids = try Int64.fetchAll(
                db,
                sql: "SELECT internalID FROM soft_deletion"
            )
            var softDeletion = SoftDeletion()
            for id in ids {
                softDeletion.markDeleted(UInt32(id))
            }
            return softDeletion
        }
    }
}

// MARK: - MetadataStore Persistence
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
            ), let data = json.data(using: .utf8) else {
                return MetadataStore()
            }
            return try JSONDecoder().decode(MetadataStore.self, from: data)
        }
    }
}
```

**Step 4.4: Verify new tests pass**

```bash
swift test --filter "testSaveAndLoadConfig|testSaveAndLoadSoftDeletion|testSaveAndLoadMetadataStore|testSaveAndLoadEmptyMetadataStore" 2>&1 | tail -10
```

Expected: PASS — all four green.

**Step 4.5: Run full test suite (zero regressions)**

```bash
swift test 2>&1 | grep -E "passed|failed|error:" | tail -5
```

Expected: All previously passing tests still pass. Zero new failures.

**Step 4.6: Commit + update todo**

```bash
git add Sources/MetalANNS/Storage/IndexDatabase.swift \
        Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift
git commit -m "feat: add config, soft-deletion, and metadata persistence to IndexDatabase"
```

---

### Definition of Done

Phase 1 is complete when ALL of the following are true:

- [ ] `swift test --filter IndexDatabaseTests` → all tests PASS
- [ ] `swift test` → zero regressions (same pass/fail as before Phase 1)
- [ ] `swift build` → succeeds (Metal shader warnings OK)
- [ ] `tasks/grdb-phase1-todo.md` → all checkboxes marked `[x]`
- [ ] Four commits in git log with the exact messages above

---

### Anti-patterns (do NOT do these)

| Anti-pattern | Why |
|---|---|
| Calling `idMap.assign()` in `loadIDMap()` | Breaks gap preservation — always use `makeForPersistence` |
| Storing MetadataStore as row-per-entry | Internal structure is UInt32-keyed dicts, not relatable as SQL rows |
| Adding `async` to `saveIDMap` / `loadIDMap` | Storage layer is synchronous; callers manage async context |
| Modifying `ANNSIndex.swift` or `StreamingIndex.swift` | Out of scope for Phase 1 |
| Using `nonisolated(unsafe)` to quiet Swift 6 warnings | Indicates a real concurrency problem — fix it properly |
| Skipping `prepareForFileMove()` before any file copy | WAL sidecars will be left in inconsistent state |

---

### What Comes Next (Phase 2)

Phase 2 wires `IndexDatabase` into `ANNSIndex.save()` and `ANNSIndex.load()`,
replacing the JSON sidecar with `index.db`. That prompt is in
`docs/prompts/grdb-phase2-annsindex-integration.md` (to be written after Phase 1 review).
