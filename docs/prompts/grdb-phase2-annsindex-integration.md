# GRDB Phase 2: ANNSIndex Integration

### Mission

Wire `IndexDatabase` into `ANNSIndex.save()` and all three `ANNSIndex.load*()` variants,
replacing the legacy `SQLiteStructuredStore` (raw SQLite3 KV blob) and JSON `.meta.json`
sidecar with the GRDB-backed `.db` file from Phase 1.

After this phase:
- `save()` writes `index.anns` (binary) + `index.db` (SQLite). No more `.meta.json`.
- `load()` reads `index.db` first (if fresh). Falls back to `.meta.db` → `.meta.json` for old files.
- `saveMmapCompatible()`, `loadMmap()`, `loadDiskBacked()` follow the same pattern.
- `SQLiteStructuredStore` remains in the codebase (fallback read path) — removal is Phase 4.

**Track every completed step in `tasks/grdb-phase2-todo.md`.**
Mark each checkbox `[x]` immediately after the step passes verification.

---

### Constraints (read before writing a single line of code)

1. **TDD is mandatory.** Write failing tests first, confirm failure, then implement.
2. **Swift 6 strict concurrency.** All new static helpers must be `nonisolated`. No `nonisolated(unsafe)`.
3. **Do NOT delete `SQLiteStructuredStore.swift`** — it's still needed for fallback reads of old `.meta.db` files.
4. **Do NOT delete `PersistedMetadata`** — it's still needed for JSON decoding of old `.meta.json` files.
5. **Do NOT change `IndexDatabase.swift`** — Phase 1's storage layer is frozen. If you need new methods, stop and flag it.
6. **Do NOT touch** any Metal shader, `MetalANNSCore` types, or `StreamingIndex.swift`.
7. **Atomic saves** — never write directly to the final `.db` path. Always use a temp directory, checkpoint WAL, then move.
8. **Commit after each task.** Use exact commit messages from the todo.

---

### Verified Codebase Facts

Read each file before touching it.

| Fact | Source |
|------|--------|
| `ANNSIndex` is a `public actor` | `ANNSIndex.swift:5` |
| `PersistedMetadata` is a private `Codable` struct with `configuration`, `softDeletion`, `metadataStore?`, `idMap?` | `ANNSIndex.swift:8-12` |
| Current `save(to:)` writes: binary via `IndexSerializer.save` → JSON via `JSONEncoder` → SQLite via `SQLiteStructuredStore.save` | `ANNSIndex.swift:835-858` |
| Current `saveMmapCompatible(to:)` follows the same triple-write pattern | `ANNSIndex.swift:860-883` |
| Current `load(from:)` calls `loadPersistedMetadataIfPresent` then `IndexSerializer.load` | `ANNSIndex.swift:885-910` |
| Current `loadMmap(from:)` follows the same pattern with `MmapIndexLoader.load` | `ANNSIndex.swift:912-938` |
| Current `loadDiskBacked(from:)` follows the same pattern with `DiskBackedIndexLoader.load` | `ANNSIndex.swift:940-967` |
| `loadPersistedMetadataIfPresent` tries SQLiteStructuredStore (`.meta.db`) first, then JSON (`.meta.json`) | `ANNSIndex.swift:1065-1077` |
| `metadataURL(for:)` returns `fileURL.path + ".meta.json"` | `ANNSIndex.swift:1050-1052` |
| `metadataDBURL(for:)` returns `fileURL.path + ".meta.db"` | `ANNSIndex.swift:1054-1056` |
| `resolveLoadedIDMap` prefers persisted IDMap if count matches serializer IDMap | `ANNSIndex.swift:1079-1090` |
| `applyLoadedState` accepts config, vectors, graph, idMap, entryPoint, softDeletion, metadataStore, isReadOnlyLoadedIndex, mmapLifetime | `ANNSIndex.swift:988-1011` |
| `SQLiteStructuredStore` is a raw SQLite3 KV blob store at `Storage/SQLiteStructuredStore.swift` — save/load `Codable` as JSON blob in `kv` table | `SQLiteStructuredStore.swift:5` |
| `IndexDatabase` (from Phase 1) has `saveIDMap`, `saveConfiguration`, `saveSoftDeletion`, `saveMetadataStore` + corresponding loads | `Storage/IndexDatabase.swift` |
| `IndexDatabase.prepareForFileMove()` does `PRAGMA wal_checkpoint(TRUNCATE)` | `Storage/IndexDatabase.swift` |
| Existing test `saveAndLoadLifecycle` checks for `.meta.db` file and removes `.meta.json` before load | `ANNSIndexTests.swift:64-103` |
| Existing test uses `.mann` extension (not `.anns`) — `databasePath` must handle any extension | `ANNSIndexTests.swift:81` |
| `PersistenceTests.swift` has `saveAndLoadRoundtrip`, `corruptMagicThrows`, `corruptVersionThrows` | `PersistenceTests.swift:7,86,111` |

---

### TDD Implementation Order

Work strictly in this order. Never jump ahead.

**Round 1 — Save Integration (Task 5)**
Write failing test that asserts `.db` exists after save and `.meta.json` does NOT.
Rewrite `save()` and `saveMmapCompatible()` to use `IndexDatabase` + temp-dir atomic writes.
Update the existing `saveAndLoadLifecycle` test for new `.db` path.

**Round 2 — Load Integration (Task 6)**
Write three failing tests: SQLite roundtrip, JSON fallback, corrupt-DB fallback.
Rewrite `load()`, `loadMmap()`, `loadDiskBacked()` to prefer `IndexDatabase` with staleness check.
Run full regression suite.

Run after every step:
```
swift test --filter ANNSIndexTests 2>&1 | tail -15
```

---

### Step-by-Step Instructions

---

#### Task 5 — Wire IndexDatabase into ANNSIndex Save

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift`
- Modify: `Tests/MetalANNSTests/ANNSIndexTests.swift`

**Step 5.1: Write the failing test**

Add to `ANNSIndexTests.swift`:

```swift
@Test("Save creates SQLite sidecar")
func saveCreatesDBFile() async throws {
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
    #expect(FileManager.default.fileExists(atPath: url.path))

    // SQLite file should exist at new .db path
    let dbPath = URL(fileURLWithPath: url.deletingPathExtension().path)
        .appendingPathExtension("db").path
    #expect(FileManager.default.fileExists(atPath: dbPath))

    // Old meta.json should NOT be created
    #expect(!FileManager.default.fileExists(atPath: url.path + ".meta.json"))
}
```

**Step 5.2: Confirm test fails**

```bash
swift test --filter saveCreatesDBFile 2>&1 | tail -10
```

Expected: FAIL — `.meta.json` still created; `.db` not found (old path is `.meta.db`).

**Step 5.3: Add file helper methods to ANNSIndex**

Add these private static helpers. They replace `metadataURL(for:)` and `metadataDBURL(for:)`:

```swift
/// Returns the .db sidecar path for a given .anns file URL.
/// Example: /path/to/index.anns → /path/to/index.db
private nonisolated static func databasePath(for fileURL: URL) -> String {
    let base = fileURL.deletingPathExtension()
    return base.appendingPathExtension("db").path
}

/// Atomically replaces a file at `destination` with the file at `source`.
private nonisolated static func replaceFile(at destination: URL, with source: URL) throws {
    let fm = FileManager.default
    if fm.fileExists(atPath: destination.path) {
        _ = try fm.replaceItemAt(destination, withItemAt: source)
    } else {
        try fm.moveItem(at: source, to: destination)
    }
}

/// Atomically replaces a SQLite database and its WAL/SHM sidecars.
private nonisolated static func replaceSQLiteFiles(at destinationDB: URL, with sourceDB: URL) throws {
    let fm = FileManager.default

    func replace(_ source: URL, _ destination: URL) throws {
        guard fm.fileExists(atPath: source.path) else { return }
        if fm.fileExists(atPath: destination.path) {
            _ = try fm.replaceItemAt(destination, withItemAt: source)
        } else {
            try fm.moveItem(at: source, to: destination)
        }
    }

    func replaceSidecar(suffix: String) throws {
        let source = URL(fileURLWithPath: sourceDB.path + suffix)
        let destination = URL(fileURLWithPath: destinationDB.path + suffix)
        if fm.fileExists(atPath: source.path) {
            try replace(source, destination)
        } else if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
    }

    try replace(sourceDB, destinationDB)
    try replaceSidecar(suffix: "-wal")
    try replaceSidecar(suffix: "-shm")
}
```

**Step 5.4: Rewrite `save(to:)`**

Replace the current `save(to:)` method (lines 835-858) with:

```swift
public func save(to url: URL) async throws {
    guard isBuilt, let vectors, let graph else {
        throw ANNSError.indexEmpty
    }

    let fileManager = FileManager.default
    let parentURL = url.deletingLastPathComponent()
    let tempDirURL = parentURL.appendingPathComponent(".save-tmp-\(UUID().uuidString)")
    let tempANNS = tempDirURL.appendingPathComponent(url.lastPathComponent)
    let dbURL = URL(fileURLWithPath: Self.databasePath(for: url))
    let tempDB = tempDirURL.appendingPathComponent(dbURL.lastPathComponent)

    do {
        try fileManager.createDirectory(at: tempDirURL, withIntermediateDirectories: true)

        try IndexSerializer.save(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            entryPoint: entryPoint,
            metric: configuration.metric,
            to: tempANNS
        )

        var db: IndexDatabase? = try IndexDatabase(path: tempDB.path)
        try db?.saveIDMap(idMap)
        try db?.saveConfiguration(configuration)
        try db?.saveSoftDeletion(softDeletion)
        try db?.saveMetadataStore(metadataStore)
        try db?.prepareForFileMove()
        db = nil  // Close the database connection before moving files

        try Self.replaceFile(at: url, with: tempANNS)
        try Self.replaceSQLiteFiles(at: dbURL, with: tempDB)

        // Clean up temp dir (may still contain empty dir after moves)
        try? fileManager.removeItem(at: tempDirURL)
    } catch {
        try? fileManager.removeItem(at: tempDirURL)
        throw error
    }
}
```

**Step 5.5: Rewrite `saveMmapCompatible(to:)` with the same pattern**

Replace the current `saveMmapCompatible(to:)` method (lines 860-883) with:

```swift
public func saveMmapCompatible(to url: URL) async throws {
    guard isBuilt, let vectors, let graph else {
        throw ANNSError.indexEmpty
    }

    let fileManager = FileManager.default
    let parentURL = url.deletingLastPathComponent()
    let tempDirURL = parentURL.appendingPathComponent(".save-tmp-\(UUID().uuidString)")
    let tempANNS = tempDirURL.appendingPathComponent(url.lastPathComponent)
    let dbURL = URL(fileURLWithPath: Self.databasePath(for: url))
    let tempDB = tempDirURL.appendingPathComponent(dbURL.lastPathComponent)

    do {
        try fileManager.createDirectory(at: tempDirURL, withIntermediateDirectories: true)

        try IndexSerializer.saveMmapCompatible(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            entryPoint: entryPoint,
            metric: configuration.metric,
            to: tempANNS
        )

        var db: IndexDatabase? = try IndexDatabase(path: tempDB.path)
        try db?.saveIDMap(idMap)
        try db?.saveConfiguration(configuration)
        try db?.saveSoftDeletion(softDeletion)
        try db?.saveMetadataStore(metadataStore)
        try db?.prepareForFileMove()
        db = nil

        try Self.replaceFile(at: url, with: tempANNS)
        try Self.replaceSQLiteFiles(at: dbURL, with: tempDB)

        try? fileManager.removeItem(at: tempDirURL)
    } catch {
        try? fileManager.removeItem(at: tempDirURL)
        throw error
    }
}
```

**Step 5.6: Update existing `saveAndLoadLifecycle` test**

In `ANNSIndexTests.swift`, update the `saveAndLoadLifecycle` test:

Change:
```swift
let tempMetaDBURL = URL(fileURLWithPath: tempURL.path + ".meta.db")
```
To:
```swift
let tempDBURL = URL(fileURLWithPath:
    tempURL.deletingPathExtension().appendingPathExtension("db").path)
```

Update the defer block to clean up `.db` instead of `.meta.db`:
```swift
defer {
    try? FileManager.default.removeItem(at: tempURL)
    try? FileManager.default.removeItem(at: tempMetaURL)
    try? FileManager.default.removeItem(at: tempDBURL)
}
```

Update the assertion:
```swift
#expect(FileManager.default.fileExists(atPath: tempDBURL.path))
```

**Step 5.7: Verify all save tests pass**

```bash
swift test --filter "saveCreatesDBFile|saveAndLoadLifecycle" 2>&1 | tail -15
```

Expected: PASS — both tests green.

**Step 5.8: Commit + update todo**

```bash
git add Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/ANNSIndexTests.swift
git commit -m "feat: ANNSIndex.save writes SQLite alongside binary format"
```

---

#### Task 6 — Wire IndexDatabase into ANNSIndex Load (with Fallback)

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift`
- Modify: `Tests/MetalANNSTests/ANNSIndexTests.swift`

**Step 6.1: Write the failing tests**

Add all three tests to `ANNSIndexTests.swift`:

```swift
@Test("Load roundtrip prefers SQLite when fresh")
func loadFromSQLiteRoundtrip() async throws {
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

    // Delete .meta.json if it exists — force SQLite-only path
    let metaJSON = URL(fileURLWithPath: url.path + ".meta.json")
    try? FileManager.default.removeItem(at: metaJSON)

    let loaded = try await ANNSIndex.load(from: url)
    #expect(await loaded.count == 50)

    // Verify search works
    let query = vectors[0]
    let results = try await loaded.search(query: query, k: 5)
    #expect(results.first?.id == "node-0")
}

@Test("Load falls back to JSON when DB missing")
func loadFallsBackToJSONSidecar() async throws {
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
    let dbPath = url.deletingPathExtension().appendingPathExtension("db").path
    try? FileManager.default.removeItem(atPath: dbPath)

    // Write a legacy .meta.json sidecar manually.
    // PersistedMetadata is private, so encode an equivalent structure.
    struct LegacyPersistedMetadata: Encodable {
        let configuration: IndexConfiguration
        let softDeletion: SoftDeletion
        let metadataStore: MetadataStore?
        let idMap: IDMap?
    }
    let legacyMeta = LegacyPersistedMetadata(
        configuration: .default,
        softDeletion: SoftDeletion(),
        metadataStore: nil,
        idMap: nil
    )
    let metaJSON = try JSONEncoder().encode(legacyMeta)
    let metaURL = URL(fileURLWithPath: url.path + ".meta.json")
    try metaJSON.write(to: metaURL, options: .atomic)

    // Load should fall back to JSON sidecar since .db is missing
    let loaded = try await ANNSIndex.load(from: url)

    // The binary format embeds IDMap, so count should still be correct
    #expect(await loaded.count == 30)
}

@Test("Load falls back when DB is stale or unreadable")
func loadFallsBackWhenDBInvalid() async throws {
    let index = ANNSIndex(configuration: .default)
    var vectors: [[Float]] = []
    var ids: [String] = []
    for i in 0..<20 {
        vectors.append((0..<8).map { _ in Float.random(in: -1...1) })
        ids.append("corrupt-\(i)")
    }
    try await index.batchInsert(vectors: vectors, ids: ids)
    try await index.build()

    let dir = NSTemporaryDirectory() + "test-\(UUID().uuidString)"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: dir) }

    let url = URL(fileURLWithPath: dir).appendingPathComponent("corrupt.anns")
    try await index.save(to: url)

    // Corrupt the DB sidecar so SQLite load path throws
    let dbURL = URL(fileURLWithPath: url.deletingPathExtension()
        .appendingPathExtension("db").path)
    try Data([0x00, 0x01, 0x02, 0x03]).write(to: dbURL, options: .atomic)

    // Ensure JSON sidecar exists for fallback
    struct LegacyMeta: Encodable {
        let configuration: IndexConfiguration
        let softDeletion: SoftDeletion
        let metadataStore: MetadataStore?
    }
    let fallbackJSON = try JSONEncoder().encode(
        LegacyMeta(configuration: .default, softDeletion: SoftDeletion(), metadataStore: nil)
    )
    try fallbackJSON.write(
        to: URL(fileURLWithPath: url.path + ".meta.json"),
        options: .atomic
    )

    let loaded = try await ANNSIndex.load(from: url)
    #expect(await loaded.count == 20)
}
```

**Step 6.2: Confirm tests fail**

```bash
swift test --filter "loadFromSQLiteRoundtrip|loadFallsBackToJSONSidecar|loadFallsBackWhenDBInvalid" 2>&1 | tail -10
```

Expected: FAIL — current load path doesn't know about `.db` path.

**Step 6.3: Add freshness check helper**

Add to `ANNSIndex.swift`:

```swift
/// Returns true if a GRDB .db sidecar exists and is at least as recent as the .anns file.
private nonisolated static func hasFreshDatabase(at dbPath: String, forANNS annsPath: String) -> Bool {
    let fm = FileManager.default
    guard
        fm.fileExists(atPath: dbPath),
        fm.fileExists(atPath: annsPath),
        let dbAttrs = try? fm.attributesOfItem(atPath: dbPath),
        let annsAttrs = try? fm.attributesOfItem(atPath: annsPath),
        let dbDate = dbAttrs[.modificationDate] as? Date,
        let annsDate = annsAttrs[.modificationDate] as? Date
    else {
        return false
    }
    return dbDate >= annsDate
}
```

**Step 6.4: Rewrite `load(from:)`**

Replace the current `load(from:)` method (lines 885-910):

```swift
public static func load(from url: URL) async throws -> ANNSIndex {
    let dbPath = databasePath(for: url)
    let hasFreshDB = hasFreshDatabase(at: dbPath, forANNS: url.path)

    let persistedConfig: IndexConfiguration?
    let persistedSoftDeletion: SoftDeletion
    let persistedMetadataStore: MetadataStore
    let persistedIDMap: IDMap?

    if hasFreshDB {
        do {
            let db = try IndexDatabase(path: dbPath)
            persistedConfig = try db.loadConfiguration()
            persistedSoftDeletion = try db.loadSoftDeletion()
            persistedMetadataStore = try db.loadMetadataStore()
            persistedIDMap = try db.loadIDMap()
        } catch {
            // Corrupt or partially-written DB — fall back to legacy paths
            let legacy = try loadPersistedMetadataIfPresent(from: url)
            persistedConfig = legacy?.configuration
            persistedSoftDeletion = legacy?.softDeletion ?? SoftDeletion()
            persistedMetadataStore = legacy?.metadataStore ?? MetadataStore()
            persistedIDMap = legacy?.idMap
        }
    } else {
        // No fresh .db — backward compatibility via legacy paths
        let legacy = try loadPersistedMetadataIfPresent(from: url)
        persistedConfig = legacy?.configuration
        persistedSoftDeletion = legacy?.softDeletion ?? SoftDeletion()
        persistedMetadataStore = legacy?.metadataStore ?? MetadataStore()
        persistedIDMap = legacy?.idMap
    }

    let initialConfiguration = persistedConfig ?? .default
    let index = ANNSIndex(configuration: initialConfiguration)
    let loaded = try IndexSerializer.load(from: url, device: await index.currentDevice())

    let resolvedIDMap: IDMap
    if let persistedIDMap, persistedIDMap.count == loaded.idMap.count {
        resolvedIDMap = persistedIDMap
    } else {
        resolvedIDMap = loaded.idMap
    }

    var resolvedConfiguration = persistedConfig ?? .default
    resolvedConfiguration.metric = loaded.metric
    resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16
    resolvedConfiguration.useBinary = loaded.vectors is BinaryVectorBuffer

    await index.applyLoadedState(
        configuration: resolvedConfiguration,
        vectors: loaded.vectors,
        graph: loaded.graph,
        idMap: resolvedIDMap,
        entryPoint: loaded.entryPoint,
        softDeletion: persistedSoftDeletion,
        metadataStore: persistedMetadataStore
    )
    try await index.rebuildHNSWFromCurrentState()

    return index
}
```

**Step 6.5: Apply the same pattern to `loadMmap(from:)`**

Replace the current `loadMmap(from:)` (lines 912-938) with the same
`hasFreshDatabase` → `IndexDatabase` → fallback pattern. Use `MmapIndexLoader.load`
for the binary payload. Pass `isReadOnlyLoadedIndex: true` and `mmapLifetime`.

**Step 6.6: Apply the same pattern to `loadDiskBacked(from:)`**

Replace the current `loadDiskBacked(from:)` (lines 940-967) with the same pattern.
Use `DiskBackedIndexLoader.load` for the binary payload. Pass `isReadOnlyLoadedIndex: true`
and `mmapLifetime`.

> **Implementation note:** The metadata resolution logic (try IndexDatabase → catch → fallback)
> is identical across all three load variants. Consider extracting a private static helper
> `resolvePersistedState(for url:)` that returns a tuple of `(config, softDeletion, metadataStore, idMap?)`.
> This keeps the three load methods DRY. The helper is `nonisolated` and synchronous.

**Step 6.7: Verify new load tests pass**

```bash
swift test --filter "loadFromSQLiteRoundtrip|loadFallsBackToJSONSidecar|loadFallsBackWhenDBInvalid" 2>&1 | tail -15
```

Expected: PASS — all three green.

**Step 6.8: Run backward compatibility regression**

```bash
swift test --filter "PersistenceTests|MmapTests|DiskBackedTests" 2>&1 | tail -20
```

Expected: All existing persistence tests still pass.

```bash
swift test --filter ANNSIndexTests 2>&1 | tail -15
```

Expected: All ANNSIndex tests pass (including updated `saveAndLoadLifecycle`).

**Step 6.9: Run full suite**

```bash
swift test 2>&1 | grep -E "passed|failed|error:" | tail -5
```

Expected: Zero regressions.

**Step 6.10: Commit + update todo**

```bash
git add Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/ANNSIndexTests.swift
git commit -m "feat: ANNSIndex.load prefers SQLite with JSON sidecar fallback"
```

---

### Definition of Done

Phase 2 is complete when ALL of the following are true:

- [ ] `swift test --filter "saveCreatesDBFile|loadFromSQLiteRoundtrip|loadFallsBackToJSONSidecar|loadFallsBackWhenDBInvalid"` → all PASS
- [ ] `swift test --filter "PersistenceTests|MmapTests|DiskBackedTests"` → all PASS (backward compat)
- [ ] `swift test --filter ANNSIndexTests` → all PASS
- [ ] `swift test` → zero regressions
- [ ] `tasks/grdb-phase2-todo.md` → all checkboxes marked `[x]`
- [ ] Two commits in git log with exact messages above

---

### Anti-patterns (do NOT do these)

| Anti-pattern | Why |
|---|---|
| Writing `.meta.json` in `save()` | Phase 2 stops creating JSON sidecars — only `.db` |
| Deleting `SQLiteStructuredStore.swift` | Still needed for fallback reads of old `.meta.db` files |
| Writing directly to the final `.db` path | Partial writes on crash = corrupt database. Always temp-dir + move. |
| Skipping `prepareForFileMove()` before moving `.db` | WAL journal left behind = corrupt read on reopen |
| Adding `async` to `hasFreshDatabase` or `databasePath` | These are pure file-system checks — keep `nonisolated static` |
| Removing `PersistedMetadata` struct | Needed for JSON decoding of old `.meta.json` files |
| Ignoring `loadMmap` and `loadDiskBacked` | All three load variants must use the same IndexDatabase-first pattern |
| Calling `IndexDatabase.loadIDMap()` and ignoring the binary IDMap | Must still validate: if `persistedIDMap.count != serializer.idMap.count`, prefer serializer |

---

### What Comes Next (Phase 3)

Phase 3 wires `IndexDatabase` into `StreamingIndex` (Tasks 7-8 from the plan).
Phase 4 removes dead code: `SQLiteStructuredStore.swift`, JSON sidecar write paths,
and the `metadataURL`/`metadataDBURL` helpers (Tasks 9-10).
