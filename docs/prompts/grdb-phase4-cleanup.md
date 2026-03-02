# GRDB Migration — Phase 4 Execution Prompt

## Role

You are a senior Swift engineer executing the final phase of a GRDB storage migration
in the MetalANNS Swift Package. All implementation is done. Your job is to:

1. Prove correctness via a backward-compatibility integration test
2. Run the full test suite to close the regression gate
3. Add one documentation MARK to the legacy load path

**This phase is deliberately small.** Do not refactor beyond what is listed.

---

## Context

**Todo**: `tasks/grdb-phase4-todo.md`
**Migration Plan**: `docs/plans/2026-02-28-grdb-storage-migration.md` (Tasks 9–10)

### What has been done (Phases 1–3)

| Phase | Scope | Status |
|---|---|---|
| 1 | GRDB dependency + `IndexDatabase` + IDMap/metadata persistence | ✅ Complete |
| 2 | `ANNSIndex.save/load` wired to `IndexDatabase` | ✅ Complete |
| 3 | `StreamingDatabase` + `StreamingIndex.save/load` wired | ✅ Complete |

### Current storage model

```
BEFORE:                            AFTER (now):
index.anns                         index.anns         (unchanged binary)
index.anns.meta.json               index.db           (GRDB SQLite)

streaming/                         streaming/
  base.anns                          base.anns        (unchanged binary)
  base.anns.meta.json                streaming.db     (GRDB SQLite)
  streaming.meta.json
```

### Backward-compat fallback chain (ANNSIndex)

```
load(from: url)
  └─ resolvePersistedState(for: url)
       ├─ hasFreshDatabase? → IndexDatabase.load()   ← new path
       │       └─ on error → loadPersistedMetadataIfPresent()  ← fallback
       └─ (no fresh DB) → loadPersistedMetadataIfPresent()     ← legacy path
                           ├─ SQLiteStructuredStore (.meta.db)
                           └─ JSONDecoder (.meta.json)
```

---

## Codebase Facts

**Read these before touching any file. Do not guess.**

| Symbol | File | Line | Notes |
|---|---|---|---|
| `ANNSIndex.save(to:)` | `Sources/MetalANNS/ANNSIndex.swift` | 832 | Writes `.db` only — no `.meta.json` |
| `ANNSIndex.saveMmapCompatible(to:)` | same | 873 | Same pattern |
| `resolvePersistedState(for:)` | same | 1164 | Tries fresh DB → fallback |
| `loadPersistedMetadataIfPresent(from:)` | same | 1210 | Tries `.meta.db` then `.meta.json` |
| `metadataURL(for:)` | same | 1092 | Returns `<anns>.meta.json` — **do not delete** |
| `metadataDBURL(for:)` | same | 1096 | Returns `<anns>.meta.db` — **do not delete** |
| `hasFreshDatabase(at:forANNS:)` | same | 1149 | Compares mtime `.db` >= `.anns` |
| `databasePath(for:)` | same | 1100 | `.anns` → `.db` (strips extension) |
| `LegacyPersistedMetadata` local struct | `Tests/MetalANNSTests/ANNSIndexTests.swift` | ~183 | Already used in `loadFallsBackToJSONSidecar` |
| `IndexConfiguration.default` | — | — | Valid default for JSON fixture |
| `ANNSIndex.build(vectors:ids:)` | — | — | ⚠️ Combined API — NOT `batchInsert` + `build()` |

> **Critical API distinction**: `ANNSIndex` uses `build(vectors:ids:)` as the combined insert+build.
> `batchInsert(vectors:ids:)` is `StreamingIndex`'s API. Using the wrong one will cause a compile error.

---

## Constraints

- **Do not delete** `loadPersistedMetadataIfPresent`, `metadataURL(for:)`, `metadataDBURL(for:)`, or `SQLiteStructuredStore.swift` — all still needed for legacy fallback reads
- **Do not add** any new async/await to static file helpers
- **Do not refactor** `resolvePersistedState` or the load chain — it is correct as-is
- **Test framework**: Swift Testing (`@Test`, `#expect`) — NOT XCTest
- **Commit messages**: use the exact strings specified in each task

---

## Task 9 — Backward Compatibility Integration Test

### 9a — Write `backwardCompatLoadThenSaveUpgrades`

Add to `Tests/MetalANNSTests/ANNSIndexTests.swift` inside `ANNSIndexTests`:

```swift
@Test("Backward compatibility: old JSON sidecar loads and re-saves as SQLite")
func backwardCompatLoadThenSaveUpgrades() async throws {
    // 1. Build index
    var vectors: [[Float]] = (0..<40).map { i in
        (0..<8).map { j in Float(i * 8 + j) * 0.01 }
    }
    let ids = (0..<40).map { "compat-\($0)" }
    let index = ANNSIndex(configuration: .default)
    try await index.build(vectors: vectors, ids: ids)

    let dir = NSTemporaryDirectory() + "compat-\(UUID().uuidString)"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: dir) }

    let url = URL(fileURLWithPath: dir).appendingPathComponent("test.anns")
    let dbURL = URL(fileURLWithPath: url.deletingPathExtension().appendingPathExtension("db").path)

    // 2. Save (creates .db)
    try await index.save(to: url)
    #expect(FileManager.default.fileExists(atPath: dbURL.path), "Expected .db after save")

    // 3. Remove .db to simulate pre-migration file
    try FileManager.default.removeItem(at: dbURL)

    // 4. Write a valid .meta.json sidecar (old format)
    struct LegacyMeta: Encodable {
        let configuration: IndexConfiguration
        let softDeletion: SoftDeletion
        let metadataStore: MetadataStore?
    }
    let metaData = try JSONEncoder().encode(
        LegacyMeta(configuration: .default, softDeletion: SoftDeletion(), metadataStore: nil)
    )
    let metaURL = URL(fileURLWithPath: url.path + ".meta.json")
    try metaData.write(to: metaURL, options: .atomic)

    // 5. Load from legacy format (JSON fallback)
    let loaded = try await ANNSIndex.load(from: url)
    #expect(await loaded.count == 40, "Load from JSON sidecar should restore 40 vectors")

    // 6. Re-save — upgrades to SQLite
    try await loaded.save(to: url)
    #expect(FileManager.default.fileExists(atPath: dbURL.path), "Re-save should create .db")

    // 7. Load again from new SQLite format
    let reloaded = try await ANNSIndex.load(from: url)
    #expect(await reloaded.count == 40, "Load from SQLite should restore 40 vectors")

    // 8. Search still works
    let results = try await reloaded.search(query: vectors[0], k: 5)
    #expect(results.contains(where: { $0.id == "compat-0" }), "Search should find compat-0")
}
```

### 9b — TDD verification

```bash
# Must PASS
swift test --filter backwardCompatLoadThenSaveUpgrades 2>&1 | tail -10
```

If it fails, investigate and fix before continuing. Common failure causes:
- Used `batchInsert` instead of `build(vectors:ids:)` → compile error
- JSON fixture struct field mismatch with `PersistedMetadata`'s `Codable` init → decode error
- `dbURL` path derived differently than `databasePath(for:)` produces → mtime check mismatch

### 9c — Full suite gate

```bash
swift test 2>&1 | grep -E "passed|failed|error:" | tail -5
```

**Expected**: zero failures. This closes the pending full-suite gates from Phases 2 and 3.

### 9d — Commit

```bash
git add Tests/MetalANNSTests/ANNSIndexTests.swift
git commit -m "test: add backward compatibility and integration tests for GRDB migration"
```

---

## Task 10 — Document the Legacy Load Path

### 10a — Verify no `.meta.json` writes remain

```bash
grep -rn "meta\.json" Sources/
```

Expected output: only the `metadataURL(for:)` helper (line ~1092). No write calls.
If any write calls appear, **do not proceed** — investigate and fix.

### 10b — Add MARK and doc comment

In `Sources/MetalANNS/ANNSIndex.swift`, locate `loadPersistedMetadataIfPresent(from:)` (line ~1210).

Add above it:

```swift
// MARK: - Legacy JSON Sidecar (backward compatibility only)

/// Loads metadata from legacy sidecar formats.
/// Used only when no fresh `.db` file exists (pre-migration indexes).
/// Tries `.meta.db` (SQLiteStructuredStore) first, then `.meta.json`.
/// Do not call this directly — use `resolvePersistedState(for:)`.
private nonisolated static func loadPersistedMetadataIfPresent(from fileURL: URL) throws -> PersistedMetadata? {
```

No other changes to the function body.

### 10c — Confirm suite still passes

```bash
swift test 2>&1 | grep -E "passed|failed|error:" | tail -5
```

### 10d — Commit

```bash
git add Sources/MetalANNS/ANNSIndex.swift
git commit -m "refactor: mark JSON sidecar loading as legacy, document backward-compat path"
```

---

## Definition of Done

- [ ] `backwardCompatLoadThenSaveUpgrades` test passes
- [ ] Full suite: zero failures
- [ ] `loadPersistedMetadataIfPresent` has MARK + doc comment
- [ ] Two commits with exact messages above
- [ ] Todo file `tasks/grdb-phase4-todo.md` fully checked

---

## Anti-Patterns

| Don't | Why |
|---|---|
| Delete `loadPersistedMetadataIfPresent` | Still used for `.meta.db` and `.meta.json` fallback |
| Delete `metadataURL(for:)` or `metadataDBURL(for:)` | Called by `loadPersistedMetadataIfPresent` |
| Delete `SQLiteStructuredStore.swift` | Called by `loadPersistedMetadataIfPresent` |
| Use `batchInsert(vectors:ids:)` on `ANNSIndex` | That's `StreamingIndex`'s API — will not compile |
| Use `{}` as JSON fixture value | `IndexConfiguration.Codable` init requires valid fields |
| Modify `resolvePersistedState` | Correct as-is — leave it alone |
| Add new tests to other test files | Only `ANNSIndexTests.swift` in this phase |
