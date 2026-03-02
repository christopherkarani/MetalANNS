# GRDB Migration — Phase 3: StreamingIndex Integration

> **Status**: PENDING
> **Owner**: (unassigned)
> **Last Updated**: 2026-03-02
> **Reference Plan**: `docs/plans/2026-02-28-grdb-storage-migration.md` Tasks 7–8
> **Depends On**: Phase 2 complete (`tasks/grdb-phase2-todo.md`)

---

## Goal

Create `StreamingDatabase` (SQLite-backed streaming state) and wire it into
`StreamingIndex.save()` and `StreamingIndex.load()`, replacing the legacy
`SQLiteStructuredStore` (`.meta.db`) and JSON sidecar (`.meta.json`) with a
proper GRDB-backed `streaming.db` file.

Key behaviors:
- **Save** writes `base.anns` (binary via ANNSIndex) + `streaming.db` (GRDB). No more JSON.
- **Load** reads `streaming.db` first (if fresh). Falls back to `.meta.db` → `.meta.json`.
- Vectors are stored as individual BLOB rows (not a giant JSON `[[Float]]` blob).
- `MetadataValue` enum serialized as JSON strings per key-value pair.

---

## Task Checklist

### Task 7 — Create StreamingDatabase

#### 7a — Write failing tests
- [x] Create `Tests/MetalANNSTests/Storage/StreamingDatabaseTests.swift`
- [x] Write `testInsertAndFetchVectors` — insert 3 vectors, load all, verify count + data
- [x] Write `testIncrementalInsert` — two separate inserts, verify both present
- [x] Write `testMarkDeleted` — insert 3, mark 1 deleted, verify deleted set
- [x] Write `testSaveAndLoadConfig` — round-trip `StreamingConfiguration`
- [x] Write `testSaveAndLoadPerVectorMetadata` — round-trip string metadata
- [x] Run `swift test --filter StreamingDatabaseTests 2>&1 | tail -10` — confirm FAIL

#### 7b — Implement StreamingDatabase
- [x] Create `Sources/MetalANNS/Storage/StreamingDatabase.swift`
- [x] Implement schema migration `v1-streaming` with tables: `vectors`, `deleted`, `vector_metadata`, `config`, `state`
- [x] Implement `insertVectors(_:ids:)` — Float arrays → BLOB rows
- [x] Implement `loadAllVectors()` — BLOB rows → Float arrays
- [x] Implement `markDeleted(ids:)` / `loadDeletedIDs()`
- [x] Implement `saveConfig(_:)` / `loadConfig()`
- [x] Implement `saveVectorMetadata(id:metadata:)` / `loadVectorMetadata(id:)`
- [x] Implement `saveAllVectorMetadata(_:)` / `loadAllVectorMetadata()`
- [x] Implement `saveVectorDimension(_:)` / `loadVectorDimension()`
- [x] Run `swift test --filter StreamingDatabaseTests 2>&1 | tail -10` — confirm PASS
- [x] Commit: `feat: add StreamingDatabase for SQLite-backed streaming state`

### Task 8 — Wire StreamingDatabase into StreamingIndex Save/Load

#### 8a — Write failing tests
- [x] Write `saveCreatesStreamingDB` in `StreamingIndexPersistenceTests.swift`:
  - Insert 20 vectors, flush, save → assert `streaming.db` exists, `streaming.meta.json` does NOT
- [x] Write `saveLoadRoundtripViaSQLite`:
  - Insert 30 vectors, flush, save → load → verify count == 30
- [x] Run `swift test --filter "saveCreatesStreamingDB|saveLoadRoundtripViaSQLite" 2>&1 | tail -10` — confirm FAIL

#### 8b — Implement save path
- [x] Rewrite `StreamingIndex.save(to:)` to:
  - Snapshot state, create temp dir, save `base.anns`, write `StreamingDatabase` with all data
  - Slice flat `allVectorData` → `[[Float]]` by `vectorDimension` for `insertVectors`
  - Convert `MetadataValue` → JSON strings for `saveAllVectorMetadata`
  - Replace directory atomically
  - No more `streaming.meta.json` or `streaming.meta.db` writes
- [x] Run `swift test --filter "saveCreatesStreamingDB" 2>&1 | tail -10` — confirm PASS

#### 8c — Implement load path
- [x] Add `hasFreshStreamingDatabase(at:forBaseANNS:)` static helper
- [x] Rewrite `StreamingIndex.load(from:)` to:
  - Try `StreamingDatabase` first (if fresh `streaming.db` exists)
  - Reconstruct `PersistedMeta` from database fields (slice vectors, decode MetadataValue)
  - Fall back to `SQLiteStructuredStore.load` (`.meta.db`) → JSON (`.meta.json`)
- [x] Run `swift test --filter "saveLoadRoundtripViaSQLite" 2>&1 | tail -10` — confirm PASS

#### 8d — Update existing tests + regression check
- [x] Update `saveAndLoadEmpty` test: change `.meta.db` → `streaming.db` assertion
- [x] Update `searchAfterLoad` test: change `.meta.json` cleanup to match new paths
- [x] Update `saveRequiresFlush` test: change `.meta.json` cleanup to match new paths
- [x] Run `swift test --filter StreamingIndexPersistenceTests 2>&1 | tail -15` — confirm all pass
- [ ] Run `swift test 2>&1 | grep -E "passed|failed|error:" | tail -5` — zero regressions
- [x] Commit: `feat: StreamingIndex save/load uses SQLite, JSON fallback for backward compat`

---

## Verification Gate

```bash
swift test --filter StreamingDatabaseTests 2>&1 | tail -15
swift test --filter StreamingIndexPersistenceTests 2>&1 | tail -15
swift test 2>&1 | grep -E "passed|failed|error:" | tail -5
```

- All `StreamingDatabaseTests` pass
- All `StreamingIndexPersistenceTests` pass (including updated existing tests)
- Zero regressions in full suite
- Two commits with exact messages above

---

## Review Checklist

- [x] `StreamingDatabase` is `Sendable` (uses `DatabasePool`)
- [x] Vectors stored as per-row BLOBs (not a single JSON blob)
- [x] `insertVectors` correctly converts `[Float]` → `Data` via `withUnsafeBytes`
- [x] `loadAllVectors` correctly decodes BLOB → `[Float]` with size validation
- [x] Flat `allVectorData` sliced correctly by `vectorDimension` before `insertVectors`
- [x] `MetadataValue` round-trips through JSON encoding (private enum stays private)
- [x] `PersistedMeta` reconstructed correctly from StreamingDatabase fields
- [x] `validateLoadedMeta` still called on both SQLite and JSON load paths
- [x] Save uses temp directory + `replaceDirectory` (atomic)
- [x] `SQLiteStructuredStore` NOT deleted (still needed for `.meta.db` fallback reads)
