# GRDB Migration — Phase 2: ANNSIndex Integration

> **Status**: PENDING
> **Owner**: (unassigned)
> **Last Updated**: 2026-03-02
> **Reference Plan**: `docs/plans/2026-02-28-grdb-storage-migration.md` Tasks 5–6
> **Depends On**: Phase 1 complete (`tasks/grdb-phase1-todo.md`)

---

## Goal

Wire `IndexDatabase` (from Phase 1) into `ANNSIndex.save()` and `ANNSIndex.load()`,
replacing the legacy `SQLiteStructuredStore` (.meta.db) and JSON sidecar (.meta.json)
with a proper GRDB-backed `.db` file.

Key behaviors:
- **Save** writes binary `.anns` + SQLite `.db` (atomic two-phase via temp dir). No more `.meta.json`.
- **Load** reads SQLite `.db` first (if fresh). Falls back to legacy `.meta.db`/`.meta.json` for backward compat.
- **All three load variants** (`load`, `loadMmap`, `loadDiskBacked`) get the same treatment.

---

## Task Checklist

### Task 5 — Wire IndexDatabase into ANNSIndex Save

#### 5a — Save test
- [x] Write failing test `saveCreatesDBFile` in `ANNSIndexTests.swift`:
  - Builds 50-vector index, saves to temp dir
  - Asserts `.anns` binary exists
  - Asserts `.db` SQLite sidecar exists (new path — NOT `.meta.db`)
  - Asserts `.meta.json` does NOT exist
- [x] Run `swift test --filter saveCreatesDBFile 2>&1 | tail -10` — confirm FAIL

#### 5b — Implement new save path
- [x] Add `databasePath(for:)` static helper to `ANNSIndex.swift` — strips file extension, appends `.db`
- [x] Add `replaceFile(at:with:)` static helper for atomic file swap
- [x] Add `replaceSQLiteFiles(at:with:)` static helper — moves `.db`, `-wal`, `-shm` atomically
- [x] Rewrite `save(to:)` to use two-phase temp-dir pattern with `IndexDatabase`
- [x] Rewrite `saveMmapCompatible(to:)` with same pattern
- [x] Run `swift test --filter saveCreatesDBFile 2>&1 | tail -10` — confirm PASS

#### 5c — Update existing save test
- [x] Update `saveAndLoadLifecycle` test in `ANNSIndexTests.swift`:
  - Change `.meta.db` expectation to `.db`
  - Update cleanup `defer` block for new `.db` path
- [x] Run `swift test --filter "saveCreatesDBFile|saveAndLoadLifecycle" 2>&1 | tail -10` — confirm PASS
- [x] Commit: `feat: ANNSIndex.save writes SQLite alongside binary format`

### Task 6 — Wire IndexDatabase into ANNSIndex Load (with Fallback)

#### 6a — Load tests
- [x] Write failing test `loadFromSQLiteRoundtrip` in `ANNSIndexTests.swift`:
  - Save → delete `.meta.json` → load → verify count + search
- [x] Write failing test `loadFallsBackToJSONSidecar` in `ANNSIndexTests.swift`:
  - Save → delete `.db` → create legacy `.meta.json` manually → load → verify count
- [x] Write failing test `loadFallsBackWhenDBInvalid` in `ANNSIndexTests.swift`:
  - Save → corrupt `.db` with junk bytes → create `.meta.json` fallback → load → verify count
- [x] Run `swift test --filter "loadFromSQLiteRoundtrip|loadFallsBackToJSONSidecar|loadFallsBackWhenDBInvalid" 2>&1 | tail -10` — confirm FAIL

#### 6b — Implement new load path
- [x] Add `hasFreshDatabase(at:forANNS:)` static helper — checks `.db` exists and mtime >= `.anns` mtime
- [x] Rewrite `loadPersistedMetadataIfPresent(from:)` OR replace with direct `IndexDatabase` usage in `load(from:)`:
  - Try `IndexDatabase` path first (if fresh)
  - On failure or staleness, fall back to `SQLiteStructuredStore.load` then JSON sidecar
- [x] Apply same pattern to `loadMmap(from:)` and `loadDiskBacked(from:)`
- [x] Run `swift test --filter "loadFromSQLiteRoundtrip|loadFallsBackToJSONSidecar|loadFallsBackWhenDBInvalid" 2>&1 | tail -10` — confirm PASS

#### 6c — Backward compat regression check
- [x] Run `swift test --filter "PersistenceTests|MmapTests|DiskBackedTests" 2>&1 | tail -20` — confirm all pass
- [x] Run `swift test --filter ANNSIndexTests 2>&1 | tail -15` — confirm all pass
- [ ] Run full suite: `swift test 2>&1 | grep -E "passed|failed|error:" | tail -5` — zero regressions
- [x] Commit: `feat: ANNSIndex.load prefers SQLite with JSON sidecar fallback`

---

## Verification Gate (all must pass before Phase 2 is complete)

```bash
swift test --filter "saveCreatesDBFile|loadFromSQLiteRoundtrip|loadFallsBackToJSONSidecar|loadFallsBackWhenDBInvalid" 2>&1 | tail -15
swift test --filter "PersistenceTests|ANNSIndexTests" 2>&1 | tail -15
swift test 2>&1 | grep -E "passed|failed|error:" | tail -5
```

- All new tests pass
- All existing persistence tests still pass (backward compat)
- Zero regressions in full suite
- Two commits with exact messages above

---

## Review Checklist

- [x] `save()` uses a temp directory — never writes directly to the final `.db` path (avoids partial writes)
- [x] `save()` calls `db.prepareForFileMove()` before moving SQLite files
- [x] `replaceSQLiteFiles` handles `-wal` and `-shm` sidecars (move existing, remove stale)
- [x] `hasFreshDatabase` compares mtime of `.db` vs `.anns` — prevents reading stale SQLite
- [x] Load falls back gracefully when `.db` is missing (legacy format)
- [x] Load falls back gracefully when `.db` is corrupt (junk bytes)
- [x] `loadMmap` and `loadDiskBacked` use the same fallback pattern as `load`
- [ ] `PersistedMetadata` struct is unchanged (backward compat for JSON decoding)
- [x] No `async` added to any static file helpers (they're `nonisolated`)
- [x] `SQLiteStructuredStore` is NOT deleted yet (Phase 4 cleanup — still needed for fallback reads)
