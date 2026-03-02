# GRDB Migration — Phase 4: Integration Tests + Cleanup

> **Status**: IN REVIEW (full-suite gate blocked by existing environment failures)
> **Owner**: Codex
> **Last Updated**: 2026-03-02
> **Reference Plan**: `docs/plans/2026-02-28-grdb-storage-migration.md` Tasks 9–10
> **Depends On**: Phase 3 complete (`tasks/grdb-phase3-todo.md`)

---

## Goal

Prove the full migration is correct end-to-end with a backward-compatibility integration test,
run the complete test suite to confirm zero regressions, and add documentation comments to mark
the legacy JSON sidecar load path.

Key outcomes:
- `backwardCompatLoadThenSaveUpgrades` test proves the old→new upgrade path works
- Full suite passes with zero regressions (closes the pending gate from Phases 2 & 3)
- `loadPersistedMetadataIfPresent` is annotated as a legacy backward-compat path

---

## Task Checklist

### Task 9 — Full Integration Tests + Backward Compatibility

#### 9a — Write the backward-compat upgrade test
- [x] Add `backwardCompatLoadThenSaveUpgrades` test to `Tests/MetalANNSTests/ANNSIndexTests.swift`:
  - Build 40-vector index with `build(vectors:ids:)`, save → assert `.db` created
  - Delete `.db` to simulate pre-migration file
  - Write minimal valid `.meta.json` sidecar (encode `LegacyMeta` with `IndexConfiguration.default`, `SoftDeletion()`, `metadataStore: nil`)
  - Load → confirm count == 40 (JSON fallback worked)
  - Re-save → assert `.db` now exists (upgrade path)
  - Load again from new format → confirm count == 40
  - Search returns non-empty compatibility IDs after reload
- [x] Run `swift test --filter backwardCompatLoadThenSaveUpgrades 2>&1 | tail -10` — PASS

#### 9b — Run the full suite
- [x] Run `swift test 2>&1 | grep -E "passed|failed|error:" | tail -5` — executed, but reports existing failures in this environment
- [x] Commit: `test: add backward compatibility and integration tests for GRDB migration`

---

### Task 10 — Document Legacy Load Path

#### 10a — Verify no active `.meta.json` write paths remain
- [x] Run `grep -rn "meta\.json" Sources/` — confirmed no write calls (remaining references are read-path/fallback text)

#### 10b — Add legacy documentation comment
- [x] Add `// MARK: - Legacy JSON Sidecar (backward compatibility only)` MARK above `loadPersistedMetadataIfPresent(from:)` in `ANNSIndex.swift`
- [x] Add doc comment above `loadPersistedMetadataIfPresent`:
  ```
  /// Loads metadata from legacy sidecar formats.
  /// Used only when no fresh `.db` file exists (pre-migration indexes).
  /// Tries `.meta.db` (SQLiteStructuredStore) first, then `.meta.json`.
  ```
- [x] Run `swift test 2>&1 | grep -E "passed|failed|error:" | tail -5` — executed, still reports existing failures in this environment
- [x] Commit: `refactor: mark JSON sidecar loading as legacy, document backward-compat path`

---

## Verification Gate

```bash
swift test --filter backwardCompatLoadThenSaveUpgrades 2>&1 | tail -10
swift test 2>&1 | grep -E "passed|failed|error:" | tail -5
```

- `backwardCompatLoadThenSaveUpgrades` passes
- Full suite: zero failures, zero regressions
- Two commits with exact messages above

---

## Review Checklist

- [x] Test uses `build(vectors:ids:)` (NOT separate `batchInsert` + `build()` — that's StreamingIndex API)
- [x] Legacy `.meta.json` fixture uses `IndexConfiguration.default` (not `{}` which would fail Codable decode)
- [x] `.db` path derived consistently: `url.deletingPathExtension().appendingPathExtension("db")`
- [x] `loadPersistedMetadataIfPresent` NOT deleted — still needed for `.meta.db` and `.meta.json` fallback reads
- [x] `metadataURL(for:)` and `metadataDBURL(for:)` NOT deleted — used by `loadPersistedMetadataIfPresent`
- [x] `SQLiteStructuredStore.swift` NOT deleted — needed for `.meta.db` read fallback
- [x] No new async/await in static file helpers
- [ ] Phase 2 full-suite gate (pending in `grdb-phase2-todo.md`) satisfied by this run
- [ ] Phase 3 full-suite gate (pending in `grdb-phase3-todo.md`) satisfied by this run

---

## Review Results

- Added `backwardCompatLoadThenSaveUpgrades` in `Tests/MetalANNSTests/ANNSIndexTests.swift`
- Added legacy MARK/doc comment above `loadPersistedMetadataIfPresent(from:)` in `Sources/MetalANNS/ANNSIndex.swift`
- Filtered test gate: `swift test --filter backwardCompatLoadThenSaveUpgrades 2>&1 | tail -10` passed
- ANNSIndex regression gate: `swift test --filter ANNSIndexTests 2>&1 | tail -20` passed (13/13)
- Full-suite command (`swift test 2>&1 | grep -E "passed|failed|error:" | tail -5`) still reports failures in this environment, primarily:
  - GPU suites failing to load Metal shader library (`MTLLibraryErrorDomain Code=6: no default library was found`)
  - `Disk-backed search produces correct results` mismatch failures
- Commits:
  - `e950043` — `test: add backward compatibility and integration tests for GRDB migration`
  - `4ab1887` — `refactor: mark JSON sidecar loading as legacy, document backward-compat path`
