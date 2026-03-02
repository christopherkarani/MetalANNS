# GRDB Phase 1: Storage Foundation

## Status
- Last Updated: 2026-03-02
- Branch: phase2-gpu-structures

## Task 1: Add GRDB dependency
- [x] Add GRDB.swift dependency and imports in Package.swift
- [x] Update MetalANNS target dependencies with GRDB
- [x] Update MetalANNSTests target dependencies with GRDB
- [x] Run `swift package resolve`
- [x] Run `swift build` and confirm build succeeds
- [x] Commit `deps: add GRDB.swift for SQLite-backed persistence`
  - Notes: Resolved GRDB@7.10.0 via cache and verified `swift build` succeeds after subsequent access-control fix.

## Task 2: IndexDatabase foundation
- [x] Create `Sources/MetalANNS/Storage/IndexDatabase.swift`
- [x] Add `IndexDatabase` type with `DatabasePool` and migration
- [x] Create `Tests/MetalANNSTests/Storage/IndexDatabaseTests.swift` with create/reopen tests
- [x] Confirm foundation tests fail before implementation
- [x] Confirm foundation tests pass after implementation
- [x] Commit `feat: add IndexDatabase foundation with schema v1`
  - Notes: Implemented full v1 schema migration with `idmap`, `config`, and `soft_deletion` and added WAL + mmap settings.

## Task 3: IDMap persistence
- [x] Add failing save/load IDMap test cases
- [x] Add `IDMap.makeForPersistence(rows:nextID:)` extension in `Sources/MetalANNSCore/IDMap.swift`
- [x] Add `saveIDMap` and `loadIDMap` to IndexDatabase
- [x] Confirm IDMap tests pass
- [x] Commit `feat: add IDMap save/load to IndexDatabase`
  - Notes: Used package-level factory helper to reconstruct IDs without `assign()` to preserve explicit IDs and `nextInternalID`.

## Task 4: Metadata persistence
- [x] Add failing configuration, soft deletion, and metadata tests
- [x] Add config persistence methods
- [x] Add soft-deletion persistence methods
- [x] Add MetadataStore persistence methods
- [x] Confirm metadata tests pass
- [x] Run full regression suite
- [x] Commit `feat: add config, soft-deletion, and metadata persistence to IndexDatabase`
  - Notes: Full `swift test` still shows pre-existing environment-related Metal shader failures (12 issues) and one streaming merge timing expectation; no failures introduced by GRDB persistence tests.
