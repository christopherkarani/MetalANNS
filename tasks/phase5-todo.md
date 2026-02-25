# MetalANNS — Phase 5: Persistence & Incremental Operations

> **Status**: PENDING
> **Owner**: Subagent (dispatched by orchestrator)
> **Reviewer**: Orchestrator (main session)
> **Last Updated**: 2026-02-25 08:34:37

---

## How This File Works

This file is the **shared communication layer** between the orchestrator and executing agents.

**Executing agent**: Check off items `[x]` as you complete them. Add notes under any item if you hit issues. Update `Last Updated` timestamp after each task. Do NOT check off an item unless the verification step passes.

**Orchestrator**: Reviews checked items against actual codebase state. Unchecks items that don't pass review. Adds review comments prefixed with `> [REVIEW]`.

---

## Pre-Flight Checks

- [x] Working directory is `/Users/chriskarani/CodingProjects/MetalANNS`
- [x] Phase 1–4 code exists: `BeamSearchCPU.swift`, `SearchGPU.swift`, `SearchResult.swift`, `NNDescentCPU.swift`, `NNDescentGPU.swift`, `Distance.metal`, `NNDescent.metal`, `Sort.metal` all present
- [x] `git log --oneline` baseline verified (expect 17 commits before Phase 5 execution)
- [x] Full test suite passes (36 tests, zero failures): `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
- [x] Read implementation plan: `docs/plans/2026-02-25-metalanns-implementation.md` (lines 2515–2567, Tasks 16–18)
- [x] Read this phase's prompt: `docs/prompts/phase-5-persistence.md` (detailed spec for Tasks 16–18)

---

## Task 16: Index Serialization

**Acceptance**: `PersistenceTests` suite passes (3 tests). Eighteenth commit.

- [x] 16.1 — Create `Tests/MetalANNSTests/PersistenceTests.swift` — 3 tests using Swift Testing:
  - `saveAndLoadRoundtrip`:
    - Build small index: 50 nodes, dim=8, degree=4 via NNDescentCPU
    - Create VectorBuffer + GraphBuffer, populate from CPU graph
    - Save to temp file, load back
    - Verify: loaded dimensions match, entry point matches
    - Run search on both original and loaded, verify same top-5 results
  - `corruptMagicThrows`:
    - Write a file with wrong magic bytes ('XXXX' instead of 'MANN')
    - Assert `IndexSerializer.load(from:device:)` throws `ANNSError.corruptFile`
  - `corruptVersionThrows`:
    - Write valid magic but version=99
    - Assert throws `ANNSError.corruptFile`
- [x] 16.2 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/PersistenceTests 2>&1 | grep -E '(PASS|FAIL|error:)'` → confirms FAIL
- [x] 16.3 — Create `Sources/MetalANNSCore/IndexSerializer.swift`:
  - `public enum IndexSerializer` — stateless, all static methods
  - `static func save(vectors:graph:idMap:entryPoint:metric:to:) throws`
  - `static func load(from:device:) throws -> (vectors: VectorBuffer, graph: GraphBuffer, idMap: IDMap, entryPoint: UInt32, metric: Metric)`
  - Binary format: 24-byte header (MANN magic, version, nodeCount, degree, dim, metric) + raw buffer bytes + JSON IDMap + entryPoint
  - Throws `ANNSError.corruptFile` on invalid magic or version
- [x] 16.4 — **DECISION POINT (16.5)**: How to populate GraphBuffer from NNDescentCPU output `[[(UInt32, Float)]]` in tests. Options: (a) helper on GraphBuffer, (b) inline in test, (c) utility function. **Document approach in notes below.**
- [x] 16.5 — **GREEN**: All 3 tests pass. Specifically confirm `saveAndLoadRoundtrip` produces identical search results.
- [x] 16.6 — **REGRESSION**: All Phase 1–4 tests still pass (36 prior tests)
- [x] 16.7 — **GIT**: `git add Sources/MetalANNSCore/IndexSerializer.swift Tests/MetalANNSTests/PersistenceTests.swift && git commit -m "feat: implement index serialization with binary file format"`

> **Agent notes** _(REQUIRED — document 16.5 decision)_:
>
> Used inline test helpers in `PersistenceTests.swift` to map NNDescentCPU rows directly into `GraphBuffer` slots (no production API changes).

---

## Task 17: Incremental Insert

**Acceptance**: `IncrementalTests` suite passes (2 tests). Nineteenth commit.

- [x] 17.1 — Create `Tests/MetalANNSTests/IncrementalTests.swift` — 2 tests using Swift Testing:
  - `insertAndFindNew`:
    - Build index with 100 vectors (dim=8, degree=4) via NNDescentCPU
    - Populate VectorBuffer + GraphBuffer
    - Insert 10 new random vectors via IncrementalBuilder
    - For each newly inserted vector, search — verify it appears in its own top-1 result (distance ≈ 0)
  - `insertRecallDegradation`:
    - Build index with 200 vectors (dim=16, degree=8) via NNDescentCPU
    - Measure baseline recall (10 queries, k=5, ef=32)
    - Insert 20 new vectors via IncrementalBuilder
    - Measure recall again (10 queries over all 220 vectors)
    - Assert: recall degradation < 5% (new recall > baseline - 0.05)
- [x] 17.2 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/IncrementalTests 2>&1 | grep -E '(PASS|FAIL|error:)'` → confirms FAIL
- [x] 17.3 — Create `Sources/MetalANNSCore/IncrementalBuilder.swift`:
  - `public enum IncrementalBuilder` — stateless, all static methods
  - `static func insert(vector:at:into:graph:vectors:entryPoint:metric:degree:) throws`
  - Algorithm: beam search from entry point → find `degree` nearest neighbors → set new node's neighbors → reverse update into neighbors' lists
  - Inline distance computation (same pattern as BeamSearchCPU)
- [x] 17.4 — **DECISION POINT (17.5)**: How to search on GraphBuffer in tests. Options: (a) extract to arrays and use BeamSearchCPU, (b) use SearchGPU if GPU available, (c) write thin adapter. **Document approach in notes below.**
- [x] 17.5 — **GREEN**: Both tests pass. Specifically confirm `insertRecallDegradation` shows < 5% degradation.
- [x] 17.6 — **REGRESSION**: All prior tests still pass (36 + 3 from Task 16 = 39)
- [x] 17.7 — **GIT**: `git add Sources/MetalANNSCore/IncrementalBuilder.swift Tests/MetalANNSTests/IncrementalTests.swift && git commit -m "feat: implement incremental vector insertion with local graph repair"`

> **Agent notes** _(REQUIRED — document 17.5 decision)_:
>
> Extracted `GraphBuffer` and `VectorBuffer` to CPU arrays and used `BeamSearchCPU` for assertions in both incremental tests.

---

## Task 18: Soft Deletion

**Acceptance**: `DeletionTests` suite passes (3 tests). Twentieth commit.

- [x] 18.1 — Create `Tests/MetalANNSTests/DeletionTests.swift` — 3 tests using Swift Testing:
  - `deletedNotInResults`:
    - Build index with 50 vectors (dim=8, degree=4) via NNDescentCPU
    - Mark internal IDs 0, 5, 10 as deleted
    - Run search, filter results through `SoftDeletion.filterResults()`
    - Assert: none of the deleted IDs appear in results
  - `deletedCountTracking`:
    - Create SoftDeletion, mark 5 IDs as deleted
    - Assert `deletedCount == 5`
    - Mark same ID again — assert `deletedCount` still 5 (idempotent)
  - `undeleteRestores`:
    - Mark ID as deleted, verify `isDeleted` returns true
    - Undelete, verify `isDeleted` returns false
- [x] 18.2 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/DeletionTests 2>&1 | grep -E '(PASS|FAIL|error:)'` → confirms FAIL
- [x] 18.3 — Create `Sources/MetalANNSCore/SoftDeletion.swift`:
  - `public struct SoftDeletion: Sendable, Codable`
  - `private var deletedIDs: Set<UInt32>`
  - Methods: `markDeleted(_:)`, `undelete(_:)`, `isDeleted(_:)`, `deletedCount`, `filterResults(_:)`
  - `filterResults` removes any `SearchResult` whose `internalID` is in the deleted set
- [x] 18.4 — **DECISION POINT (18.4)**: Should `filterResults` request extra results to compensate for deletions? Recommended: no — keep it a pure filter, caller handles ef adjustment. **Document decision in notes below.**
- [x] 18.5 — **GREEN**: All 3 tests pass. Specifically confirm `deletedNotInResults` shows zero deleted IDs in results.
- [x] 18.6 — **REGRESSION**: All prior tests still pass (36 + 3 + 2 from Tasks 16–17 = 41)
- [x] 18.7 — **GIT**: `git add Sources/MetalANNSCore/SoftDeletion.swift Tests/MetalANNSTests/DeletionTests.swift && git commit -m "feat: implement soft deletion with filtered search results"`

> **Agent notes** _(REQUIRED — document 18.4 decision)_:
>
> Kept `filterResults(_:)` as a pure filtering step and deferred any `ef` compensation strategy to higher-level API layers.

---

## Phase 5 Complete — Signal

When all items above are checked, update this section:

```
STATUS: _complete_
FINAL TEST RESULT: _all tests passed (44 tests, 17 suites)_
TOTAL COMMITS: _20_
ISSUES ENCOUNTERED:
- Corrupt binary read used a misaligned `UnsafeRawPointer` load pattern initially; fixed via explicit little-endian byte assembly in `IndexSerializer.readUInt32`.
- `IncrementalBuilder` was adjusted so fallback reverse-linking keeps inserted nodes reachable when neighbor replacement is not immediately beneficial.
DECISIONS MADE:
- 16.5: populated `GraphBuffer` from `NNDescentCPU` output inline in `PersistenceTests.swift` with helper functions.
- 17.5: converted `GraphBuffer`/`VectorBuffer` content to CPU arrays in tests and reused `BeamSearchCPU` for deterministic assertions.
- 18.4: kept `filterResults(_:)` as a pure filter; search-side `ef` compensation stays in higher-level API.
```

---

## Orchestrator Review Checklist (DO NOT MODIFY — Orchestrator use only)

- [ ] R1 — Git log shows exactly 20 commits with correct conventional commit messages
- [ ] R2 — Full test suite passes: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'` — zero failures including all Phase 1–4 tests
- [ ] R3 — `IndexSerializer` is a stateless `enum` with `static func save(...)` and `static func load(...)`
- [ ] R4 — Binary format starts with 'MANN' magic (4 bytes), version=1, followed by nodeCount, degree, dim, metric
- [ ] R5 — `load` throws `ANNSError.corruptFile` on invalid magic or version
- [ ] R6 — Round-trip save/load produces identical search results
- [ ] R7 — `IncrementalBuilder` is a stateless `enum` with `static func insert(...)`
- [ ] R8 — Incremental insert uses beam search to find neighbors, not brute force
- [ ] R9 — Reverse neighbor update is implemented (new node appears in existing neighbors' lists)
- [ ] R10 — Recall degradation after 20 inserts into 200-node index is < 5%
- [ ] R11 — `SoftDeletion` is a `struct` conforming to `Sendable` and `Codable`
- [ ] R12 — Deleted IDs never appear in filtered search results
- [ ] R13 — `filterResults` is a pure filter (no search logic, no ef adjustment)
- [ ] R14 — `deletedCount` is idempotent (marking same ID twice doesn't double-count)
- [ ] R15 — No `import XCTest` or `XCTSkip` anywhere
- [ ] R16 — No Phase 6+ code leaked in (no ANNSIndex actor, no integration test, no README)
- [ ] R17 — No Phase 1–4 files were modified (or changes are documented and justified)
- [ ] R18 — Agent notes filled in for decision points 16.5, 17.5, and 18.4
- [ ] R19 — All new types are `Sendable` (IndexSerializer as enum is inherently Sendable, SoftDeletion conforms explicitly)
- [ ] R20 — Test file cleanup: temp files created during persistence tests are removed after test
