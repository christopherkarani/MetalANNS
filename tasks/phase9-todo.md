# MetalANNS — Phase 9: Float16 Support

> **Status**: PENDING
> **Owner**: Subagent (dispatched by orchestrator)
> **Reviewer**: Orchestrator (main session)
> **Last Updated**: 2026-02-25

---

## How This File Works

This file is the **shared communication layer** between the orchestrator and executing agents.

**Executing agent**: Check off items `[x]` as you complete them. Add notes under any item if you hit issues. Update `Last Updated` timestamp after each task. Do NOT check off an item unless the verification step passes.

**Orchestrator**: Reviews checked items against actual codebase state. Unchecks items that don't pass review. Adds review comments prefixed with `> [REVIEW]`.

---

## Pre-Flight Checks

- [ ] Working directory is `/Users/chriskarani/CodingProjects/MetalANNS`
- [ ] Phase 1–8 code exists: `ANNSIndex.swift`, `VectorBuffer.swift`, `FullGPUSearch.swift`, `NNDescentGPU.swift`, `IndexSerializer.swift`, `GraphPruner.swift`, `Search.metal`, `Distance.metal`, `NNDescent.metal`, `Sort.metal` all present
- [ ] `git log --oneline | wc -l` baseline verified (expect 29 commits before Phase 9 execution)
- [ ] Full test suite passes (61 tests, zero failures): `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
- [ ] Read implementation plan: `docs/plans/2026-02-25-metalanns-v2-performance-features.md` (Phase 9 section, Tasks 26–27)
- [ ] Read this phase's prompt: `docs/prompts/phase-9-float16.md` (detailed spec for Tasks 26–27)

---

## Task 26: Float16 VectorBuffer + Distance Kernels + Beam Search Kernel

**Acceptance**: `Float16Tests` suite passes (2 tests), new files created correctly. Thirtieth commit.

- [ ] 26.1 — Create `Sources/MetalANNSCore/VectorStorage.swift` — `VectorStorage` protocol:
  - Properties: `buffer: MTLBuffer`, `dim: Int`, `capacity: Int`, `count: Int`, `isFloat16: Bool`
  - Methods: `setCount(_:)`, `insert(vector:at:)`, `batchInsert(vectors:startingAt:)`, `vector(at:) -> [Float]`
  - Must be `AnyObject & Sendable`
- [ ] 26.2 — Modify `Sources/MetalANNSCore/VectorBuffer.swift` — add `VectorStorage` conformance:
  - Add `extension VectorBuffer: VectorStorage { public var isFloat16: Bool { false } }`
- [ ] 26.3 — Create `Sources/MetalANNSCore/Float16VectorBuffer.swift`:
  - Same API shape as `VectorBuffer`: `init(capacity:dim:device:)`, `insert(vector:at:)`, `batchInsert(vectors:startingAt:)`, `vector(at:) -> [Float]`
  - Stores `UInt16` (Float16 bit patterns) internally
  - Converts `[Float]` → Float16 on write, Float16 → `[Float]` on read
  - Conforms to `VectorStorage` with `isFloat16 == true`
  - `@unchecked Sendable`
  - Allocates `capacity * dim * MemoryLayout<UInt16>.stride` bytes
- [ ] 26.4 — **DECISION POINT (26.1)**: Float16 conversion strategy — (a) Accelerate vImage, (b) Swift `Float16` type, (c) manual bits. **Document choice in notes.**
- [ ] 26.5 — Create `Sources/MetalANNSCore/Shaders/DistanceFloat16.metal` — 3 Float16 distance kernels:
  - `cosine_distance_f16(half *query, half *corpus, float *output, uint &dim, uint &n, uint tid)`
  - `l2_distance_f16(half *query, half *corpus, float *output, uint &dim, uint &n, uint tid)`
  - `inner_product_distance_f16(half *query, half *corpus, float *output, uint &dim, uint &n, uint tid)`
  - All read `half`, accumulate in `float`, output `float`
  - Also include `compute_initial_distances_f16` (mirrors NNDescent.metal's `compute_initial_distances` but reads `half`)
- [ ] 26.6 — Create `Sources/MetalANNSCore/Shaders/SearchFloat16.metal` — Float16 beam search kernel:
  - Kernel name: `beam_search_f16`
  - Same logic as `Search.metal` but `device const half *vectors` instead of `device const float *vectors`
  - Query buffer stays `device const float *query` (users provide Float32 queries)
  - `compute_distance_f16` inline function casts `half` → `float` for accumulation
- [ ] 26.7 — Create `Tests/MetalANNSTests/Float16Tests.swift` — 2 tests using Swift Testing:
  - `float16DistanceMatchesFloat32`:
    - Create both `VectorBuffer` and `Float16VectorBuffer` with same 50 vectors of dim=64
    - Compare read-back values: diff < 0.01 per dimension
    - Compare CPU cosine distances: relative error < 5%
  - `float16RecallComparable`:
    - Build two `ANNSIndex` instances (Float32 and Float16) with 200 vectors, dim=32, degree=8
    - Search with 5 queries, k=10
    - Assert recall overlap >= 50%
- [ ] 26.8 — **DECISION POINT (26.2)**: Float16 recall threshold — (a) >= 0.80, (b) >= 0.70, (c) >= 0.50. **Document choice in notes.**
- [ ] 26.9 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/Float16Tests 2>&1 | grep -E '(PASS|FAIL|error:)'` — if tests fail to compile (Float16Tests references Float16VectorBuffer not yet created), confirms RED. If tests already compile and fail on assertions, also confirms RED.
- [ ] 26.10 — **GREEN**: All Float16Tests pass. Specifically confirm:
  - `float16DistanceMatchesFloat32` shows dimension errors < 0.01 and distance relative error < 5%
  - `float16RecallComparable` shows recall >= threshold chosen in 26.8
- [ ] 26.11 — **REGRESSION**: All Phase 1–8 tests still pass (61 prior tests + 2 new = 63 total)
- [ ] 26.12 — **GIT**: `git add Sources/MetalANNSCore/Float16VectorBuffer.swift Sources/MetalANNSCore/VectorStorage.swift Sources/MetalANNSCore/VectorBuffer.swift Sources/MetalANNSCore/Shaders/DistanceFloat16.metal Sources/MetalANNSCore/Shaders/SearchFloat16.metal Tests/MetalANNSTests/Float16Tests.swift && git commit -m "feat: add Float16 vector buffer and distance kernels"`

> **Agent notes** _(REQUIRED — document decisions 26.1 and 26.2, and any compilation issues)_:
>

---

## Task 27: Float16 Construction + Search Integration + Serialization

**Acceptance**: `Float16IntegrationTests` suite passes (2 tests), all existing tests pass, format v2 backward-compatible. Thirty-first commit.

- [ ] 27.1 — Modify `Sources/MetalANNSCore/FullGPUSearch.swift`:
  - Change `vectors: VectorBuffer` → `vectors: any VectorStorage` in `search` method signature
  - Select kernel: `let kernelName = vectors.isFloat16 ? "beam_search_f16" : "beam_search"`
  - Use `vectors.buffer` (available via protocol) for buffer binding
  - Use `vectors.dim` (available via protocol) for dimension
- [ ] 27.2 — Modify `Sources/MetalANNSCore/NNDescentGPU.swift`:
  - Change `vectors: VectorBuffer` → `vectors: any VectorStorage` in `computeInitialDistances` and `build` method signatures
  - In `computeInitialDistances`: select `"compute_initial_distances_f16"` vs `"compute_initial_distances"` based on `vectors.isFloat16`
  - In `build`: select `"local_join_f16"` vs `"local_join"` based on `vectors.isFloat16`
- [ ] 27.3 — **DECISION POINT (27.1)**: local_join Float16 shader location — (a) new `NNDescentFloat16.metal`, (b) existing `NNDescent.metal`, (c) `DistanceFloat16.metal`. Move `compute_initial_distances_f16` to chosen location too. **Document choice in notes.**
- [ ] 27.4 — Create `local_join_f16` kernel (wherever decided in 27.3):
  - Mirrors `local_join` from `NNDescent.metal` but reads `device const half *vectors`
  - `compute_metric_distance_f16` inline function casts `half` → `float` for accumulation
  - All other logic (reverse lists, try_insert_neighbor, atomic CAS) unchanged
- [ ] 27.5 — Modify `Sources/MetalANNSCore/IncrementalBuilder.swift`:
  - Change `vectors: VectorBuffer` → `vectors: any VectorStorage` in `insert` signature
  - Change `vectors: VectorBuffer` → `vectors: any VectorStorage` in `nearestNeighbors` signature
  - Existing calls to `vectors.vector(at:)`, `vectors.dim`, `vectors.capacity` work via protocol
- [ ] 27.6 — Modify `Sources/MetalANNSCore/GraphPruner.swift`:
  - Change `vectors: VectorBuffer` → `vectors: any VectorStorage` in `prune` signature
  - Existing calls to `vectors.vector(at:)`, `vectors.count` work via protocol
- [ ] 27.7 — Modify `Sources/MetalANNSCore/SearchGPU.swift`:
  - Change `vectors: VectorBuffer` → `vectors: any VectorStorage` in `search` signature
  - Change `vectors: VectorBuffer` → `vectors: any VectorStorage` in `computeDistancesOnGPU` signature
  - Select distance kernel: `vectors.isFloat16 ? "cosine_distance_f16" : "cosine_distance"` (etc.)
- [ ] 27.8 — Modify `Sources/MetalANNS/ANNSIndex.swift`:
  - Change `private var vectors: VectorBuffer?` → `private var vectors: (any VectorStorage)?`
  - In `build()`: create `Float16VectorBuffer` when `configuration.useFloat16 == true`, else `VectorBuffer`
  - Update `extractVectors(from:)` to accept `any VectorStorage`
  - In `applyLoadedState`: change parameter type to `any VectorStorage`
  - All other code works via protocol (insert, search, save)
- [ ] 27.9 — Modify `Sources/MetalANNSCore/IndexSerializer.swift`:
  - Change `version` from `1` to `2`
  - Add `storageType` field after metric in save (0=Float32, 1=Float16)
  - Save: accept `any VectorStorage`, compute byte count based on `isFloat16`
  - Load: accept both v1 (assume Float32) and v2 (read storageType)
  - Load: create `Float16VectorBuffer` when `storageType == 1`
  - Return type: `any VectorStorage` instead of `VectorBuffer`
- [ ] 27.10 — Create `Tests/MetalANNSTests/Float16IntegrationTests.swift` — 2 tests:
  - `float16FullLifecycle`:
    - Build index with `useFloat16: true`, 100 vectors, dim=32, degree=8, metric=.cosine
    - Search with k=5 — assert 5 results with non-empty IDs
    - Insert a new vector — assert count is n+1
    - Search for the new vector — assert it appears in top-5 results
  - `float16SaveLoadPreservesData`:
    - Build Float16 index with 50 vectors, dim=16, metric=.l2
    - Search before save — record top result
    - Save to temp directory
    - Load from temp directory
    - Search after load — assert top result matches
    - Cleanup temp directory
- [ ] 27.11 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/Float16IntegrationTests 2>&1 | grep -E '(PASS|FAIL|error:)'`
- [ ] 27.12 — **GREEN**: Both Float16IntegrationTests pass. Specifically confirm:
  - `float16FullLifecycle`: 5 search results, count increments, inserted vector found
  - `float16SaveLoadPreservesData`: top result preserved across save/load
- [ ] 27.13 — **REGRESSION**: All prior tests still pass (63 from Task 26 + 2 new = 65 total)
- [ ] 27.14 — **GIT**: Commit all modified and new files with message `"feat: integrate Float16 into construction, search, and persistence"`

> **Agent notes** _(REQUIRED — document decision 27.1, any Sendable/existential boxing issues, and backward compatibility verification)_:
>

---

## Phase 9 Complete — Signal

When all items above are checked, update this section:

```
STATUS: [pending|complete]
FINAL TEST RESULT: [PASS|FAIL] — `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'` (N/N)
TOTAL COMMITS: [number] (expected: 31)
TOTAL TESTS: [number] (expected: 65 = 61 prior + 2 from Task 26 + 2 from Task 27)
ISSUES ENCOUNTERED:
- [list any issues]
DECISIONS MADE:
- 26.1: [Float16 conversion strategy chosen]
- 26.2: [recall threshold chosen]
- 27.1: [local_join_f16 location chosen]
PHASE 1–8 FILES MODIFIED:
- [list all previously-existing files that were changed]
```

---

## Orchestrator Review Checklist (DO NOT MODIFY — Orchestrator use only)

- [ ] R1 — Git log shows exactly 31 commits with correct conventional commit messages
- [ ] R2 — Full test suite passes: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'` — zero failures including all Phase 1–8 tests
- [ ] R3 — `VectorStorage` protocol exists in `MetalANNSCore` target with all required properties/methods
- [ ] R4 — `VectorBuffer` conforms to `VectorStorage` with `isFloat16 == false`
- [ ] R5 — `Float16VectorBuffer` exists, conforms to `VectorStorage` with `isFloat16 == true`
- [ ] R6 — `Float16VectorBuffer` stores `UInt16` (half-precision bit patterns), NOT `Float`
- [ ] R7 — `Float16VectorBuffer` converts `[Float]` → Float16 on write and Float16 → `[Float]` on read
- [ ] R8 — `DistanceFloat16.metal` contains `cosine_distance_f16`, `l2_distance_f16`, `inner_product_distance_f16`
- [ ] R9 — All Float16 distance kernels accumulate in `float`, only storage is `half`
- [ ] R10 — `SearchFloat16.metal` contains `beam_search_f16` kernel
- [ ] R11 — `beam_search_f16` reads `device const half *vectors` but `device const float *query` (query stays Float32)
- [ ] R12 — `FullGPUSearch.search` accepts `VectorStorage` (or `any VectorStorage`) and dispatches to correct kernel
- [ ] R13 — `NNDescentGPU.build` accepts `VectorStorage` and dispatches to correct compute/local_join kernel
- [ ] R14 — `local_join_f16` kernel exists and mirrors `local_join` logic with `half` vector reads
- [ ] R15 — `compute_initial_distances_f16` kernel exists and mirrors `compute_initial_distances` with `half` vector reads
- [ ] R16 — `IncrementalBuilder.insert` accepts `VectorStorage`
- [ ] R17 — `GraphPruner.prune` accepts `VectorStorage`
- [ ] R18 — `SearchGPU.search` accepts `VectorStorage` and dispatches to correct distance kernel
- [ ] R19 — `ANNSIndex.vectors` property uses `VectorStorage` (or `any VectorStorage`)
- [ ] R20 — `ANNSIndex.build` creates `Float16VectorBuffer` when `useFloat16 == true`
- [ ] R21 — `IndexSerializer` format version is 2
- [ ] R22 — `IndexSerializer.save` writes `storageType` field (0=Float32, 1=Float16)
- [ ] R23 — `IndexSerializer.load` reads v1 files (backward compatible, assumes Float32)
- [ ] R24 — `IndexSerializer.load` reads v2 files and creates correct buffer type
- [ ] R25 — Float16 distance test verifies precision within acceptable bounds
- [ ] R26 — Float16 recall test has documented threshold and rationale
- [ ] R27 — Float16 full lifecycle test covers build → search → insert → search
- [ ] R28 — Float16 save/load test verifies round-trip data preservation
- [ ] R29 — No `import XCTest` or `XCTSkip` anywhere
- [ ] R30 — Agent notes filled in for decisions 26.1, 26.2, and 27.1
- [ ] R31 — Total test count is at least 65 (61 prior + 4 new)
- [ ] R32 — New .metal files are picked up by `.process("Shaders")` in Package.swift
