# MetalANNS Wax-Readiness — Task Tracker

> **Plan:** `docs/plans/2026-02-28-metalanns-wax-readiness.md`
> **Goal:** Fix all blockers so MetalANNS replaces USearch as Wax's sole vector search backend.
> **Status:** COMPLETE
> **Last Updated:** 2026-03-03

---

## Phase 1: Buffer Pool for GPU Search

> **Prompt:** `docs/prompts/phase-1-buffer-pool.md`
> **Branch:** TBD
> **Depends on:** Nothing (start here)
> **Note:** Shader compile blockers were fixed (`UINT_MAX` and `memory_order_relaxed`), and full test validation now passes.

- [x] **1.1 Create SearchBufferPool**
  - [x] Write 3 failing tests in `Tests/MetalANNSTests/SearchBufferPoolTests.swift`
  - [x] Verify tests fail (compilation error — type does not exist)
  - [x] Implement `Sources/MetalANNSCore/SearchBufferPool.swift`
  - [x] Verify 3 tests pass
  - [x] Commit: `feat: add SearchBufferPool to eliminate per-search MTLBuffer allocation`

- [x] **1.2 Wire pool into FullGPUSearch**
  - [x] Write safety test `fullGPUSearchCorrectAfterPoolRefactor` in `SearchBufferPoolTests.swift`
  - [x] Verify safety test passes BEFORE refactoring (baseline)
  - [x] Add `searchBufferPool` property to `MetalContext` in `MetalDevice.swift`
  - [x] Replace `device.makeBuffer` calls in `FullGPUSearch.swift:58-74` with pool acquire/release
  - [x] Verify full test suite passes (zero regressions)
  - [x] Commit: `refactor: wire SearchBufferPool into FullGPUSearch, eliminating per-search allocation`

**Phase 1 exit criteria:**
- [x] `FullGPUSearch.search()` has zero `device.makeBuffer` calls
- [x] `MetalContext` exposes `searchBufferPool`
- [x] 4 tests in `SearchBufferPoolTests` (3 unit + 1 integration)
- [x] Full suite green

---

## Phase 2: Lift 4096-Node GPU Search Ceiling

> **Prompt:** `docs/prompts/phase-2-gpu-ceiling.md`
> **Branch:** TBD
> **Depends on:** Phase 1 (uses SearchBufferPool for visited buffer)

- [x] **2.1 Verification tests for >4096 node search**
  - [x] Write `searchAbove4096NodesReturnsResults` test in `Tests/MetalANNSTests/FullGPUSearchTests.swift`
  - [x] Write `gpuSearchMatchesCPUAtSmallScale` parity test
  - [x] Run targeted tests with `xcodebuild` (blocked by pre-existing `NNDescent*.metal` compile errors: `uint_max`, `memory_order_release`)
  - [x] Commit: `test: verify GPU search works above 4096-node limit and matches CPU recall`

- [x] **2.2 Add visited buffer pooling to SearchBufferPool**
  - [x] Add failing tests for `acquireVisited`/`releaseVisited`
  - [x] Implement `VisitedBuffers`, `acquireVisited(nodeCount:)`, `releaseVisited(_:capacity:)`
  - [x] Verify visited-pool tests run attempt (blocked by pre-existing `NNDescent*.metal` compile errors: `uint_max`, `memory_order_release`)
  - [x] Commit: `feat: add visited-buffer pooling with generation counter to SearchBufferPool`

- [x] **2.3 Wire visited pooling into FullGPUSearch**
  - [x] Replace per-search visited alloc+memset with pool acquire/release
  - [x] Verify no `.initialize(repeating: 0)` calls remain in `FullGPUSearch.swift`
  - [x] Verify full suite pass attempt (blocked by same pre-existing `NNDescent*.metal` compile errors)
  - [x] Commit: `perf: replace per-search visited-buffer alloc+memset with generation-counter pool`

**Phase 2 exit criteria:**
- [x] GPU search works at 5000+ nodes (test proves it)
- [x] GPU results match CPU reference (recall >= 0.7)
- [x] Both FP32 and FP16 kernels updated
- [x] Full suite green

---

## Phase 3: Fix StreamingIndex Unbounded Memory Growth

> **Prompt:** `docs/prompts/phase-3-streaming-memory.md` (to be written)
> **Branch:** TBD
> **Depends on:** Nothing (can run in parallel with Phase 1-2)

- [x] **3.1 Evict merged/deleted vectors**
  - [x] Write `mergedVectorsAreEvictedFromHistory` test
  - [x] Write `deletedVectorsAreRemovedFromHistory` test
  - [x] Verify first test fails (meta file too large)
  - [x] Clear `allVectorsList`/`allIDsList` after merge completes (keep only post-merge pending)
  - [x] Remove deleted vectors from history in `delete(id:)`
  - [x] Verify both tests pass
  - [x] Verify full suite passes
  - [x] Commit: `fix: evict merged and deleted vectors from StreamingIndex history to prevent OOM`

**Phase 3 exit criteria:**
- [x] Meta file size < 50KB after merge of 15 small vectors
- [x] Deleted vectors not retained in history
- [x] Save/load round-trip still correct
- [x] Full suite green

---

## Phase 4: Native UInt64 ID Support

> **Prompt:** `docs/prompts/phase-4-uint64-ids.md` (to be written)
> **Branch:** TBD
> **Depends on:** Nothing (can run in parallel with Phase 1-3)

- [x] **4.1 Add UInt64 keys to IDMap**
  - [x] Write 4 tests in `Tests/MetalANNSTests/IDMapTests.swift`
  - [x] Verify tests fail
  - [x] Add `numericToInternal`/`internalToNumeric` dictionaries to `IDMap`
  - [x] Add `assign(numericID:)`, `internalID(forNumeric:)`, `numericID(for:)` methods
  - [x] Update `Codable` conformance to encode new dictionaries
  - [x] Verify tests pass
  - [x] Commit: `feat: add native UInt64 key support to IDMap for Wax frameId compatibility`

- [x] **4.2 Add UInt64 insert/search to ANNSIndex**
  - [x] Write `insertAndSearchWithUInt64IDs` test
  - [x] Verify test fails
  - [x] Add `insert(_:numericID:)` to ANNSIndex
  - [x] Add `numericID: UInt64?` field to SearchResult
  - [x] Wire IDMap numeric lookups into search result mapping
  - [x] Verify test passes
  - [x] Verify full suite passes
  - [x] Commit: `feat: add UInt64-keyed insert and search to ANNSIndex for Wax integration`

**Phase 4 exit criteria:**
- [x] `IDMap` supports both String and UInt64 keys independently
- [x] `ANNSIndex.insert(_:numericID:)` works end-to-end
- [x] `SearchResult.numericID` populated when numeric key was used
- [x] Full suite green

---

## Phase 5: GPU-vs-CPU Search Parity Tests

> **Prompt:** `docs/prompts/phase-5-parity-tests.md` (to be written)
> **Branch:** TBD
> **Depends on:** Phase 2 (validates the new visited set)

- [x] **5.1 Parameterized parity test at multiple scales**
  - [x] Create `Tests/MetalANNSTests/GPUCPUParityTests.swift`
  - [x] Parameterized over: (100, dim=32), (500, dim=64), (2000, dim=128), (8000, dim=384)
  - [x] Each config: build graph, run 5 queries through GPU and CPU, compare recall >= 0.6
  - [x] All use `SeededGenerator` for reproducibility
  - [x] Verify all pass
  - [x] Commit: `test: add GPU-vs-CPU search parity tests at multiple scales`

**Phase 5 exit criteria:**
- [x] 4 parameterized configs × 5 queries = 20 parity checks
- [x] All recall >= 0.6
- [x] Tests are seeded and reproducible

---

## Phase 6: Algorithmic Optimizations and Bug Fixes

> **Prompt:** `docs/prompts/phase-6-optimizations.md` (to be written)
> **Branch:** TBD
> **Depends on:** Phase 5 for Task 6.3 (uses parity tests to verify). Tasks 6.1, 6.2, 6.4 are independent.

- [x] **6.1 Fix rangeSearch guard inconsistency**
  - [x] Write `rangeSearchWithZeroDistanceReturnsExactMatches` test
  - [x] Verify test fails
  - [x] Change `StreamingIndex.swift:193` from `> 0` to `>= 0`
  - [x] Verify test passes
  - [x] Commit: `fix: allow maxDistance=0 in StreamingIndex.rangeSearch for exact match queries`

- [x] **6.2 Seed all test RNG**
  - [x] Create `Tests/MetalANNSTests/TestUtilities.swift` with shared `SeededGenerator`
  - [x] Replace all unseeded `Float.random(in:)` across test files
  - [x] Verify full suite passes
  - [x] Commit: `test: use SeededGenerator across all tests for reproducible results`

- [x] **6.3 Early-exit + symmetric updates in local_join**
  - [x] Read worst distance before computing pair distance
  - [x] Skip `try_insert_neighbor` when `pair_dist >= worst`
  - [x] Add symmetric update: insert `a` into `b`'s list too
  - [x] Apply same changes to `NNDescentFloat16.metal`
  - [x] Verify NNDescentGPU tests pass with recall >= 0.80
  - [x] Verify GPU-CPU parity tests still pass
  - [x] Commit: `perf: add early-exit and symmetric updates to local_join kernel`

- [x] **6.4 PQ threadgroup memory guard**
  - [x] Add guard `tableLengthBytes <= device.maxThreadgroupMemoryLength` in `GPUADCSearch.swift`
  - [x] Verify GPUADCSearch tests pass
  - [x] Commit: `fix: guard against PQ distance table exceeding threadgroup memory limit`

**Phase 6 exit criteria:**
- [x] `StreamingIndex.rangeSearch(maxDistance: 0)` returns results
- [x] Zero unseeded `Float.random` calls in test target
- [x] `local_join` does early-exit + symmetric inserts
- [x] PQ ADC scan fails gracefully on oversized tables
- [x] Full suite green

---

## Final Validation

- [x] All 6 phases complete
- [x] Full suite green (required command): `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS,arch=arm64' 2>&1 | grep -E "(Test Suite|passed|failed|error:)" | tail -40`
- [x] xcresult summary confirms zero failures: `result == Passed`, `failedTests == 0`, `passedTests == 265`, `totalTestCount == 265`
- [x] Final tail validation includes `** TEST SUCCEEDED **`: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS,arch=arm64' 2>&1 | tail -20`
- [x] No force-unwraps introduced
- [x] No new compiler warnings
- [x] All commits on branch, ready for review
