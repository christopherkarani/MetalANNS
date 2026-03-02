# MetalANNS Wax-Readiness — Task Tracker

> **Plan:** `docs/plans/2026-02-28-metalanns-wax-readiness.md`
> **Goal:** Fix all blockers so MetalANNS replaces USearch as Wax's sole vector search backend.
> **Status:** NOT STARTED
> **Last Updated:** 2026-02-28

---

## Phase 1: Buffer Pool for GPU Search

> **Prompt:** `docs/prompts/phase-1-buffer-pool.md`
> **Branch:** TBD
> **Depends on:** Nothing (start here)
> **Note:** `xcodebuild test` is currently blocked by pre-existing shader compile errors in `NNDescent.metal` and `NNDescentFloat16.metal` (`uint_max`, `memory_order_release`).

- [ ] **1.1 Create SearchBufferPool**
  - [x] Write 3 failing tests in `Tests/MetalANNSTests/SearchBufferPoolTests.swift`
  - [x] Verify tests fail (compilation error — type does not exist)
  - [x] Implement `Sources/MetalANNSCore/SearchBufferPool.swift`
  - [ ] Verify 3 tests pass
  - [x] Commit: `feat: add SearchBufferPool to eliminate per-search MTLBuffer allocation`

- [ ] **1.2 Wire pool into FullGPUSearch**
  - [x] Write safety test `fullGPUSearchCorrectAfterPoolRefactor` in `SearchBufferPoolTests.swift`
  - [ ] Verify safety test passes BEFORE refactoring (baseline)
  - [x] Add `searchBufferPool` property to `MetalContext` in `MetalDevice.swift`
  - [x] Replace `device.makeBuffer` calls in `FullGPUSearch.swift:58-74` with pool acquire/release
  - [ ] Verify full test suite passes (zero regressions)
  - [ ] Commit: `refactor: wire SearchBufferPool into FullGPUSearch, eliminating per-search allocation`

**Phase 1 exit criteria:**
- [ ] `FullGPUSearch.search()` has zero `device.makeBuffer` calls
- [ ] `MetalContext` exposes `searchBufferPool`
- [ ] 4 tests in `SearchBufferPoolTests` (3 unit + 1 integration)
- [ ] Full suite green

---

## Phase 2: Lift 4096-Node GPU Search Ceiling

> **Prompt:** `docs/prompts/phase-2-visited-set.md` (to be written)
> **Branch:** TBD
> **Depends on:** Phase 1 (uses SearchBufferPool for visited buffer)

- [ ] **2.1 Generation-counter visited set**
  - [ ] Write `searchAbove4096NodesReturnsResults` test in `Tests/MetalANNSTests/FullGPUSearchTests.swift`
  - [ ] Write `gpuSearchMatchesCPUAtSmallScale` parity test
  - [ ] Verify `searchAbove4096Nodes` fails with "exceeds visited-table capacity"
  - [ ] Replace `try_visit` hash table in `Search.metal` with device-memory generation counter
  - [ ] Update `beam_search` kernel signature to accept visited buffer + generation
  - [ ] Add visited buffer acquire/release to `SearchBufferPool`
  - [ ] Remove `maxVisited` guard in `FullGPUSearch.swift:43-47`
  - [ ] Verify both tests pass
  - [ ] Verify full suite passes
  - [ ] Commit: `feat: lift 4096-node GPU search ceiling using generation-counter visited set`

- [ ] **2.2 Float16 beam search update**
  - [ ] Apply identical visited-set changes to `SearchFloat16.metal`
  - [ ] Add Float16 variant of parity test
  - [ ] Verify full suite passes
  - [ ] Commit: `feat: lift 4096-node ceiling for Float16 beam search kernel`

**Phase 2 exit criteria:**
- [ ] GPU search works at 5000+ nodes (test proves it)
- [ ] GPU results match CPU reference (recall >= 0.7)
- [ ] Both FP32 and FP16 kernels updated
- [ ] Full suite green

---

## Phase 3: Fix StreamingIndex Unbounded Memory Growth

> **Prompt:** `docs/prompts/phase-3-streaming-memory.md` (to be written)
> **Branch:** TBD
> **Depends on:** Nothing (can run in parallel with Phase 1-2)

- [ ] **3.1 Evict merged/deleted vectors**
  - [ ] Write `mergedVectorsAreEvictedFromHistory` test
  - [ ] Write `deletedVectorsAreRemovedFromHistory` test
  - [ ] Verify first test fails (meta file too large)
  - [ ] Clear `allVectorsList`/`allIDsList` after merge completes (keep only post-merge pending)
  - [ ] Remove deleted vectors from history in `delete(id:)`
  - [ ] Verify both tests pass
  - [ ] Verify full suite passes
  - [ ] Commit: `fix: evict merged and deleted vectors from StreamingIndex history to prevent OOM`

**Phase 3 exit criteria:**
- [ ] Meta file size < 50KB after merge of 15 small vectors
- [ ] Deleted vectors not retained in history
- [ ] Save/load round-trip still correct
- [ ] Full suite green

---

## Phase 4: Native UInt64 ID Support

> **Prompt:** `docs/prompts/phase-4-uint64-ids.md` (to be written)
> **Branch:** TBD
> **Depends on:** Nothing (can run in parallel with Phase 1-3)

- [ ] **4.1 Add UInt64 keys to IDMap**
  - [ ] Write 4 tests in `Tests/MetalANNSTests/IDMapTests.swift`
  - [ ] Verify tests fail
  - [ ] Add `numericToInternal`/`internalToNumeric` dictionaries to `IDMap`
  - [ ] Add `assign(numericID:)`, `internalID(forNumeric:)`, `numericID(for:)` methods
  - [ ] Update `Codable` conformance to encode new dictionaries
  - [ ] Verify tests pass
  - [ ] Commit: `feat: add native UInt64 key support to IDMap for Wax frameId compatibility`

- [ ] **4.2 Add UInt64 insert/search to ANNSIndex**
  - [ ] Write `insertAndSearchWithUInt64IDs` test
  - [ ] Verify test fails
  - [ ] Add `insert(_:numericID:)` to ANNSIndex
  - [ ] Add `numericID: UInt64?` field to SearchResult
  - [ ] Wire IDMap numeric lookups into search result mapping
  - [ ] Verify test passes
  - [ ] Verify full suite passes
  - [ ] Commit: `feat: add UInt64-keyed insert and search to ANNSIndex for Wax integration`

**Phase 4 exit criteria:**
- [ ] `IDMap` supports both String and UInt64 keys independently
- [ ] `ANNSIndex.insert(_:numericID:)` works end-to-end
- [ ] `SearchResult.numericID` populated when numeric key was used
- [ ] Full suite green

---

## Phase 5: GPU-vs-CPU Search Parity Tests

> **Prompt:** `docs/prompts/phase-5-parity-tests.md` (to be written)
> **Branch:** TBD
> **Depends on:** Phase 2 (validates the new visited set)

- [ ] **5.1 Parameterized parity test at multiple scales**
  - [ ] Create `Tests/MetalANNSTests/GPUCPUParityTests.swift`
  - [ ] Parameterized over: (100, dim=32), (500, dim=64), (2000, dim=128), (8000, dim=384)
  - [ ] Each config: build graph, run 5 queries through GPU and CPU, compare recall >= 0.6
  - [ ] All use `SeededGenerator` for reproducibility
  - [ ] Verify all pass
  - [ ] Commit: `test: add GPU-vs-CPU search parity tests at multiple scales`

**Phase 5 exit criteria:**
- [ ] 4 parameterized configs × 5 queries = 20 parity checks
- [ ] All recall >= 0.6
- [ ] Tests are seeded and reproducible

---

## Phase 6: Algorithmic Optimizations and Bug Fixes

> **Prompt:** `docs/prompts/phase-6-optimizations.md` (to be written)
> **Branch:** TBD
> **Depends on:** Phase 5 for Task 6.3 (uses parity tests to verify). Tasks 6.1, 6.2, 6.4 are independent.

- [ ] **6.1 Fix rangeSearch guard inconsistency**
  - [ ] Write `rangeSearchWithZeroDistanceReturnsExactMatches` test
  - [ ] Verify test fails
  - [ ] Change `StreamingIndex.swift:193` from `> 0` to `>= 0`
  - [ ] Verify test passes
  - [ ] Commit: `fix: allow maxDistance=0 in StreamingIndex.rangeSearch for exact match queries`

- [ ] **6.2 Seed all test RNG**
  - [ ] Create `Tests/MetalANNSTests/TestUtilities.swift` with shared `SeededGenerator`
  - [ ] Replace all unseeded `Float.random(in:)` across test files
  - [ ] Verify full suite passes
  - [ ] Commit: `test: use SeededGenerator across all tests for reproducible results`

- [ ] **6.3 Early-exit + symmetric updates in local_join**
  - [ ] Read worst distance before computing pair distance
  - [ ] Skip `try_insert_neighbor` when `pair_dist >= worst`
  - [ ] Add symmetric update: insert `a` into `b`'s list too
  - [ ] Apply same changes to `NNDescentFloat16.metal`
  - [ ] Verify NNDescentGPU tests pass with recall >= 0.80
  - [ ] Verify GPU-CPU parity tests still pass
  - [ ] Commit: `perf: add early-exit and symmetric updates to local_join kernel`

- [ ] **6.4 PQ threadgroup memory guard**
  - [ ] Add guard `tableLengthBytes <= device.maxThreadgroupMemoryLength` in `GPUADCSearch.swift`
  - [ ] Verify GPUADCSearch tests pass
  - [ ] Commit: `fix: guard against PQ distance table exceeding threadgroup memory limit`

**Phase 6 exit criteria:**
- [ ] `StreamingIndex.rangeSearch(maxDistance: 0)` returns results
- [ ] Zero unseeded `Float.random` calls in test target
- [ ] `local_join` does early-exit + symmetric inserts
- [ ] PQ ADC scan fails gracefully on oversized tables
- [ ] Full suite green

---

## Final Validation

- [ ] All 6 phases complete
- [ ] Full test suite passes: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
- [ ] No force-unwraps introduced
- [ ] No new compiler warnings
- [ ] All commits on branch, ready for review
