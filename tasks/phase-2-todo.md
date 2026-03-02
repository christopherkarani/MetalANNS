# Phase 2: Validate GPU Search at Scale + Buffer Pooling — Task Tracker

> **Prompt:** `docs/prompts/phase-2-gpu-search-validation.md`
> **Plan:** `docs/plans/2026-02-28-metalanns-wax-readiness.md`
> **Status:** NOT STARTED
> **Last Updated:** 2026-03-02
> **Depends on:** Phase 1 (SearchBufferPool must exist first)

## Current State

The generation-counter visited set is **already implemented** in both kernels:
- `Search.metal` — `try_visit_global()` with device-memory `visited_generation` buffer at index 12
- `SearchFloat16.metal` — `try_visit_global_f16()` with same pattern at index 12
- `FullGPUSearch.swift` — allocates `visitedGenerationBuffer`, binds at indices 12-13, no nodeCount cap
- Old `MAX_VISITED = 4096` constant is **removed**

**What's missing:**
- No test proves search works above 4096 nodes
- No test validates GPU results match CPU reference
- `visitedGenerationBuffer` is allocated fresh per-search (not pooled) — `FullGPUSearch.swift:67-70`
- `queryBuffer`, `outputDistanceBuffer`, `outputIDBuffer` also allocated per-search — `FullGPUSearch.swift:53-66`

---

## Task 2.1: Pool the visited buffer alongside search buffers

- [ ] Extend `SearchBufferPool.Buffers` to include `visitedBuffer: MTLBuffer` and `visitedCapacity: Int`
- [ ] Update `SearchBufferPool.acquire()` to accept `nodeCount` parameter
- [ ] Update `SearchBufferPool.allocate()` to create visited buffer sized to nodeCount
- [ ] Add pool acquire matching: `visitedCapacity >= nodeCount`
- [ ] Add generation counter to pool (increment on each acquire, return with buffers)
- [ ] Write test: `acquireIncludesVisitedBuffer` — validates visited buffer exists and has correct size
- [ ] Write test: `generationCounterIncrements` — validates generation is unique per acquire
- [ ] Verify tests pass
- [ ] Commit: `feat: extend SearchBufferPool with visited buffer and generation counter`

## Task 2.2: Wire pooled visited buffer into FullGPUSearch

- [ ] Replace `visitedGenerationBuffer` allocation at `FullGPUSearch.swift:67-77` with pool acquire
- [ ] Replace `queryBuffer`/output buffer allocation at `FullGPUSearch.swift:53-66` with pool acquire
- [ ] Use pool's generation counter instead of hardcoded `visitGenerationValue = 1`
- [ ] Single `defer { pool.release(buffers) }` for all buffers
- [ ] Write test: `searchCorrectAfterPooledVisitedBuffer` — build + search at small scale, validate results
- [ ] Verify full suite passes
- [ ] Commit: `refactor: use pooled visited buffer in FullGPUSearch, zero per-search allocations`

## Task 2.3: Test GPU search above 4096 nodes

- [ ] Create `Tests/MetalANNSTests/FullGPUSearchTests.swift`
- [ ] Add `SeededGenerator` (or import from shared `TestUtilities.swift` if Phase 6.2 is done)
- [ ] Write test: `searchAt5000NodesReturnsResults`
  - [ ] Build graph with 5000 nodes, dim=32, degree=16
  - [ ] Search for a vector that's in the index
  - [ ] Assert result count == k
  - [ ] Assert first result has score < 0.01 (near-exact match)
- [ ] Write test: `searchAt10000NodesReturnsResults`
  - [ ] Build graph with 10000 nodes, dim=32, degree=16
  - [ ] Same assertions
- [ ] Verify both tests pass
- [ ] Commit: `test: validate GPU search at 5k and 10k nodes (above old 4096 ceiling)`

## Task 2.4: GPU-vs-CPU search parity test

- [ ] Create `Tests/MetalANNSTests/GPUCPUParityTests.swift`
- [ ] Write parameterized test over 4 configs:
  - [ ] (nodeCount=100, dim=32, degree=8, k=5, ef=32)
  - [ ] (nodeCount=500, dim=64, degree=16, k=10, ef=64)
  - [ ] (nodeCount=2000, dim=128, degree=32, k=20, ef=128)
  - [ ] (nodeCount=8000, dim=384, degree=32, k=10, ef=64)
- [ ] For each config: build graph, run 5 queries through both `FullGPUSearch` and `BeamSearchCPU`
- [ ] Compare top-k results — overlap (recall) must be >= 0.6
- [ ] All vectors generated with `SeededGenerator` for reproducibility
- [ ] Verify all pass
- [ ] Commit: `test: add GPU-vs-CPU search parity tests at multiple scales`

---

## Phase 2 Exit Criteria

- [ ] `SearchBufferPool` includes visited buffer + generation counter
- [ ] `FullGPUSearch.search()` has zero `device.makeBuffer` calls
- [ ] GPU search passes at 5000 and 10000 nodes
- [ ] GPU-CPU recall >= 0.6 at all 4 test scales
- [ ] All tests use `SeededGenerator`
- [ ] Full test suite green: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
- [ ] 4 commits on branch
