# MetalANNS — Phase 8: Full GPU Search & Graph Pruning

> **Status**: PENDING
> **Owner**: Subagent (dispatched by orchestrator)
> **Reviewer**: Orchestrator (main session)
> **Last Updated**: 2026-02-25 11:37

---

## How This File Works

This file is the **shared communication layer** between the orchestrator and executing agents.

**Executing agent**: Check off items `[x]` as you complete them. Add notes under any item if you hit issues. Update `Last Updated` timestamp after each task. Do NOT check off an item unless the verification step passes.

**Orchestrator**: Reviews checked items against actual codebase state. Unchecks items that don't pass review. Adds review comments prefixed with `> [REVIEW]`.

---

## Pre-Flight Checks

- [x] Working directory is `/Users/chriskarani/CodingProjects/MetalANNS`
- [x] Phase 1–7 code exists: `ANNSIndex.swift` (actor), `BeamSearchCPU.swift`, `SearchGPU.swift`, `SIMDDistance.swift`, `IncrementalBuilder.swift`, `IndexSerializer.swift`, `SoftDeletion.swift`, `NNDescentCPU.swift`, `NNDescentGPU.swift`, `Distance.metal`, `NNDescent.metal`, `Sort.metal` all present
- [x] `git log --oneline | wc -l` baseline verified (expect 27 commits before Phase 8 execution)
- [x] Full test suite passes (57 tests, zero failures): `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
- [x] Read implementation plan: `docs/plans/2026-02-25-metalanns-v2-performance-features.md` (Phase 8 section, Tasks 24–25)
- [x] Read this phase's prompt: `docs/prompts/phase-8-gpu-search.md` (detailed spec for Tasks 24–25)

---

## Task 24: Full GPU Beam Search Kernel

**Acceptance**: `FullGPUSearchTests` suite passes (2 tests), `ANNSIndex` GPU path uses `FullGPUSearch`. Twenty-eighth commit.

- [x] 24.1 — Create `Tests/MetalANNSTests/FullGPUSearchTests.swift` — 2 tests using Swift Testing:
  - `gpuSearchReturnsK`:
    - Guard: `MTLCreateSystemDefaultDevice() != nil else { return }`
    - Build small graph: 200 nodes, dim=16, degree=8 via `NNDescentCPU.build`
    - Populate `VectorBuffer` + `GraphBuffer` from CPU graph (inline helper, pad to degree)
    - Query: random 16-dim vector
    - `FullGPUSearch.search(context:query:vectors:graph:entryPoint:k:5 ef:32 metric:.cosine)`
    - Assert: `results.count == 5`
    - Assert: results sorted by score ascending
  - `gpuSearchRecallMatchesHybrid`:
    - Guard: `MTLCreateSystemDefaultDevice() != nil else { return }`
    - Build graph: 500 nodes, dim=32, degree=16 via `NNDescentCPU.build`
    - 20 random queries, k=10, ef=64
    - For each query: brute-force ground truth via `SIMDDistance.cosine`
    - Compute recall for `FullGPUSearch` and `SearchGPU` (existing hybrid)
    - Assert: `fullGPURecall > hybridRecall - 0.05`
    - Assert: `fullGPURecall > 0.80`
- [x] 24.2 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/FullGPUSearchTests 2>&1 | grep -E '(PASS|FAIL|error:)'` → confirms FAIL (FullGPUSearch not found)
- [x] 24.3 — Create `Sources/MetalANNSCore/Shaders/Search.metal`:
  - Constants: `MAX_EF = 256`, `MAX_VISITED = 4096`
  - `struct CandidateEntry { uint nodeID; float distance; }`
  - Inline `compute_distance()` — cosine (metric=0), L2 (metric=1), innerProduct (metric=2)
  - Inline `try_visit()` — open-addressed hash insert with linear probing (up to 32 probes), `atomic_compare_exchange_weak_explicit` with `memory_order_relaxed`
  - Kernel `beam_search` — 12 buffer bindings (vectors, adjacency, query, output_dists, output_ids, node_count, degree, dim, k, ef, entry_point, metric_type)
  - Shared memory: `candidates[MAX_EF]`, `results[MAX_EF]`, `visited[MAX_VISITED]` (atomic_uint), `candidate_count`, `result_count`, `candidate_head`
  - Main loop: thread 0 pops candidate, all threads expand neighbors in parallel, thread 0 sorts and trims
  - Output: thread 0 writes top-k to output buffers, pads with sentinels
- [x] 24.4 — Create `Sources/MetalANNSCore/FullGPUSearch.swift`:
  - `public enum FullGPUSearch` — stateless, all static methods
  - `search(context:query:vectors:graph:entryPoint:k:ef:metric:)` → `[SearchResult]`
  - Gets `beam_search` pipeline from cache
  - Creates query buffer, output dist/id buffers
  - Packs scalar constants (nodeCount, degree, dim, k, ef clamped to 256, entryPoint, metricType)
  - Dispatches: 1 threadgroup, `min(degree, maxThreads)` threads per threadgroup (or larger — see decision 24.1)
  - Reads output buffers, constructs `[SearchResult]`, skips sentinels
- [x] 24.5 — **GREEN**: Both FullGPUSearch tests pass. Specifically confirm:
  - `gpuSearchReturnsK` returns 5 results, sorted by score
  - `gpuSearchRecallMatchesHybrid` shows recall > 0.80 and within 5% of hybrid
- [x] 24.6 — **DECISION POINT (24.1)**: Threadgroup size. Options: (a) `min(degree, maxThreads)` — matches neighbor count, (b) fixed 64, (c) fixed 128. **Document choice and reasoning in notes.**
- [x] 24.7 — **DECISION POINT (24.2)**: Sorting approach. Options: (a) thread-0 insertion sort per iteration, (b) parallel merge sort, (c) sort only at output. **Document approach in notes.**
- [x] 24.8 — **DECISION POINT (24.3)**: Confirm `SearchGPU.swift` is kept (not deleted) as fallback. **Document in notes.**
- [x] 24.9 — **DECISION POINT (24.4)**: If shared memory adjustments were needed (`MAX_EF` or `MAX_VISITED` changes), document them. If no changes needed, confirm defaults worked.
- [x] 24.10 — Wire `FullGPUSearch` into `ANNSIndex.swift`:
  - In the `search` method, replace `SearchGPU.search(...)` with `FullGPUSearch.search(...)` in the GPU path
  - Keep CPU fallback (`BeamSearchCPU`) unchanged
- [x] 24.11 — **REGRESSION**: All Phase 1–7 tests still pass (57 prior tests + 2 new = 59 total)
- [x] 24.12 — **GIT**: `git add Sources/MetalANNSCore/Shaders/Search.metal Sources/MetalANNSCore/FullGPUSearch.swift Tests/MetalANNSTests/FullGPUSearchTests.swift Sources/MetalANNS/ANNSIndex.swift && git commit -m "feat: full GPU beam search kernel with shared-memory candidate queue"`

> **Agent notes** _(REQUIRED — document decisions 24.1, 24.2, 24.3, 24.4, and any Phase 1–7 file modifications)_:
>
> - 24.1 Threadgroup size choice: used `min(degree, pipeline.maxTotalThreadsPerThreadgroup)` (implemented in `FullGPUSearch.swift`). Degree-bounded neighbor expansion means fixed 64/128 would mostly idle and add barrier overhead.
> - 24.2 Sorting choice: kept thread-0 insertion sort each iteration for both `results` and active candidate range. With `MAX_EF=256`, this stayed simple and met correctness/recall targets in tests.
> - 24.3 `SearchGPU.swift` retained and untouched as fallback/backward-compatible path.
> - 24.4 Shared memory constants stayed at defaults (`MAX_EF=256`, `MAX_VISITED=4096`); kernel compiled and tests passed without reducing either.
> - Phase 1–7 file modifications: `ANNSIndex.swift` GPU search path switched from `SearchGPU.search` to `FullGPUSearch.search` only.

---

## Task 25: CAGRA Post-Processing (Graph Pruning)

**Acceptance**: `GraphPrunerTests` suite passes (2 tests), `ANNSIndex.build()` calls pruning. Twenty-ninth commit.

- [x] 25.1 — Create `Tests/MetalANNSTests/GraphPrunerTests.swift` — 2 tests using Swift Testing:
  - `pruningReducesRedundancy`:
    - Build graph: 200 nodes, dim=16, degree=8 via `NNDescentCPU.build`
    - Populate `VectorBuffer` + `GraphBuffer`
    - Count total valid (non-sentinel) edges before pruning
    - Call `GraphPruner.prune(graph:vectors:nodeCount:metric:)`
    - Count total valid edges after pruning
    - Assert: `edgesAfter <= edgesBefore` (only removes edges)
    - Assert: `edgesAfter > 0` (not completely emptied)
    - Assert: average neighbors per node > `degree / 2` (not over-pruned)
  - `pruningMaintainsRecall`:
    - Guard: `MTLCreateSystemDefaultDevice() != nil else { return }`
    - Build graph: 300 nodes, dim=32, degree=16 via `NNDescentCPU.build`
    - Populate `VectorBuffer` + `GraphBuffer`
    - Measure baseline recall: 20 queries, k=10, ef=64, brute-force ground truth via `SIMDDistance.cosine`
    - Search using `FullGPUSearch` (or `SearchGPU` if FullGPUSearch not yet wired)
    - Call `GraphPruner.prune(...)`
    - Measure pruned recall: same queries, same parameters
    - Assert: `prunedRecall > baselineRecall - 0.02` (no more than 2% drop)
- [x] 25.2 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/GraphPrunerTests 2>&1 | grep -E '(PASS|FAIL|error:)'` → confirms FAIL (GraphPruner not found)
- [x] 25.3 — Create `Sources/MetalANNSCore/GraphPruner.swift`:
  - `public enum GraphPruner` — stateless, all static methods
  - `static func prune(graph:vectors:nodeCount:metric:) throws`
  - For each node `u` in 0..<nodeCount:
    - Read neighbors sorted by distance (filter sentinels)
    - For each candidate `v` (ascending distance):
      - Check: does any already-selected pruned neighbor `w` satisfy `d(w,v) < d(u,v)`?
      - If yes: skip `v` (redundant)
      - If no: select `v`
    - Write pruned list back to `GraphBuffer`, pad to `degree`
  - Uses `SIMDDistance.distance(...)` for distance computation between pairs
  - Uses `VectorBuffer.vector(at:)` to read vector data
- [x] 25.4 — **DECISION POINT (25.1)**: Pruning always-on vs opt-in. Options: (a) always-on in `ANNSIndex.build()`, (b) opt-in via `IndexConfiguration.enablePruning`, (c) separate `ANNSIndex.prune()` method. **Document choice in notes.**
- [x] 25.5 — **DECISION POINT (25.2)**: Pruning alpha. Options: (a) strict `alpha=1.0` (d(w,v) < d(u,v)), (b) relaxed `alpha=1.2` (d(w,v) < 1.2 * d(u,v)), (c) configurable. **Document choice in notes.**
- [x] 25.6 — **DECISION POINT (25.3)**: Monitor average post-pruning neighbor count. If < `degree/2`, pruning is too aggressive. **Report the numbers in notes.**
- [x] 25.7 — **GREEN**: Both GraphPruner tests pass. Specifically confirm:
  - `pruningReducesRedundancy`: edges reduced, average > degree/2
  - `pruningMaintainsRecall`: recall drop < 2%
- [x] 25.8 — Wire `GraphPruner.prune()` into `ANNSIndex.build()`:
  - Call after graph construction (both GPU and CPU paths)
  - Before assigning `self.graph = graphBuffer`
- [x] 25.9 — **REGRESSION**: All prior tests still pass (59 from Task 24 + 2 new = 61 total)
- [x] 25.10 — **GIT**: `git add Sources/MetalANNSCore/GraphPruner.swift Tests/MetalANNSTests/GraphPrunerTests.swift Sources/MetalANNS/ANNSIndex.swift && git commit -m "feat: add CAGRA-style graph pruning for higher quality edges"`

> **Agent notes** _(REQUIRED — document decisions 25.1, 25.2, 25.3, and any Phase 1–7 file modifications)_:
>
> - 25.1 Pruning enablement: implemented as always-on inside `ANNSIndex.build()` (single post-construction pass on both CPU/GPU build paths).
> - 25.2 Alpha choice: strict `alpha = 1.0` (`d(w,v) < d(u,v)`).
> - 25.3 Post-pruning neighbor count monitoring: measured `average neighbors after pruning = 5.205` for the redundancy test case (`degree = 8`, threshold = `4.0`), so pruning was not over-aggressive.
> - Additional observed quality metric: recall in pruning test stayed `baseline = 1.0`, `pruned = 1.0`.
> - Phase 1–7 file modifications: `ANNSIndex.swift` build path now invokes `GraphPruner.prune(...)` after graph construction and before assigning state.

---

## Phase 8 Complete — Signal

When all items above are checked, update this section:

```
STATUS: complete
FINAL TEST RESULT: pass (`xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'`)
TOTAL COMMITS: 29 (expected: 29)
TOTAL TESTS: 61 (expected: 61 = 57 prior + 2 from Task 24 + 2 from Task 25)
ISSUES ENCOUNTERED:
- Minor environment quirk: direct `xcodebuild` invocation once returned an inconsistent package-detection error; rerun via wrapped `/bin/zsh -lc` command completed normally.
DECISIONS MADE:
- 24.1: `min(degree, maxThreads)` threadgroup width selected.
- 24.2: thread-0 insertion sort per iteration retained.
- 24.3: `SearchGPU.swift` kept as fallback (not deleted).
- 24.4: no shared-memory downsizing needed (`MAX_EF=256`, `MAX_VISITED=4096`).
- 25.1: pruning always-on in `ANNSIndex.build()`.
- 25.2: strict alpha `1.0` (`d(w,v) < d(u,v)`).
- 25.3: average post-pruning neighbor count observed `5.205` (`degree=8`, threshold `4.0`).
PHASE 1–7 FILES MODIFIED:
- ANNSIndex.swift: replaced `SearchGPU` with `FullGPUSearch` in GPU search path; added `GraphPruner.prune(...)` in build path.
BENCHMARK COMPARISON (if measured):
- Hybrid search latency (SearchGPU): _pending_
- Full GPU search latency (FullGPUSearch): _pending_
- Speedup: _pending_
```

---

## Orchestrator Review Checklist (DO NOT MODIFY — Orchestrator use only)

- [ ] R1 — Git log shows exactly 29 commits with correct conventional commit messages
- [ ] R2 — Full test suite passes: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'` — zero failures including all Phase 1–7 tests
- [ ] R3 — `Search.metal` exists at `Sources/MetalANNSCore/Shaders/Search.metal`
- [ ] R4 — `beam_search` kernel has correct buffer bindings (12 buffers: vectors, adjacency, query, output_dists, output_ids, node_count, degree, dim, k, ef, entry_point, metric_type)
- [ ] R5 — Kernel uses `threadgroup` shared memory for candidates, results, and visited hash table
- [ ] R6 — Visited hash table uses `atomic_compare_exchange_weak_explicit` with `memory_order_relaxed`
- [ ] R7 — Kernel implements all three distance metrics (cosine=0, l2=1, innerProduct=2)
- [ ] R8 — `MAX_EF` and `MAX_VISITED` fit within 32KB threadgroup memory
- [ ] R9 — `FullGPUSearch` is a stateless `public enum` in `MetalANNSCore` target
- [ ] R10 — `FullGPUSearch.search()` dispatches exactly 1 threadgroup
- [ ] R11 — `FullGPUSearch.search()` clamps `ef` to `MAX_EF` (256 or adjusted value)
- [ ] R12 — `ANNSIndex.search()` GPU path now uses `FullGPUSearch` (not `SearchGPU`)
- [ ] R13 — `SearchGPU.swift` is retained (not deleted) as fallback
- [ ] R14 — `FullGPUSearch` recall is > 0.80 and within 5% of `SearchGPU` hybrid
- [ ] R15 — `GraphPruner` is a stateless `public enum` in `MetalANNSCore` target
- [ ] R16 — `GraphPruner.prune()` implements path-based diversification (check d(w,v) < d(u,v))
- [ ] R17 — `GraphPruner` uses `SIMDDistance` for pairwise distance computation
- [ ] R18 — Pruning reduces edge count (fewer valid non-sentinel neighbors after pruning)
- [ ] R19 — Pruning does not over-prune (average neighbors per node > degree/2)
- [ ] R20 — Pruning recall degradation < 2%
- [ ] R21 — `ANNSIndex.build()` calls `GraphPruner.prune()` after graph construction
- [ ] R22 — No `import XCTest` or `XCTSkip` anywhere
- [ ] R23 — No Phase 1–7 files modified beyond `ANNSIndex.swift` (or changes documented with justification)
- [ ] R24 — Agent notes filled in for all 7 decision points (24.1-24.4, 25.1-25.3)
- [ ] R25 — Total test count is at least 61 (57 prior + 2 + 2 new)
- [ ] R26 — All new types are `Sendable` (GraphPruner as enum is inherently Sendable, FullGPUSearch as enum is inherently Sendable)
