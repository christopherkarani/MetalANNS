# MetalANNS — Phase 3: Graph Construction — NN-Descent via Metal

> **Status**: COMPLETE
> **Owner**: Subagent (dispatched by orchestrator)
> **Reviewer**: Orchestrator (main session)
> **Last Updated**: 2026-02-25 04:50:34 EAT

---

## How This File Works

This file is the **shared communication layer** between the orchestrator and executing agents.

**Executing agent**: Check off items `[x]` as you complete them. Add notes under any item if you hit issues. Update `Last Updated` timestamp after each task. Do NOT check off an item unless the verification step passes.

**Orchestrator**: Reviews checked items against actual codebase state. Unchecks items that don't pass review. Adds review comments prefixed with `> [REVIEW]`.

---

## Pre-Flight Checks

- [x] Working directory is `/Users/chriskarani/CodingProjects/MetalANNS`
- [x] Phase 1+2 code exists: `VectorBuffer.swift`, `GraphBuffer.swift`, `MetalDevice.swift`, `PipelineCache.swift`, `Distance.metal` all present
- [x] `git log --oneline` shows 10 commits (6 Phase 1 + 1 orchestration + 3 Phase 2)
- [x] Full test suite passes (26 tests, zero failures): `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
- [x] Read implementation plan: `docs/plans/2026-02-25-metalanns-implementation.md` (lines 1419–2370, Tasks 10–13)

---

## Task 10: CPU NN-Descent (Reference Implementation)

**Acceptance**: `NNDescentCPUTests` suite passes (3 tests). Eleventh commit.

- [x] 10.1 — Create `Tests/MetalANNSTests/NNDescentCPUTests.swift` — 3 tests using Swift Testing:
  - `graphDimensions` — 50 nodes, dim=8, degree=4: verify `graph.count == 50` and each node has exactly `degree` neighbors
  - `noSelfLoops` — same graph: verify no node appears in its own neighbor list
  - `recallCheck` — build graph, compute brute-force kNN via `AccelerateBackend`, verify average recall > 0.85
- [x] 10.2 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/NNDescentCPUTests 2>&1 | grep -E '(PASS|FAIL|error:)'` → confirms FAIL (NNDescentCPU not defined)
- [x] 10.3 — Create `Sources/MetalANNSCore/NNDescentCPU.swift`:
  - `public enum NNDescentCPU` — stateless, all static methods
  - `static func build(vectors:degree:metric:maxIterations:convergenceThreshold:) async throws -> (graph: [[(UInt32, Float)]], entryPoint: UInt32)`
  - Algorithm steps:
    1. Random init — each node gets `degree` random unique non-self neighbors
    2. Iteration loop: build reverse edges → local join (forward × reverse pairs) → check convergence
    3. Local join: for each pair (a, b), compute distance, try to improve both a's and b's neighbor lists
    4. Convergence: stop when `updateCount < convergenceThreshold * degree * n`
    5. Entry point: node with minimum mean distance to neighbors
  - Inline distance computation (cosine/l2/innerProduct) — not using AccelerateBackend for the join loop
  - Neighbor lists maintained sorted by distance ascending
- [x] 10.4 — **GREEN**: All 3 tests pass. Specifically confirm `recallCheck` achieves > 0.85
- [x] 10.5 — **REGRESSION**: All Phase 1 + Phase 2 tests still pass (26 prior tests)
- [x] 10.6 — **GIT**: `git add Sources/MetalANNSCore/NNDescentCPU.swift Tests/MetalANNSTests/NNDescentCPUTests.swift && git commit -m "feat: implement CPU NN-Descent reference (Accelerate backend)"`

> **Agent notes** _(write issues/decisions here)_:

---

## Task 11: Metal NN-Descent Shaders — Random Init & Initial Distances

**Acceptance**: `NNDescentGPUTests/randomInitValid` passes on Mac with GPU. Twelfth commit.

- [x] 11.1 — Create `Tests/MetalANNSTests/NNDescentGPUTests.swift` — 1 test:
  - `randomInitValid` — 100 nodes, degree=8: verify no self-loops, no duplicate neighbors, all IDs < nodeCount
  - Guard with `guard MTLCreateSystemDefaultDevice() != nil else { return }` — NOT `XCTSkip`
- [x] 11.2 — **RED**: Test fails (NNDescentGPU not defined)
- [x] 11.3 — Create `Sources/MetalANNSCore/Shaders/NNDescent.metal` — 2 kernels:
  - `random_init`:
    - `buffer(0)` = adjacency (device uint*), `buffer(1)` = node_count, `buffer(2)` = degree, `buffer(3)` = seed
    - LCG PRNG: `state * 1664525u + 1013904223u`, per-node seed: `seed ^ (tid * 2654435761u)`
    - Retry loop (max 100 attempts) to avoid self-loops and duplicates
    - Guard: `if (tid >= node_count) return;`
  - `compute_initial_distances`:
    - `buffer(0)` = vectors, `buffer(1)` = adjacency, `buffer(2)` = distances, `buffer(3)` = node_count, `buffer(4)` = degree, `buffer(5)` = dim, `buffer(6)` = metric_type
    - One thread per (node, slot): `tid / degree` = node, `tid % degree` = slot
    - metric_type: 0=cosine, 1=l2, 2=innerProduct
    - Guard: `if (tid >= node_count * degree) return;`
- [x] 11.4 — Create `Sources/MetalANNSCore/NNDescentGPU.swift`:
  - `public enum NNDescentGPU` — stateless, all static methods
  - `static func randomInit(context:graph:nodeCount:seed:) async throws`
  - `static func computeInitialDistances(context:vectors:graph:nodeCount:metric:) async throws`
  - Both methods: get pipeline → set buffers/bytes → dispatch threads → execute
  - **Metric mapping**: `var metricType: UInt32 = switch metric { case .cosine: 0; case .l2: 1; case .innerProduct: 2 }`
- [x] 11.5 — **GREEN**: `randomInitValid` test passes
- [x] 11.6 — **METRIC MAPPING DECISION**: Verify that the `metric_type` UInt32 mapping (cosine=0, l2=1, innerProduct=2) is consistent between the Metal shader's `if/else` chain and the Swift wrapper. **Write confirmation in the notes below.**
- [x] 11.7 — **REGRESSION**: All prior tests still pass
- [x] 11.8 — **GIT**: `git add Sources/MetalANNSCore/Shaders/NNDescent.metal Sources/MetalANNSCore/NNDescentGPU.swift Tests/MetalANNSTests/NNDescentGPUTests.swift && git commit -m "feat: add Metal random_init and compute_initial_distances kernels"`

> **Agent notes** _(REQUIRED — document your 11.6 confirmation here)_:
> Metric mapping verified as consistent in both layers:
> `NNDescentGPU.computeInitialDistances` maps `.cosine -> 0`, `.l2 -> 1`, `.innerProduct -> 2`,
> and `compute_initial_distances` in `NNDescent.metal` dispatches `if metric_type == 0` (cosine),
> `else if metric_type == 1` (l2), `else` (inner product).

---

## Task 12: Metal Reverse Edge & Local Join Kernels

**Acceptance**: `NNDescentGPUTests/fullGPUConstruction` passes with recall > 0.80. Thirteenth commit.

**This is the most complex task in the project. Read the plan code carefully.**

- [x] 12.1 — Add integration test to `Tests/MetalANNSTests/NNDescentGPUTests.swift`:
  - `fullGPUConstruction` — 200 nodes, dim=16, degree=8, maxIter=15
  - Generate random vectors, load into VectorBuffer, create GraphBuffer
  - Call `NNDescentGPU.build(context:vectors:graph:nodeCount:metric:maxIterations:)`
  - Compute brute-force recall via AccelerateBackend
  - Assert `avgRecall > 0.80`
  - Guard with `guard MTLCreateSystemDefaultDevice() != nil else { return }`
- [x] 12.2 — **RED**: Test fails (`NNDescentGPU.build` not implemented)
- [x] 12.3 — Add 2 kernels to `Sources/MetalANNSCore/Shaders/NNDescent.metal`:
  - `build_reverse_list`:
    - Buffers: adjacency(0), reverse_list(1), reverse_counts(2, atomic_uint), node_count(3), degree(4), max_reverse(5)
    - One thread per edge (total = nodeCount * degree)
    - For edge u→v: `atomic_fetch_add_explicit(&reverse_counts[v], 1u, memory_order_relaxed)` to get slot
    - Write u into `reverse_list[v * max_reverse + slot]` if slot < max_reverse
  - `local_join`:
    - 11 buffers: vectors(0), adj_ids(1, atomic_uint), adj_dists_bits(2, atomic_uint), reverse_list(3), reverse_counts_r(4, regular uint), node_count(5), degree(6), max_reverse(7), dim(8), metric_type(9), update_counter(10, atomic_uint)
    - One thread per node
    - Collects forward neighbors from atomic adjacency → `fwd[64]`
    - Collects reverse neighbors from reverse_list → `rev[128]`
    - For each (fwd, rev) pair: compute distance, CAS update on both a's and b's lists
    - **CAS pattern**: find worst slot → `as_type<uint>(distance)` → check not already neighbor → `atomic_compare_exchange_weak_explicit` on distance bits → store ID
    - Symmetric update: tries to insert into both a and b
- [x] 12.4 — Add `NNDescentGPU.build()` to `Sources/MetalANNSCore/NNDescentGPU.swift`:
  - Full orchestration: randomInit → computeInitialDistances → iteration loop
  - Allocate 3 extra buffers: reverseListBuffer (nodeCount * maxReverse * 4), reverseCountBuffer (nodeCount * 4), updateCountBuffer (4 bytes)
  - Each iteration: memset reverse counts + update counter → build_reverse_list → local_join → read back updateCount → check convergence
  - `maxReverse = degree * 2`
  - After loop: `graph.setCount(nodeCount)`
- [x] 12.5 — **GREEN**: `fullGPUConstruction` test passes with recall > 0.80
- [x] 12.6 — If recall is borderline, try increasing `maxIterations` or adjusting threshold. Document any tuning in notes.
- [x] 12.7 — **CONVERGENCE DECISION**: For 200 nodes with degree 8, the threshold is `0.001 * 8 * 200 = 1.6` updates. Verify this allows convergence within `maxIterations=15`. **Write your observation in the notes below.**
- [x] 12.8 — **REGRESSION**: All prior tests still pass
- [x] 12.9 — **GIT**: `git add Sources/MetalANNSCore/Shaders/NNDescent.metal Sources/MetalANNSCore/NNDescentGPU.swift Tests/MetalANNSTests/NNDescentGPUTests.swift && git commit -m "feat: implement GPU NN-Descent with reverse edges, local join, and convergence"`

> **Agent notes** _(REQUIRED — document 12.7 convergence observation and any recall tuning)_:
> No recall tuning was needed; the `fullGPUConstruction` test passed with `maxIterations = 15`.
> Convergence threshold kept at `0.001 * degree * nodeCount` (1.6 updates for 200x8), and this setting was sufficient for the test target.

---

## Task 13: Bitonic Sort Kernel

**Acceptance**: `BitonicSortTests/sortNeighborLists` passes. Fourteenth commit.

- [x] 13.1 — Create `Tests/MetalANNSTests/BitonicSortTests.swift` — 1 test:
  - `sortNeighborLists` — 100 nodes, degree=32, fill with random IDs and distances
  - Call `NNDescentGPU.sortNeighborLists(context:graph:nodeCount:)`
  - Verify every node's distance list is sorted ascending: `dists[j] >= dists[j-1]` for all j
  - Guard with `guard MTLCreateSystemDefaultDevice() != nil else { return }`
- [x] 13.2 — **RED**: Test fails (`sortNeighborLists` not defined)
- [x] 13.3 — Create `Sources/MetalANNSCore/Shaders/Sort.metal`:
  - `bitonic_sort_neighbors` kernel
  - One threadgroup per node, `degree/2` threads per threadgroup
  - Uses threadgroup shared memory: `threadgroup float sharedDists[MAX_DEGREE]` and `threadgroup uint sharedIDs[MAX_DEGREE]`
  - Bitonic sort: outer loop over `k` (2, 4, 8, ..., degree), inner loop over `j` (k/2, k/4, ..., 1)
  - When swapping distances, must also swap corresponding IDs
  - **degree must be power of 2** (32 in test)
- [x] 13.4 — Add `NNDescentGPU.sortNeighborLists(context:graph:nodeCount:)` to `NNDescentGPU.swift`:
  - Get pipeline for "bitonic_sort_neighbors"
  - Set buffers: adjacencyBuffer(0), distanceBuffer(1), degree(2)
  - Dispatch: `nodeCount` threadgroups, `degree/2` threads per threadgroup
- [x] 13.5 — **SORT TIMING DECISION**: Should the sort be called per NN-Descent iteration or once at the end? **Document your decision in the notes below.** Recommended: once at end — less overhead, final ordering is what search needs.
- [x] 13.6 — **GREEN**: Sort test passes — every node's distances are ascending
- [x] 13.7 — **REGRESSION**: All prior tests still pass (26 + 3 Task 10 + 1 Task 11 + 1 Task 12)
- [x] 13.8 — **FULL SUITE**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|passed|failed)'` → **zero failures**
- [x] 13.9 — **GIT LOG**: `git log --oneline` shows exactly 14 commits
- [x] 13.10 — **GIT**: `git add Sources/MetalANNSCore/Shaders/Sort.metal Sources/MetalANNSCore/NNDescentGPU.swift Tests/MetalANNSTests/BitonicSortTests.swift && git commit -m "feat: add bitonic sort kernel for neighbor list ordering"`

> **Agent notes** _(REQUIRED — document your 13.5 decision here)_:
> Chosen approach: sort once at the end of `NNDescentGPU.build()` (not per-iteration).
> Rationale: it avoids repeated sort overhead during convergence while still guaranteeing final neighbor ordering for search.

---

## Phase 3 Complete — Signal

When all items above are checked, update this section:

```
STATUS: COMPLETE
FINAL TEST RESULT: Test run with 32 tests in 12 suites passed after 0.154 seconds. (xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS')
TOTAL COMMITS:
8187eb6 feat: add bitonic sort kernel for neighbor list ordering
6a06d6f feat: implement GPU NN-Descent with reverse edges, local join, and convergence
fe4ac93 feat: add Metal random_init and compute_initial_distances kernels
0a8c20a feat: implement CPU NN-Descent reference (Accelerate backend)
06c7d29 feat: add MetadataBuffer and bidirectional IDMap
aab0459 feat: add GraphBuffer for GPU-resident adjacency list storage
06ddde2 feat: add VectorBuffer for GPU-resident vector storage
3a6a17b Add phase-2 graph data structures prompt and todo
7cb5a9c feat: implement Metal distance shaders (cosine, L2, inner product) with GPU tests
045212c feat: add MetalContext with device lifecycle and PipelineCache
ad22132 feat: implement Accelerate distance kernels (cosine, L2, inner product)
b7d21f6 feat: add ComputeBackend protocol with factory and stub backends
d0dddeb feat: add ANNSError, Metric, and IndexConfiguration types
adc01bb chore: initialize MetalANNS Swift package scaffold
ISSUES ENCOUNTERED: None blocking. Early RED command for a single Swift Testing method selector returned 0 tests, so suite-level selector was used for reliable verification.
DECISIONS MADE:
- 11.6 metric mapping kept consistent across Swift and shader: cosine=0, l2=1, innerProduct=2.
- 12.7 convergence threshold kept at 0.001 * degree * nodeCount (1.6 for 200x8); no tuning required for passing recall target within maxIterations=15.
- 13.5 sorting is executed once at the end of build(), not per iteration, to reduce overhead while guaranteeing final neighbor ordering.
```

---

## Orchestrator Review Checklist (DO NOT MODIFY — Orchestrator use only)

- [ ] R1 — Git log shows exactly 14 commits with correct conventional commit messages
- [ ] R2 — Full test suite passes: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'` — zero failures including all Phase 1+2 tests
- [ ] R3 — CPU NN-Descent `recallCheck` achieves > 0.85 recall
- [ ] R4 — GPU NN-Descent `fullGPUConstruction` achieves > 0.80 recall
- [ ] R5 — `NNDescentCPU` is a stateless `enum` — no instance state
- [ ] R6 — `NNDescentGPU` is a stateless `enum` — no instance state
- [ ] R7 — All Metal atomics use `memory_order_relaxed` exclusively — no other ordering
- [ ] R8 — `local_join` uses `as_type<uint>(float)` for CAS on distance values (not raw float comparison)
- [ ] R9 — `build_reverse_list` uses `atomic_fetch_add_explicit` with bounded `max_reverse`
- [ ] R10 — Bitonic sort kernel co-swaps IDs and distances — not just distances
- [ ] R11 — No `import XCTest` or `XCTSkip` anywhere — Swift Testing exclusively
- [ ] R12 — GPU tests guarded with `guard MTLCreateSystemDefaultDevice() != nil else { return }` or equivalent
- [ ] R13 — Buffer indices in Metal shaders match Swift `setBuffer`/`setBytes` calls exactly
- [ ] R14 — No Phase 1 or Phase 2 files were modified (or changes are documented and justified)
- [ ] R15 — Agent notes filled in for Tasks 11.6, 12.7, and 13.5 decisions
- [ ] R16 — `random_init` kernel: per-node seed uses `seed ^ (tid * 2654435761u)`, retry loop for no-self-loops
- [ ] R17 — `NNDescentGPU.build()` calls `graph.setCount(nodeCount)` after construction completes
- [ ] R18 — No Phase 4+ code leaked in (no BeamSearch, no SearchResult, no Persistence)
