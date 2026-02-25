# Phase 3 Execution Prompt: Graph Construction — NN-Descent via Metal

---

## System Context

You are implementing **Phase 3 (Tasks 10–13)** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1 and 2 are complete. The codebase has a working Swift package with dual compute backends (CPU + GPU), verified distance kernels, MetalContext, PipelineCache, and GPU-resident data structures (VectorBuffer, GraphBuffer, MetadataBuffer, IDMap). Git log shows 10 commits.

You are building the **NN-Descent graph construction pipeline** — both a CPU reference implementation and the full GPU implementation with Metal compute shaders. This is the most complex phase of the project. The GPU implementation uses atomic CAS for concurrent graph updates, reverse edge lists for bidirectional join, and bitonic sort for neighbor ordering.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/phase3-todo.md`

**Your communication contract:**
1. **`tasks/phase3-todo.md` is your shared state.** Check off `[x]` items as you complete them. The orchestrator reads this file to track your progress.
2. **Write notes under every task** — especially for decision points and any issues you hit. The orchestrator reviews your notes.
3. **Update `Last Updated`** at the top of phase3-todo.md after each task completes.
4. **When done, fill in the "Phase 3 Complete — Signal" section** at the bottom of phase3-todo.md. This is how the orchestrator knows you're finished.
5. **Do NOT modify the "Orchestrator Review Checklist"** section at the bottom — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **TDD cycle for every task**: Write test → run to see it fail (RED) → implement → run to see it pass (GREEN) → commit. No exceptions. Check off the RED and GREEN items separately in the todo.
2. **Swift 6 strict concurrency**: All types must be `Sendable`. Use `@unchecked Sendable` ONLY for classes wrapping `MTLBuffer`. Enums with only static methods (`NNDescentCPU`, `NNDescentGPU`) are naturally `Sendable`.
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest. For simulator skipping, do NOT use `throw XCTSkip(...)` — the plan's test code shows this pattern but it's wrong for Swift Testing. Instead use a guard check like:
   ```swift
   guard MTLCreateSystemDefaultDevice() != nil else { return }
   ```
4. **Build and test with `xcodebuild`**, never `swift build` or `swift test`. Metal shaders are not compiled by SPM CLI. **IMPORTANT**: The test scheme may be `MetalANNS-Package` rather than `MetalANNS`. If `MetalANNS` doesn't work for test action, try `MetalANNS-Package`.
5. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
6. **Commit after every task** with the exact conventional commit message specified in the todo.
7. **All Metal atomics use `memory_order_relaxed`** — Metal supports nothing else.
8. **Check off todo items in real time** — not at the end. This is how the orchestrator tracks live progress.
9. **Do NOT modify Phase 1 or Phase 2 files** unless strictly necessary for compilation. If you need to change an earlier file, document the reason in your notes.

---

## What Already Exists (Phases 1–2 Output)

Before writing any code, understand the existing codebase. These are the files you'll build on:

### Source Files (MetalANNSCore) — Phase 1
| File | What It Provides |
|------|-----------------|
| `Metric.swift` | `Metric` enum (`.cosine`, `.l2`, `.innerProduct`) — `Sendable`, `Codable` |
| `Errors.swift` | `ANNSError` enum — use `constructionFailed(_:)` for failures |
| `ComputeBackend.swift` | `ComputeBackend` protocol + `BackendFactory` |
| `AccelerateBackend.swift` | CPU distance computation — your CPU NN-Descent (Task 10) uses this directly |
| `MetalDevice.swift` | `MetalContext` — provides `device`, `commandQueue`, `library`, `pipelineCache`, `execute(_:)` |
| `PipelineCache.swift` | `actor PipelineCache` — `pipeline(for:) throws -> MTLComputePipelineState` |
| `MetalBackend.swift` | GPU distance computation (reference for buffer encoding patterns) |
| `Shaders/Distance.metal` | 3 distance kernels: `cosine_distance`, `l2_distance`, `inner_product_distance` |

### Source Files (MetalANNSCore) — Phase 2
| File | What It Provides |
|------|-----------------|
| `VectorBuffer.swift` | `VectorBuffer` — `.buffer: MTLBuffer`, `.dim`, `.capacity`, `.count`, `.insert(vector:at:)`, `.batchInsert(vectors:startingAt:)`, `.vector(at:)`, `.setCount(_:)`, `.floatPointer` |
| `GraphBuffer.swift` | `GraphBuffer` — `.adjacencyBuffer: MTLBuffer`, `.distanceBuffer: MTLBuffer`, `.degree`, `.capacity`, `.nodeCount`, `.setNeighbors(of:ids:distances:)`, `.neighborIDs(of:)`, `.neighborDistances(of:)`, `.setCount(_:)` |
| `MetadataBuffer.swift` | `MetadataBuffer` — 5 UInt32 fields via MTLBuffer |
| `IDMap.swift` | `IDMap` struct — bidirectional String↔UInt32 mapping |

### Key APIs You'll Use Heavily

```swift
// MetalContext — creating and executing GPU work
let context = try MetalContext()
let pipeline = try await context.pipelineCache.pipeline(for: "kernel_name")
try await context.execute { commandBuffer in
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBytes(&value, length: MemoryLayout<UInt32>.stride, index: 1)
    encoder.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
    encoder.endEncoding()
}

// VectorBuffer — loading test vectors for GPU
let vectorBuffer = try VectorBuffer(capacity: n, dim: dim, device: context.device)
try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
vectorBuffer.setCount(n)

// GraphBuffer — reading back results for verification
let ids = graph.neighborIDs(of: nodeID)      // [UInt32]
let dists = graph.neighborDistances(of: nodeID)  // [Float]

// AccelerateBackend — CPU distance computation for reference
let backend = AccelerateBackend()
let distances = try await flat.withUnsafeBufferPointer { ptr in
    try await backend.computeDistances(query: vectors[i], vectors: ptr, vectorCount: n, dim: dim, metric: .cosine)
}
```

### Existing Tests (must not regress)
- Phase 1: `PlaceholderTests` (1), `ConfigurationTests` (3), `BackendProtocolTests` (1), `DistanceTests` (8), `MetalDeviceTests` (2), `MetalDistanceTests` (2)
- Phase 2: `VectorBufferTests` (3), `GraphBufferTests` (3), `MetadataTests` (3)
- **Total: 26 tests** — all must continue passing

---

## Success Criteria

Phase 3 is done when ALL of the following are true:

- [ ] `NNDescentCPU.build()` constructs a graph with correct dimensions, no self-loops, and recall > 0.85 for 50 nodes
- [ ] `NNDescentGPU.randomInit()` produces valid random graphs (no self-loops, no duplicate neighbors, valid IDs)
- [ ] `NNDescentGPU.build()` constructs a graph with recall > 0.80 for 200 nodes (GPU vs brute-force)
- [ ] `NNDescentGPU.sortNeighborLists()` sorts each node's neighbor list by distance ascending
- [ ] All new tests pass AND all Phase 1 + Phase 2 tests still pass (zero regressions)
- [ ] Git history has exactly 14 commits (10 prior + 4 new)
- [ ] `tasks/phase3-todo.md` has all items checked and the completion signal filled in

---

## Execution Instructions

### Before You Start

1. Read `tasks/phase3-todo.md` — this is your checklist. Every item you must do is there.
2. Read `docs/plans/2026-02-25-metalanns-implementation.md` (Tasks 10–13, lines ~1419–2370) — this has the **complete code** for every file, test, and shader. Use it as your primary reference.
3. Run the full test suite to confirm Phases 1–2 are green: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
4. Complete the **Pre-Flight Checks** in phase3-todo.md first.

### For Each Task (10 through 13)

Follow this exact loop:

```
1. Read the task's items in tasks/phase3-todo.md
2. Write the test file (check off the "create test" item)
3. Run the test, verify RED (check off the "RED" item)
4. Write the implementation file(s) — shader + Swift wrapper (check off each file item)
5. Run the test, verify GREEN (check off the "GREEN" item)
6. Run regression check — ALL prior tests still pass
7. Git commit with the specified message (check off the "GIT" item)
8. Update "Last Updated" in phase3-todo.md header
9. Write any notes under the task in phase3-todo.md
```

### After All 4 Tasks

1. Run full test suite: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'`
2. Run `git log --oneline` and verify 14 commits
3. Fill in the **"Phase 3 Complete — Signal"** section in phase3-todo.md
4. Do NOT touch the **"Orchestrator Review Checklist"** section

---

## Task-by-Task Reference

The complete code for every task is in `docs/plans/2026-02-25-metalanns-implementation.md` (lines ~1419–2370). Below is a summary of each task with key gotchas. **Read the plan file for full code.**

### Task 10: CPU NN-Descent (Reference Implementation)

**Purpose**: Pure-CPU NN-Descent using Accelerate for distance computation. This is the ground truth for GPU construction and enables full TDD on simulator.

- Create `Sources/MetalANNSCore/NNDescentCPU.swift` and `Tests/MetalANNSTests/NNDescentCPUTests.swift`
- 3 tests: `graphDimensions` (50 nodes, degree 4), `noSelfLoops`, `recallCheck` (recall > 0.85 vs brute-force)
- `NNDescentCPU` is a `public enum` with `static func build(...)` — no instance state needed
- Algorithm: random init → reverse edges → local join (forward × reverse pairs) → convergence check
- Convergence: stop when `updateCount < convergenceThreshold * degree * n`
- Entry point selection: node with minimum mean distance to its neighbors
- **Returns**: `(graph: [[(UInt32, Float)]], entryPoint: UInt32)` — graph is array of sorted neighbor lists
- **Key**: Uses inline distance computation (not AccelerateBackend) for simplicity within the join loop
- **Key**: The recall test uses `AccelerateBackend.computeDistances()` for brute-force reference

**Commit**: `feat: implement CPU NN-Descent reference (Accelerate backend)`

### Task 11: Metal NN-Descent Shaders — Random Init & Initial Distances

**Purpose**: GPU kernels for random graph initialization and computing initial neighbor distances.

- Create `Sources/MetalANNSCore/Shaders/NNDescent.metal` and `Sources/MetalANNSCore/NNDescentGPU.swift`, add to `Tests/MetalANNSTests/NNDescentGPUTests.swift`
- 1 test: `randomInitValid` — 100 nodes, degree 8, verify no self-loops, no duplicates, valid IDs
- **Guard GPU tests** with `guard MTLCreateSystemDefaultDevice() != nil else { return }` — NOT `XCTSkip`
- Two Metal kernels in `NNDescent.metal`:
  - `random_init` — LCG PRNG per node, retry loop for no-self-loops and no-duplicates (max 100 attempts)
  - `compute_initial_distances` — one thread per `(node, slot)`, computes distance using metric_type param (0=cosine, 1=l2, 2=innerProduct)
- `NNDescentGPU` is a `public enum` with static methods
- `randomInit(context:graph:nodeCount:seed:)` — dispatches `random_init` kernel
- `computeInitialDistances(context:vectors:graph:nodeCount:metric:)` — dispatches `compute_initial_distances` kernel
- **Buffer indices must match between Metal and Swift exactly** — see plan for precise `setBuffer`/`setBytes` order
- **DECISION POINT (11.6)**: The `metric_type` parameter maps as `UInt32`: cosine=0, l2=1, innerProduct=2. Verify this matches the `compute_initial_distances` kernel's `if/else` chain.

**Commit**: `feat: add Metal random_init and compute_initial_distances kernels`

### Task 12: Metal Reverse Edge & Local Join Kernels

**Purpose**: The core NN-Descent iteration — build reverse edge lists, then local join with atomic CAS graph updates. This is the **most complex task** in the entire project.

- Modify `NNDescent.metal` (add `build_reverse_list` + `local_join` kernels)
- Modify `NNDescentGPU.swift` (add `build()` orchestrator method)
- Add integration test to `NNDescentGPUTests.swift`
- 1 test: `fullGPUConstruction` — 200 nodes, dim=16, degree=8, recall > 0.80 vs brute-force
- **Reverse edge kernel** (`build_reverse_list`):
  - One thread per edge (total = `nodeCount * degree`)
  - For each forward edge u→v, atomically append u to `reverse_list[v]`
  - Uses `atomic_fetch_add_explicit` on `reverse_counts[v]` with `memory_order_relaxed`
  - Bounded by `max_reverse` (default: `degree * 2`) — excess edges are dropped
- **Local join kernel** (`local_join`):
  - One thread per node
  - Collects forward neighbors (from atomic adjacency) and reverse neighbors
  - For each (forward, reverse) pair: compute distance, try CAS update on both nodes
  - **CAS pattern**: `as_type<uint>(distance)` to store float as uint bits for atomic compare-exchange
  - This works because `as_type<uint>(float)` preserves comparison order for non-negative floats
  - Uses `atomic_compare_exchange_weak_explicit` with `memory_order_relaxed`
  - `update_counter` is atomically incremented for convergence tracking
  - **Fixed-size thread-local arrays**: `fwd[64]` (max degree), `rev[128]` (max reverse) — these MUST be large enough for the degree used in tests
- **`NNDescentGPU.build()` orchestrator**:
  - Calls `randomInit` → `computeInitialDistances` → iteration loop (clear reverse → build reverse → local join → check convergence)
  - Allocates 3 extra buffers: `reverseListBuffer`, `reverseCountBuffer`, `updateCountBuffer`
  - Reads back `updateCountBuffer` after each iteration to check convergence
  - `maxReverse = degree * 2`
- **CRITICAL**: The `local_join` kernel reads `adj_ids` and `adj_dists_bits` as `device atomic_uint *`, but `GraphBuffer` stores them as regular `MTLBuffer`. Metal allows reinterpreting the same buffer as atomic — the buffer contents are just uint32 values either way. **No new buffers are needed** — pass `graph.adjacencyBuffer` and `graph.distanceBuffer` directly.
- **DECISION POINT (12.7)**: The convergence threshold is `0.001 * degree * nodeCount`. For 200 nodes with degree 8, that's 1.6 updates. Verify this is a sensible threshold for the test to converge within `maxIterations`.

**Commit**: `feat: implement GPU NN-Descent with reverse edges, local join, and convergence`

### Task 13: Bitonic Sort Kernel

**Purpose**: GPU bitonic sort for ordering each node's neighbor list by distance ascending after construction.

- Create `Sources/MetalANNSCore/Shaders/Sort.metal` and `Tests/MetalANNSTests/BitonicSortTests.swift`
- Modify `NNDescentGPU.swift` (add `sortNeighborLists` method)
- 1 test: `sortNeighborLists` — 100 nodes, degree=32, fill with random, sort, verify ascending
- **Bitonic sort kernel**:
  - One threadgroup per node, `degree/2` threads per threadgroup
  - Uses threadgroup shared memory for the sort
  - Sorts paired (distance, ID) arrays — when distances are swapped, IDs must be swapped too
  - **degree must be power of 2** for bitonic sort. The plan uses degree=32 which works.
- **DECISION POINT (13.5)**: Should the sort be called after each NN-Descent iteration (inside the loop) or only once after the final iteration? The plan calls it once at the end. Calling per-iteration would help the local join find the "worst" slot faster but adds overhead. **Document your decision.**
- Optionally integrate sort into `NNDescentGPU.build()` to call after the iteration loop completes.

**Commit**: `feat: add bitonic sort kernel for neighbor list ordering`

---

## Decision Points Summary

You MUST make and document these decisions in `tasks/phase3-todo.md` notes:

| # | Decision | Recommended Approach |
|---|----------|---------------------|
| 11.6 | Metric type UInt32 mapping consistency | Verify cosine=0, l2=1, innerProduct=2 in both shader and Swift |
| 12.7 | Convergence threshold for 200-node test | 0.001 * degree * nodeCount = 1.6 updates; should converge within 15 iterations |
| 13.5 | Sort per iteration vs once at end | Once at end (recommended) — less overhead, final sort is what matters for search |

---

## Metal Shader Gotchas (Read Before Writing Shaders)

| Issue | Detail | Fix |
|-------|--------|-----|
| Atomics only support `memory_order_relaxed` | Metal has no acquire/release semantics | Always use `memory_order_relaxed` for all atomic ops |
| `as_type<uint>(float)` for CAS on floats | Preserves comparison order for non-negative floats only | All distance values are non-negative — this is safe |
| `atomic_compare_exchange_weak_explicit` may fail spuriously | Weak CAS can return false even if expected matches | This is acceptable — the update will be retried next iteration |
| Thread-local arrays have fixed size | `fwd[64]` and `rev[128]` must accommodate max degree | Verify degree < 64 in all tests |
| `if (tid >= node_count) return;` guard | Prevents out-of-bounds access for non-uniform grid sizes | Must be first line of every kernel |
| No `#include` needed between .metal files | All kernels in `Shaders/` are compiled together into one library | Functions in `Distance.metal` are separate from `NNDescent.metal` |
| Buffer reinterpretation as atomic | Same `MTLBuffer` can be read as `device uint*` or `device atomic_uint*` | No special buffer allocation needed |

---

## Common Failure Modes (Read Before Starting)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `recallCheck` fails (recall < 0.85) | NN-Descent not converging | Increase `maxIterations`, check convergence threshold |
| GPU recall < 0.80 | Local join not updating graph | Verify CAS pattern — `as_type<uint>(float)` on distance, not raw float bits |
| `random_init` produces self-loops | LCG seed collision or retry limit too low | Verify per-node seed: `seed ^ (tid * 2654435761u)`, max 100 retries |
| `local_join` crashes | Thread-local array overflow | Verify `fwd[64]` and `rev[128]` are large enough for degree used |
| `build_reverse_list` misses edges | `max_reverse` too small | Default `degree * 2` — increase if degree is large |
| Sort doesn't work for non-power-of-2 degree | Bitonic sort requires power-of-2 input | Use degree=32 (or 8, 16, 64) in tests |
| Test uses `XCTSkip` and fails to compile | XCTest API used in Swift Testing context | Use `guard MTLCreateSystemDefaultDevice() != nil else { return }` |
| Scheme `MetalANNS` not found for test action | Xcode auto-generates `MetalANNS-Package` as test scheme | Use `MetalANNS-Package` for `xcodebuild test` |
| `atomic_fetch_add_explicit` not found | Missing `<metal_stdlib>` include | Ensure `#include <metal_stdlib>` at top of shader file |

---

## Reference Files

| File | Purpose |
|------|---------|
| `docs/plans/2026-02-25-metalanns-implementation.md` (lines 1419–2370) | **Complete code** for Tasks 10–13 |
| `docs/plans/2026-02-25-metalanns-design.md` | Architecture decisions, NN-Descent algorithm description |
| `Sources/MetalANNSCore/MetalBackend.swift` | Buffer encoding pattern reference |
| `Sources/MetalANNSCore/MetalDevice.swift` | `MetalContext.execute(_:)` pattern |
| `Sources/MetalANNSCore/Shaders/Distance.metal` | Reference for kernel structure and buffer binding |
| `Sources/MetalANNSCore/VectorBuffer.swift` | `.buffer`, `.dim`, `.batchInsert`, `.setCount` |
| `Sources/MetalANNSCore/GraphBuffer.swift` | `.adjacencyBuffer`, `.distanceBuffer`, `.neighborIDs(of:)`, `.neighborDistances(of:)` |
| `tasks/phase3-todo.md` | **Your checklist** — check items off as you go |
| `tasks/lessons.md` | Record any lessons learned |

---

## Scope Boundary (What NOT To Do)

- Do NOT implement Phase 4+ code (BeamSearch, Persistence, ANNSIndex, SearchResult)
- Do NOT add features beyond the plan (no FP16 construction, no pruning pass, no CAGRA post-processing)
- Do NOT modify Phase 1 or Phase 2 files unless compilation requires it (document any changes)
- Do NOT use XCTest — Swift Testing exclusively. No `XCTSkip`, no `import XCTest`
- Do NOT use `swift build` or `swift test` — `xcodebuild` only
- Do NOT create README.md or documentation files
- Do NOT modify the Orchestrator Review Checklist in phase3-todo.md
- Do NOT attempt to optimize the local join kernel beyond what the plan specifies — correctness first
