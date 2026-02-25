# Phase 4 Execution Prompt: Beam Search & Query API

---

## System Context

You are implementing **Phase 4 (Tasks 14–15)** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1–3 are complete. The codebase has dual compute backends, GPU-resident data structures (VectorBuffer, GraphBuffer, MetadataBuffer, IDMap), CPU and GPU NN-Descent construction with bitonic sort. Git log shows 14 commits.

You are building the **search pipeline** — both a CPU reference beam search and a GPU batch search. This is the last algorithmic phase before persistence and public API. After this phase, a user can construct an index and query it.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/phase4-todo.md`

**Your communication contract:**
1. **`tasks/phase4-todo.md` is your shared state.** Check off `[x]` items as you complete them.
2. **Write notes under every task** — especially for decision points and any issues you hit.
3. **Update `Last Updated`** at the top of phase4-todo.md after each task completes.
4. **When done, fill in the "Phase 4 Complete — Signal" section** at the bottom.
5. **Do NOT modify the "Orchestrator Review Checklist"** section — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **TDD cycle for every task**: Write test → RED → implement → GREEN → commit. Check off RED and GREEN separately.
2. **Swift 6 strict concurrency**: All types `Sendable`. `SearchResult` is a struct — naturally `Sendable`. `BeamSearchCPU` and `SearchGPU` are stateless enums.
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest or `XCTSkip`.
4. **Build and test with `xcodebuild`**. Test scheme is `MetalANNS-Package`. Never `swift build` or `swift test`.
5. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
6. **Commit after every task** with the exact conventional commit message specified in the todo.
7. **All Metal atomics use `memory_order_relaxed`**.
8. **Check off todo items in real time**.
9. **Do NOT modify Phase 1–3 files** unless strictly necessary. Document any changes.

---

## What Already Exists (Phases 1–3 Output)

### Key APIs You'll Use

```swift
// NNDescentCPU — builds CPU graph for CPU beam search testing
let (graphData, entryPoint) = try await NNDescentCPU.build(
    vectors: vectors, degree: degree, metric: .cosine, maxIterations: 10
)
// Returns: graph: [[(UInt32, Float)]], entryPoint: UInt32
// graph[i] is node i's sorted neighbor list: [(neighborID, distance)]

// NNDescentGPU — builds GPU graph for GPU beam search testing
let context = try MetalContext()
let vectorBuffer = try VectorBuffer(capacity: n, dim: dim, device: context.device)
try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
vectorBuffer.setCount(n)
let graph = try GraphBuffer(capacity: n, degree: degree, device: context.device)
try await NNDescentGPU.build(
    context: context, vectors: vectorBuffer, graph: graph,
    nodeCount: n, metric: .cosine, maxIterations: 15
)

// GraphBuffer — read back for CPU search
graph.neighborIDs(of: nodeID)        // [UInt32]
graph.neighborDistances(of: nodeID)  // [Float]

// AccelerateBackend — brute-force reference for recall testing
let backend = AccelerateBackend()
let distances = try await flat.withUnsafeBufferPointer { ptr in
    try await backend.computeDistances(query: query, vectors: ptr, vectorCount: n, dim: dim, metric: .cosine)
}

// MetalContext — GPU execution
try await context.execute { commandBuffer in ... }
let pipeline = try await context.pipelineCache.pipeline(for: "kernel_name")
```

### Existing Tests (must not regress)
- Phase 1: 17 tests (Placeholder, Configuration, BackendProtocol, Distance, MetalDevice, MetalDistance)
- Phase 2: 9 tests (VectorBuffer, GraphBuffer, Metadata)
- Phase 3: 6 tests (NNDescentCPU ×3, NNDescentGPU ×2, BitonicSort ×1)
- **Total: 32 tests** — all must continue passing

---

## Success Criteria

Phase 4 is done when ALL of the following are true:

- [ ] `SearchResult` struct exists in `Sources/MetalANNS/` with `id: String`, `score: Float`, `internalID: UInt32`
- [ ] `BeamSearchCPU.search()` returns k results sorted by distance, recall > 0.90 on 1000 vectors with 20 queries
- [ ] `SearchGPU` GPU search returns results matching CPU recall within tolerance (recall > 0.85 for 500 vectors)
- [ ] All new tests pass AND all Phase 1–3 tests still pass (zero regressions)
- [ ] Git history has exactly 16 commits (14 prior + 2 new)
- [ ] `tasks/phase4-todo.md` has all items checked and the completion signal filled in

---

## Execution Instructions

### Before You Start

1. Read `tasks/phase4-todo.md` — this is your checklist.
2. Read `docs/plans/2026-02-25-metalanns-implementation.md` (Tasks 14–15, lines ~2371–2513) — reference code for Task 14. Task 15 has high-level guidance; this prompt provides the detailed spec.
3. Run the full test suite to confirm Phases 1–3 are green: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
4. Complete the **Pre-Flight Checks** in phase4-todo.md.

### For Each Task (14 and 15)

```
1. Read the task's items in tasks/phase4-todo.md
2. Write the test file (check off the "create test" item)
3. Run the test, verify RED (check off the "RED" item)
4. Write the implementation file(s) (check off each file item)
5. Run the test, verify GREEN (check off the "GREEN" item)
6. Run regression check — ALL prior tests still pass
7. Git commit with the specified message (check off the "GIT" item)
8. Update "Last Updated" in phase4-todo.md
9. Write any notes under the task
```

### After Both Tasks

1. Run full test suite: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'`
2. Run `git log --oneline` and verify 16 commits
3. Fill in the **"Phase 4 Complete — Signal"** section
4. Do NOT touch the **"Orchestrator Review Checklist"** section

---

## Task-by-Task Reference

### Task 14: CPU Beam Search (Reference Implementation)

**Purpose**: Pure-CPU greedy beam search over the NN-Descent graph. This is the ground truth for GPU search validation.

**Files to create:**
- `Sources/MetalANNS/SearchResult.swift`
- `Sources/MetalANNSCore/BeamSearchCPU.swift`
- `Tests/MetalANNSTests/SearchTests.swift`

**SearchResult** (in `MetalANNS` target):
```swift
public struct SearchResult: Sendable {
    public let id: String
    public let score: Float
    public let internalID: UInt32

    public init(id: String, score: Float, internalID: UInt32) {
        self.id = id
        self.score = score
        self.internalID = internalID
    }
}
```

**BeamSearchCPU** (in `MetalANNSCore` target):
- `public enum BeamSearchCPU` — stateless, all static methods
- `static func search(query:vectors:graph:entryPoint:k:ef:metric:) async throws -> [SearchResult]`
- Parameters:
  - `query: [Float]` — the query vector
  - `vectors: [[Float]]` — all indexed vectors (for distance computation)
  - `graph: [[(UInt32, Float)]]` — the NN-Descent graph (neighbor lists per node)
  - `entryPoint: Int` — starting node ID
  - `k: Int` — number of results to return
  - `ef: Int` — search beam width (ef >= k, larger = better recall, slower)
  - `metric: Metric` — distance metric

**Algorithm — Greedy Beam Search:**
```
1. Initialize visited set with entry point
2. Initialize candidates priority queue (min-heap by distance) with entry point
3. Initialize results list with entry point
4. While candidates is not empty:
   a. Pop the best (smallest distance) unvisited candidate
   b. If this candidate's distance > worst distance in results (and results.count >= ef), break
   c. For each neighbor of the candidate in the graph:
      - If not visited: mark visited, compute distance to query
      - If distance < worst in results (or results.count < ef): add to candidates and results
   d. Keep results sorted, trim to ef
5. Return top-k from results, sorted by distance ascending
```

**Key implementation details:**
- Use a simple sorted array for candidates/results (not a full heap) — correctness first for CPU reference
- Inline distance computation (cosine/l2/innerProduct) same as NNDescentCPU
- Results returned as `[SearchResult]` with `id: ""` (empty string for now — IDMap integration happens in Phase 6)
- The `ef` parameter controls quality/speed tradeoff: ef=k is greedy, ef=2*k is typical, ef=4*k is high-recall

**Tests — 2 tests:**
1. `cpuSearchReturnsK` — 100 nodes, dim=16, degree=8, k=5, ef=32: verify `results.count == k` and results sorted ascending by score
2. `cpuSearchRecall` — 1000 nodes, dim=32, degree=16, k=10, ef=64, 20 random queries: verify average recall > 0.90 vs brute-force

**Commit**: `feat: implement CPU beam search with SearchResult type`

---

### Task 15: GPU Beam Search

**Purpose**: GPU-accelerated beam search. Uses Metal to parallelize distance computation during neighbor expansion.

**Files to create:**
- `Sources/MetalANNSCore/Shaders/Search.metal`
- `Sources/MetalANNSCore/SearchGPU.swift`
- `Tests/MetalANNSTests/MetalSearchTests.swift`

**IMPORTANT**: The implementation plan (Task 15) only provides high-level guidance. This prompt provides the detailed spec. The GPU beam search does NOT fully parallelize the graph traversal (which is inherently sequential). Instead, it **offloads the distance computation batch to GPU** while the traversal logic remains on CPU.

**Architecture — Hybrid CPU/GPU Search:**

The beam search traversal is sequential (pop candidate, expand neighbors, decide next candidate). But the expensive part — computing distances between the query and all expanded neighbors — can be batched and sent to GPU. This gives a practical speedup on large dimension vectors while keeping the algorithm simple.

```
CPU Loop:
  1. Pop best candidate
  2. Collect all unvisited neighbor IDs
  3. Batch-compute distances on GPU (query vs all neighbor vectors)
  4. Update candidates/results with new (id, distance) pairs
  5. Repeat until done
```

**SearchGPU** (in `MetalANNSCore` target):
- `public enum SearchGPU` — stateless, all static methods
- `static func search(context:query:vectors:graph:entryPoint:k:ef:metric:) async throws -> [SearchResult]`
- Parameters same as CPU version, plus `context: MetalContext`, and `vectors: VectorBuffer`, `graph: GraphBuffer`

**Implementation approach:**

1. Start from entry point, compute distance on CPU (just one vector — not worth GPU dispatch)
2. Initialize candidates and results same as CPU version
3. In the expansion loop, collect all unvisited neighbor IDs into a batch
4. For the batch: extract neighbor vectors into a contiguous buffer, dispatch distance kernel
5. Read back distances, update candidates/results
6. The distance kernel reuses the existing `cosine_distance`/`l2_distance`/`inner_product_distance` kernels from `Distance.metal` — query is buffer(0), the batch of neighbor vectors is buffer(1), output is buffer(2), dim and count are buffers 3-4

**DECISION POINT (15.5)**: The plan mentions a full threadgroup-per-query approach with shared memory candidate queue and visited bitset. This is complex and error-prone. The recommended approach is the **hybrid CPU/GPU** approach described above — simpler, still fast for high-dim vectors, and much easier to verify. For batch multi-query search, you can dispatch multiple single searches. **Document your architecture decision in the notes.**

**Alternative simple approach**: If GPU dispatch overhead dominates for small batches, you may implement SearchGPU as a thin wrapper that delegates to BeamSearchCPU for the traversal but uses MetalBackend for the distance computation. This is a valid Phase 4 approach — the full GPU kernel can be optimized in a future phase. **Document if you take this approach.**

**Tests — 2 tests:**
1. `gpuSearchReturnsK` — 100 nodes, dim=16, degree=8, k=5: verify `results.count == k` and sorted
2. `gpuSearchRecall` — 500 nodes, dim=32, degree=16, k=10, ef=64, 10 queries: verify average recall > 0.85 vs brute-force
   - Guard with `guard MTLCreateSystemDefaultDevice() != nil else { return }`

**Commit**: `feat: implement GPU-accelerated beam search`

---

## Decision Points Summary

| # | Decision | Recommended Approach |
|---|----------|---------------------|
| 14.3 | Beam search data structure (heap vs sorted array) | Sorted array for CPU reference — correctness over performance |
| 15.5 | Full GPU kernel vs hybrid CPU/GPU search | Hybrid (recommended) — GPU for distance batches, CPU for traversal logic |
| 15.6 | Reuse Distance.metal kernels vs new Search.metal | Reuse Distance.metal kernels — they already handle all 3 metrics correctly |

---

## Common Failure Modes (Read Before Starting)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cpuSearchRecall` below 0.90 | ef too small or graph quality poor | Increase ef from 64 to 128; verify graph was built with enough iterations |
| Search returns fewer than k results | Beam terminates too early | Ensure beam doesn't break until results.count >= k |
| Search returns duplicate IDs | Visited set not checked | Check visited before adding to candidates |
| Infinite loop in beam search | Candidate never popped or always re-added | Verify candidates are removed after expansion |
| `SearchResult` not visible from test | Wrong target | `SearchResult` is in `MetalANNS` target, test needs `@testable import MetalANNS` |
| GPU search crashes | Buffer size mismatch | Verify neighbor batch buffer is allocated with correct size |
| `NNDescentCPU.build` too slow in test | Large n with high degree/iterations | Use n=100 for basic tests, n=1000 only for recall test |
| Scheme not found | Xcode auto-generates scheme name | Use `MetalANNS-Package` for `xcodebuild test` |

---

## Reference Files

| File | Purpose |
|------|---------|
| `docs/plans/2026-02-25-metalanns-implementation.md` (lines 2371–2513) | Task 14 code reference, Task 15 high-level guidance |
| `Sources/MetalANNSCore/NNDescentCPU.swift` | CPU distance computation pattern (inline cosine/l2/innerProduct) |
| `Sources/MetalANNSCore/MetalBackend.swift` | GPU distance dispatch pattern (buffer encoding, dispatch, readback) |
| `Sources/MetalANNSCore/Shaders/Distance.metal` | Distance kernels to reuse for GPU search |
| `Sources/MetalANNSCore/VectorBuffer.swift` | `.buffer`, `.vector(at:)`, `.floatPointer` |
| `Sources/MetalANNSCore/GraphBuffer.swift` | `.neighborIDs(of:)`, `.adjacencyBuffer`, `.distanceBuffer` |
| `tasks/phase4-todo.md` | **Your checklist** |
| `tasks/lessons.md` | Record any lessons learned |

---

## Scope Boundary (What NOT To Do)

- Do NOT implement Phase 5+ code (Persistence, ANNSIndex actor, Incremental Insert, Soft Deletion)
- Do NOT add batch multi-query GPU search with shared memory candidate queues (that's an optimization for later)
- Do NOT modify Phase 1–3 files unless compilation requires it (document any changes)
- Do NOT use XCTest — Swift Testing exclusively. No `XCTSkip`, no `import XCTest`
- Do NOT use `swift build` or `swift test` — `xcodebuild` only
- Do NOT create README.md or documentation files
- Do NOT modify the Orchestrator Review Checklist in phase4-todo.md
