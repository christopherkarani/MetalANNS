# Phase 8 Execution Prompt: Full GPU Search & Graph Pruning

---

## System Context

You are implementing **Phase 8 (Tasks 24–25)** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1–7 are complete. The codebase has dual compute backends, GPU-resident data structures, CPU and GPU NN-Descent construction, CPU and hybrid GPU beam search, SIMD CPU distances via Accelerate, concurrent batch search via TaskGroup, index serialization, incremental insert, soft deletion, and the `ANNSIndex` public actor API. Git log shows 27 commits. Full test suite has 57 tests, zero failures.

You are building **the most performance-critical components** — a full GPU beam search kernel that performs entire search traversal in a single kernel dispatch (eliminating CPU↔GPU round-trips), and CAGRA-style graph pruning to improve edge quality after construction.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/phase8-todo.md`

**Your communication contract:**
1. **`tasks/phase8-todo.md` is your shared state.** Check off `[x]` items as you complete them.
2. **Write notes under every task** — especially for decision points and any issues you hit.
3. **Update `Last Updated`** at the top of phase8-todo.md after each task completes.
4. **When done, fill in the "Phase 8 Complete — Signal" section** at the bottom.
5. **Do NOT modify the "Orchestrator Review Checklist"** section — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **TDD cycle for every task**: Write test → RED → implement → GREEN → commit. Check off RED and GREEN separately.
2. **Swift 6 strict concurrency**: `ANNSIndex` is an `actor`. All types remain `Sendable`.
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest or `XCTSkip`.
4. **Build and test with `xcodebuild`**. Test scheme is `MetalANNS-Package`. Never `swift build` or `swift test`.
5. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
6. **Commit after every task** with the exact conventional commit message specified in the todo.
7. **Check off todo items in real time**.
8. **Do NOT modify Phase 1–7 source files** unless strictly necessary for wiring in the new code. Acceptable reasons: updating `ANNSIndex.swift` to use `FullGPUSearch` instead of `SearchGPU` in the GPU path, or calling `GraphPruner` after `NNDescentGPU.build`. Document every change in your notes.

---

## What Already Exists (Phases 1–7 Output)

### Package Structure

```
Sources/
  MetalANNS/                      ← Public API target (depends on MetalANNSCore)
    ANNSIndex.swift               ← Public actor: build/search/insert/delete/save/load/batchSearch
    IndexConfiguration.swift      ← Public config struct (Codable)
    SearchResult.swift            ← typealias to MetalANNSCore.SearchResult
    Errors.swift                  ← typealias to MetalANNSCore.ANNSError
    Metric.swift                  ← typealias to MetalANNSCore.Metric
  MetalANNSCore/                  ← Internal implementation target
    Shaders/
      Distance.metal              ← GPU distance kernels (cosine_distance, l2_distance, inner_product_distance)
      NNDescent.metal             ← GPU construction kernels (random_init, compute_initial_distances, build_reverse_list, local_join)
      Sort.metal                  ← Bitonic sort kernel (bitonic_sort_neighbors)
    BeamSearchCPU.swift           ← CPU beam search (uses SIMDDistance)
    SearchGPU.swift               ← Hybrid CPU/GPU search (CPU traversal + GPU distance batch dispatch)
    SIMDDistance.swift             ← Accelerate vDSP distance (cosine, l2, innerProduct)
    NNDescentCPU.swift            ← CPU graph construction
    NNDescentGPU.swift            ← GPU graph construction (randomInit → computeInitialDistances → iterations → bitonicSort)
    IncrementalBuilder.swift      ← Incremental insert (uses SIMDDistance)
    IndexSerializer.swift         ← Binary serialization (.mann format)
    SoftDeletion.swift            ← Soft deletion with filtered results
    VectorBuffer.swift            ← GPU-resident flat Float32 buffer (.storageModeShared)
    GraphBuffer.swift             ← GPU-resident adjacency + distance arrays (.storageModeShared)
    IDMap.swift                   ← Bidirectional String ↔ UInt32 mapping
    MetalDevice.swift             ← MetalContext (device, commandQueue, library, pipelineCache)
    Errors.swift, Metric.swift, SearchResult.swift, IndexConfiguration.swift, etc.
  MetalANNSBenchmarks/
    BenchmarkRunner.swift, main.swift
Tests/
  MetalANNSTests/                 ← 57 tests across 21 suites
```

### Current GPU Search Path (What You're Replacing)

The current `SearchGPU.search()` at `Sources/MetalANNSCore/SearchGPU.swift` is a **hybrid** approach:
- CPU manages the beam traversal (visited set, candidate queue, result list)
- For each expansion step, CPU collects unvisited neighbor IDs
- CPU dispatches a GPU command buffer to compute distances for those neighbors (via Distance.metal kernels)
- GPU results are read back to CPU
- CPU updates candidate queue and results

**Problem**: Each beam step requires a full CPU→GPU→CPU round-trip. For `ef=64` on a graph with `degree=16`, this means ~64 separate GPU command submissions.

**Solution (Task 24)**: A single Metal kernel that performs the entire beam traversal in shared memory. One kernel dispatch per query.

### Key API Signatures

```swift
// MetalContext — execute pattern
public func execute(_ encode: (MTLCommandBuffer) throws -> Void) async throws

// PipelineCache — get compiled pipeline
public func pipeline(for functionName: String) async throws -> MTLComputePipelineState

// GraphBuffer — adjacency is flat: adjacency[nodeID * degree + slot]
public let adjacencyBuffer: MTLBuffer  // [UInt32] nodeCount * degree
public let distanceBuffer: MTLBuffer   // [Float] nodeCount * degree (parallel distances)
public let degree: Int
public var nodeCount: Int

// VectorBuffer — vectors are flat: buffer[nodeID * dim + d]
public let buffer: MTLBuffer  // [Float] capacity * dim
public let dim: Int

// SearchResult
public struct SearchResult: Sendable {
    public let id: String       // "" for internal searches
    public let score: Float     // distance value
    public let internalID: UInt32
}

// NNDescentCPU.build returns:
(graph: [[(UInt32, Float)]], entryPoint: UInt32)

// SIMDDistance.distance (used in recall comparison)
public static func cosine(_ a: [Float], _ b: [Float]) -> Float

// ANNSIndex.search currently uses SearchGPU when context != nil (lines 174-184)
if let context {
    rawResults = try await SearchGPU.search(
        context: context, query: query, vectors: vectors, graph: graph,
        entryPoint: Int(entryPoint), k: max(1, effectiveK),
        ef: max(1, effectiveEf), metric: configuration.metric
    )
}
```

### Metal Shared Memory Budget

Apple Silicon threadgroup memory is typically **32KB**. The kernel must fit within this:
- `MAX_EF = 256`: candidates array = `256 * 8 bytes` = 2KB
- `MAX_EF = 256`: results array = `256 * 8 bytes` = 2KB
- `MAX_VISITED = 4096`: visited hash table = `4096 * 4 bytes` = 16KB
- Atomic counters + head pointer: ~20 bytes
- **Total: ~20KB** — fits in 32KB with room to spare

### Metal Atomics on Apple Silicon

- Only `memory_order_relaxed` is supported (no acquire/release)
- 32-bit atomics: `atomic_uint`, `atomic_compare_exchange_weak_explicit`, `atomic_fetch_add_explicit`, `atomic_load_explicit`, `atomic_store_explicit`
- `as_type<uint>(float)` preserves comparison order for non-negative floats (used elsewhere, not needed here)

---

## Task 24: Full GPU Beam Search Kernel

### Goal

Create a Metal kernel (`beam_search`) that performs the entire beam search traversal in a single GPU dispatch. The kernel uses shared memory for the candidate queue, result list, and visited hash table. One threadgroup processes one query; threads within the group cooperate on neighbor expansion and distance computation.

### Architecture

```
Single Kernel Dispatch (1 threadgroup)
├── Thread 0: manages candidate queue + result list
├── All threads: clear visited hash table
├── Thread 0: seed entry point
└── Loop (up to ef_limit * 2 iterations):
    ├── Thread 0: pop best candidate, check termination
    ├── All threads (parallel): expand neighbors
    │   ├── Check visited (open-addressed hash table with CAS)
    │   ├── Compute distance (inline, per-thread)
    │   └── Append to results + candidates (atomic slot allocation)
    └── Thread 0: sort and trim results + candidates (insertion sort)
```

### What to Create

**1. `Sources/MetalANNSCore/Shaders/Search.metal`** — The beam search kernel.

Buffer bindings:
- `buffer(0)`: `device const float* vectors` — flat `[nodeCount * dim]`
- `buffer(1)`: `device const uint* adjacency` — flat `[nodeCount * degree]` (from `GraphBuffer.adjacencyBuffer`)
- `buffer(2)`: `device const float* query` — flat `[dim]`
- `buffer(3)`: `device float* output_dists` — flat `[k]` output
- `buffer(4)`: `device uint* output_ids` — flat `[k]` output
- `buffer(5)`: `constant uint& node_count`
- `buffer(6)`: `constant uint& degree`
- `buffer(7)`: `constant uint& dim`
- `buffer(8)`: `constant uint& k`
- `buffer(9)`: `constant uint& ef` — clamped to MAX_EF=256
- `buffer(10)`: `constant uint& entry_point`
- `buffer(11)`: `constant uint& metric_type` — 0=cosine, 1=l2, 2=innerProduct

Shared memory structures:
- `threadgroup CandidateEntry candidates[MAX_EF]` — sorted candidate queue
- `threadgroup CandidateEntry results[MAX_EF]` — sorted result list
- `threadgroup atomic_uint visited[MAX_VISITED]` — open-addressed hash table (UINT_MAX = empty)
- `threadgroup atomic_uint candidate_count, result_count` — atomic size counters
- `threadgroup uint candidate_head` — next candidate to expand

`CandidateEntry` struct: `{ uint nodeID; float distance; }`

Inline `compute_distance()`: same logic as Distance.metal but reads from the flat vectors buffer using `vectors + nodeID * dim`.

Inline `try_visit()`: open-addressed hash insert with linear probing (up to 32 probes). Uses `atomic_compare_exchange_weak_explicit` with `memory_order_relaxed`. Returns `true` if newly inserted (not visited before), `false` if already present.

Main loop: up to `ef_limit * 2` iterations. Thread 0 pops best candidate. All threads expand neighbors in parallel. Thread 0 sorts and trims after each expansion. Barrier after each phase.

Output: thread 0 writes top-k results to output buffers, pads with `UINT_MAX`/`FLT_MAX` sentinels.

**2. `Sources/MetalANNSCore/FullGPUSearch.swift`** — Swift dispatch wrapper.

```swift
public enum FullGPUSearch {
    public static func search(
        context: MetalContext,
        query: [Float],
        vectors: VectorBuffer,
        graph: GraphBuffer,
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws -> [SearchResult]
}
```

Implementation:
1. Get `beam_search` pipeline from cache
2. Create query buffer (`.storageModeShared`)
3. Create output distance and ID buffers (each `k * 4` bytes)
4. Pack scalar constants: nodeCount, degree, dim, k, ef (clamped to 256), entryPoint, metricType (0/1/2)
5. Dispatch: 1 threadgroup, `min(degree, pipeline.maxTotalThreadsPerThreadgroup)` threads per threadgroup
6. Read output buffers, construct `[SearchResult]` (skip sentinels where `nodeID == UInt32.max`)

**3. Wire into ANNSIndex** — In `ANNSIndex.swift`, replace `SearchGPU.search(...)` call with `FullGPUSearch.search(...)` in the GPU path of the `search` method.

### Decision Points

- **24.1**: Threadgroup size. The plan uses `min(degree, maxThreads)`. For degree=16 this means only 16 threads — most of the threadgroup is underutilized. Alternative: use a larger fixed threadgroup (64 or 128) so more threads can compute distances in parallel during neighbor expansion. **Evaluate and document choice.**

- **24.2**: The kernel uses insertion sort (O(n²)) on thread 0 after each expansion step. For `ef ≤ 256` this is acceptable, but if you observe performance issues, consider parallel merge sort or keeping results unsorted until output. **Document observations.**

- **24.3**: `SearchGPU.swift` remains in the codebase (not deleted) as a fallback and for backward compatibility. The new `FullGPUSearch` is the preferred path. **Confirm this approach in notes.**

- **24.4**: If the kernel doesn't compile or has shared memory issues, you may need to reduce `MAX_VISITED` or `MAX_EF`. Valid adjustments: `MAX_VISITED = 2048` (saves 8KB), `MAX_EF = 128` (saves 2KB). **Document any adjustments.**

### Tests (2 tests)

**`Tests/MetalANNSTests/FullGPUSearchTests.swift`**:

1. **`gpuSearchReturnsK`**:
   - Guard: `MTLCreateSystemDefaultDevice() != nil else { return }`
   - Build small graph: 200 nodes, dim=16, degree=8 via `NNDescentCPU.build`
   - Populate `VectorBuffer` + `GraphBuffer` from CPU graph
   - Query: random 16-dim vector
   - Call `FullGPUSearch.search(context:query:vectors:graph:entryPoint:k:5 ef:32 metric:.cosine)`
   - Assert: `results.count == 5`
   - Assert: results sorted by score ascending (`results[i].score >= results[i-1].score`)

2. **`gpuSearchRecallMatchesHybrid`**:
   - Guard: `MTLCreateSystemDefaultDevice() != nil else { return }`
   - Build graph: 500 nodes, dim=32, degree=16 via `NNDescentCPU.build`
   - 20 random queries, k=10, ef=64
   - For each query: compute brute-force ground truth using `SIMDDistance.cosine`
   - Compare recall: `FullGPUSearch` vs `SearchGPU` (existing hybrid) vs ground truth
   - Assert: `fullGPURecall > hybridRecall - 0.05` (within 5% of hybrid)
   - Assert: `fullGPURecall > 0.80` (absolute floor)

### Test Setup Helper Pattern

Both tests need to populate `GraphBuffer` from `NNDescentCPU` output. Use this inline pattern:

```swift
let (cpuGraph, cpuEntry) = try await NNDescentCPU.build(
    vectors: vectors, degree: degree, metric: .cosine, maxIterations: 10
)
let vb = try VectorBuffer(capacity: n, dim: dim, device: context.device)
try vb.batchInsert(vectors: vectors, startingAt: 0)
vb.setCount(n)
let gb = try GraphBuffer(capacity: n, degree: degree, device: context.device)
for i in 0..<n {
    let ids = cpuGraph[i].map(\.0)
    let dists = cpuGraph[i].map(\.1)
    var paddedIDs = ids + Array(repeating: UInt32.max, count: max(0, degree - ids.count))
    var paddedDists = dists + Array(repeating: Float.greatestFiniteMagnitude, count: max(0, degree - dists.count))
    paddedIDs = Array(paddedIDs.prefix(degree))
    paddedDists = Array(paddedDists.prefix(degree))
    try gb.setNeighbors(of: i, ids: paddedIDs, distances: paddedDists)
}
gb.setCount(n)
```

### Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `beam_search` not found | Pipeline cache can't find function in library | Verify `Search.metal` is in `Sources/MetalANNSCore/Shaders/` directory (picked up by `.process("Shaders")` in Package.swift) |
| Shared memory exceeded | `MAX_VISITED` or `MAX_EF` too large | Reduce `MAX_VISITED` to 2048, verify with `pipeline.maxTotalThreadsPerThreadgroup` |
| GPU hang / timeout | Infinite loop in kernel | Add iteration cap to main loop; verify `candidate_head` always advances |
| Zero results | Entry point not seeded properly | Verify barrier after seed; verify `try_visit` returns true for first insert |
| Poor recall vs hybrid | Hash table too small (many collisions → missed nodes) | Increase `MAX_VISITED` or reduce probing limit |
| Crash on buffer access | Off-by-one in `nodeID * degree + slot` | Verify `nodeID < node_count` before any adjacency access |
| Sorting produces wrong order | Insertion sort indexing error | Verify `j >= 0` check and `results[j+1] = results[j]` direction |
| `results.count < k` | Not enough nodes explored | Increase `ef` or check loop termination condition |

---

## Task 25: CAGRA Post-Processing (Graph Pruning)

### Goal

Implement CAGRA-style path-based graph pruning to improve edge quality after NN-Descent construction. This removes redundant edges where a neighbor is already reachable through a shorter path via another neighbor, producing a sparser, higher-quality graph.

### Algorithm — Path-Based Pruning

For each node `u` in the graph:
1. Get current neighbors sorted by distance (already sorted after bitonic sort)
2. Initialize empty pruned neighbor list
3. For each candidate neighbor `v` (in ascending distance order):
   - For each already-selected pruned neighbor `w`:
     - If `d(w, v) < d(u, v)`: `v` is **redundant** (reachable via `w` with shorter hop)
   - If no pruned neighbor makes `v` redundant → add `v` to pruned list
4. Pad pruned list to `degree` with sentinel values (`UInt32.max`, `Float.greatestFiniteMagnitude`)

This is the **diversification** step from CAGRA/DiskANN. It ensures each neighbor covers a different "direction" from node `u`.

### What to Create

**`Sources/MetalANNSCore/GraphPruner.swift`**:

```swift
public enum GraphPruner {
    /// Prune redundant edges from the graph using path-based diversification.
    /// For each node, removes neighbors that are reachable via shorter paths
    /// through other neighbors.
    public static func prune(
        graph: GraphBuffer,
        vectors: VectorBuffer,
        nodeCount: Int,
        metric: Metric
    ) throws
}
```

Implementation:
- Iterate over each node 0..<nodeCount
- Read current neighbors from `graph.neighborIDs(of:)` and `graph.neighborDistances(of:)`
- Filter out sentinels (`UInt32.max`)
- For each candidate `v` (in distance order), check against all already-selected pruned neighbors `w`
- Compute `d(w, v)` using `SIMDDistance.distance(vectors.vector(at: w), vectors.vector(at: v), metric:)`
- If `d(w, v) < d(u, v)` for any `w` in pruned list → skip `v`
- Otherwise → add `v` to pruned list
- Write back pruned list padded to `degree`

**Note**: This is a CPU implementation. GPU pruning is a future optimization. The CPU version is correct and sufficient since pruning runs once after build (not on the hot search path).

### What to Modify

**`Sources/MetalANNS/ANNSIndex.swift`** — After the graph construction step in `build()`, optionally call pruning:

```swift
// After NNDescentGPU.build or NNDescentCPU + graphBuffer population:
try GraphPruner.prune(
    graph: graphBuffer,
    vectors: vectorBuffer,
    nodeCount: inputVectors.count,
    metric: configuration.metric
)
```

### Decision Points

- **25.1**: Should pruning be always-on or opt-in via `IndexConfiguration`? Recommended: always-on since it improves quality at minimal build cost. But if build time regression is > 20%, make it configurable. **Document choice.**

- **25.2**: The pruning alpha parameter. CAGRA uses `alpha = 1.0` (strict: neighbor is redundant if any other neighbor is closer to it). Relaxed pruning uses `alpha > 1.0` (e.g., 1.2 means only prune if the alternative path is at least 20% shorter). Start with `alpha = 1.0` for maximum pruning. **Document if adjusted.**

- **25.3**: After pruning, the graph may have nodes with very few neighbors (aggressive pruning). If average neighbor count drops below `degree / 2`, this is too aggressive. **Monitor and document average post-pruning neighbor count.**

### Tests (2 tests)

**`Tests/MetalANNSTests/GraphPrunerTests.swift`**:

1. **`pruningReducesRedundancy`**:
   - Build graph: 200 nodes, dim=16, degree=8 via `NNDescentCPU.build`
   - Populate `GraphBuffer`
   - Count total valid (non-sentinel) edges before pruning
   - Call `GraphPruner.prune(...)`
   - Count total valid edges after pruning
   - Assert: `edgesAfter <= edgesBefore` (pruning only removes, never adds)
   - Assert: `edgesAfter > 0` (graph not completely emptied)
   - Assert: average neighbors per node > `degree / 2` (not over-pruned)

2. **`pruningMaintainsRecall`**:
   - Guard: `MTLCreateSystemDefaultDevice() != nil else { return }`
   - Build graph: 300 nodes, dim=32, degree=16 via `NNDescentCPU.build`
   - Populate `VectorBuffer` + `GraphBuffer`
   - Measure baseline recall: 20 queries, k=10, ef=64, brute-force ground truth
   - Call `GraphPruner.prune(...)`
   - Measure pruned recall: same queries, same parameters
   - Assert: `prunedRecall > baselineRecall - 0.02` (no more than 2% recall drop)
   - Note: recall may actually IMPROVE since pruning diversifies edges

### Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All edges pruned | Alpha too aggressive or distance comparison wrong | Check `d(w,v) < d(u,v)` direction; verify metric consistency |
| No edges pruned | Alpha too relaxed or candidates not sorted | Verify neighbors are processed in ascending distance order |
| Build time regresses > 2x | CPU pruning too slow for large graphs | Add early termination; consider only pruning first `degree` candidates |
| Recall drops > 5% | Over-pruning | Increase alpha or add minimum neighbor count floor |
| `SIMDDistance` not available | Wrong import | `GraphPruner` is in `MetalANNSCore` target, `SIMDDistance` is same target — should work |

---

## Scope Boundary

**In scope for Phase 8:**
- `Search.metal` — full GPU beam search kernel (Task 24)
- `FullGPUSearch.swift` — Swift dispatch wrapper (Task 24)
- Wiring `FullGPUSearch` into `ANNSIndex.swift` GPU search path (Task 24)
- `GraphPruner.swift` — CPU path-based pruning (Task 25)
- Calling `GraphPruner` in `ANNSIndex.build()` (Task 25)
- New test files: `FullGPUSearchTests.swift`, `GraphPrunerTests.swift`

**NOT in scope:**
- Float16 (Phase 9)
- Batch insert, compaction, mmap (Phase 10)
- Filtered search, range search (Phase 11)
- Deleting `SearchGPU.swift` (kept as fallback)
- GPU-accelerated pruning (future optimization)
- Changes to `Distance.metal`, `NNDescent.metal`, or `Sort.metal`
- Changes to `Package.swift`

---

## Reference Files

Read these before starting:
- `tasks/phase8-todo.md` — your shared task checklist
- `Sources/MetalANNSCore/SearchGPU.swift` — current hybrid search (understand what you're replacing)
- `Sources/MetalANNSCore/Shaders/Distance.metal` — existing distance kernel patterns
- `Sources/MetalANNSCore/MetalDevice.swift` — `MetalContext.execute()` pattern, `PipelineCache`
- `Sources/MetalANNSCore/GraphBuffer.swift` — `adjacencyBuffer`, `distanceBuffer`, `degree`, `nodeCount`
- `Sources/MetalANNSCore/VectorBuffer.swift` — `buffer`, `dim`
- `Sources/MetalANNSCore/SIMDDistance.swift` — CPU distance functions (for test recall comparison)
- `Sources/MetalANNS/ANNSIndex.swift` — `search` method lines 158-205 (GPU path to modify)
- `docs/plans/2026-02-25-metalanns-v2-performance-features.md` — overall v2 plan (Phase 8 section)

---

## Verification Commands

```bash
# Run only FullGPUSearch tests (Task 24 RED/GREEN)
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/FullGPUSearchTests 2>&1 | grep -E '(PASS|FAIL|error:)'

# Run only GraphPruner tests (Task 25 RED/GREEN)
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/GraphPrunerTests 2>&1 | grep -E '(PASS|FAIL|error:)'

# Full regression suite
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'

# Verify test count (expected: 61 = 57 prior + 2 from Task 24 + 2 from Task 25)
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | tail -3

# Git log check
git log --oneline | head -5
```
