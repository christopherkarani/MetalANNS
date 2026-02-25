# Phase 6 Execution Prompt: Public API & Polish

---

## System Context

You are implementing **Phase 6 (Tasks 19–21)** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1–5 are complete. The codebase has dual compute backends, GPU-resident data structures, CPU and GPU NN-Descent construction, CPU and GPU beam search, index serialization, incremental insert, and soft deletion. Git log shows 21 commits.

You are building the **public-facing API** — the `ANNSIndex` actor that wires all internal components into a single ergonomic interface, a comprehensive integration test, a benchmark runner, and release documentation. This is the **final phase**.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/phase6-todo.md`

**Your communication contract:**
1. **`tasks/phase6-todo.md` is your shared state.** Check off `[x]` items as you complete them.
2. **Write notes under every task** — especially for decision points and any issues you hit.
3. **Update `Last Updated`** at the top of phase6-todo.md after each task completes.
4. **When done, fill in the "Phase 6 Complete — Signal" section** at the bottom.
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
8. **Do NOT modify Phase 1–5 source files** unless strictly necessary. If you must modify a Phase 1–5 file, document exactly what changed and why in your notes. Acceptable reasons: adding `Codable` conformance to `IndexConfiguration`, adding a re-export typealias in `MetalANNS`, or fixing a compilation issue.

---

## What Already Exists (Phases 1–5 Output)

### Package Structure

```
Sources/
  MetalANNS/                      ← Public API target (depends on MetalANNSCore)
    ANNSIndex.swift               ← Currently just `import MetalANNSCore` — YOU IMPLEMENT THIS
    IndexConfiguration.swift      ← Public config struct
    SearchResult.swift            ← typealias to MetalANNSCore.SearchResult
    Errors.swift                  ← typealias to MetalANNSCore.ANNSError
    Metric.swift                  ← typealias to MetalANNSCore.Metric
  MetalANNSCore/                  ← Internal implementation target
    (all internal types — see below)
  MetalANNSBenchmarks/            ← Executable target
    main.swift                    ← Currently just `print("MetalANNS Benchmarks")`
Tests/
  MetalANNSTests/                 ← Test target (depends on MetalANNS + MetalANNSCore)
```

### Key Types You'll Wire Together

```swift
// ═══════════ CONSTRUCTION ═══════════

// CPU NN-Descent — returns graph as array + entry point
NNDescentCPU.build(
    vectors: [[Float]], degree: Int, metric: Metric,
    maxIterations: Int, convergenceThreshold: Float
) async throws -> (graph: [[(UInt32, Float)]], entryPoint: UInt32)

// GPU NN-Descent — builds in-place on GraphBuffer
NNDescentGPU.build(
    context: MetalContext, vectors: VectorBuffer, graph: GraphBuffer,
    nodeCount: Int, metric: Metric,
    maxIterations: Int, convergenceThreshold: Float
) async throws
// NOTE: GPU build does NOT return an entry point. You must pick one (see DECISION POINT 19.5).

// GPU sort — sorts each node's neighbor list by distance
NNDescentGPU.sortNeighborLists(context: MetalContext, graph: GraphBuffer, nodeCount: Int) async throws

// ═══════════ SEARCH ═══════════

// CPU beam search — operates on [[Float]] vectors and [[(UInt32, Float)]] graph
BeamSearchCPU.search(
    query: [Float], vectors: [[Float]], graph: [[(UInt32, Float)]],
    entryPoint: Int, k: Int, ef: Int, metric: Metric
) async throws -> [SearchResult]

// GPU beam search — operates on VectorBuffer + GraphBuffer
SearchGPU.search(
    context: MetalContext, query: [Float], vectors: VectorBuffer,
    graph: GraphBuffer, entryPoint: Int, k: Int, ef: Int, metric: Metric
) async throws -> [SearchResult]

// ═══════════ MUTATION ═══════════

// Incremental insert — operates directly on GraphBuffer + VectorBuffer
IncrementalBuilder.insert(
    vector: [Float], at internalID: Int, into graph: GraphBuffer,
    vectors: VectorBuffer, entryPoint: UInt32, metric: Metric, degree: Int
) throws

// Soft deletion — pure filter struct
var softDeletion = SoftDeletion()    // Sendable, Codable
softDeletion.markDeleted(internalID)
softDeletion.isDeleted(internalID) -> Bool
softDeletion.filterResults(results) -> [SearchResult]
softDeletion.deletedCount -> Int

// ═══════════ PERSISTENCE ═══════════

// Save core index data to binary file
IndexSerializer.save(
    vectors: VectorBuffer, graph: GraphBuffer, idMap: IDMap,
    entryPoint: UInt32, metric: Metric, to: URL
) throws

// Load core index data from binary file
IndexSerializer.load(from: URL, device: MTLDevice?) throws
    -> (vectors: VectorBuffer, graph: GraphBuffer, idMap: IDMap, entryPoint: UInt32, metric: Metric)

// ═══════════ DATA STRUCTURES ═══════════

VectorBuffer(capacity: Int, dim: Int, device: MTLDevice?)
  .insert(vector:at:), .batchInsert(vectors:startingAt:), .setCount(_:)
  .vector(at:) -> [Float], .count, .dim, .capacity, .buffer

GraphBuffer(capacity: Int, degree: Int, device: MTLDevice?)
  .setNeighbors(of:ids:distances:), .neighborIDs(of:) -> [UInt32]
  .neighborDistances(of:) -> [Float], .setCount(_:)
  .nodeCount, .degree, .capacity, .adjacencyBuffer, .distanceBuffer

IDMap()  // Sendable, Codable
  .assign(externalID:) -> UInt32?, .internalID(for:) -> UInt32?
  .externalID(for:) -> String?, .count

MetalContext()  // throws if no Metal device
  .device, .commandQueue, .pipelineCache
  .execute(_:) async throws

IndexConfiguration(degree: 32, metric: .cosine, efConstruction: 100,
    efSearch: 64, maxIterations: 20, useFloat16: false, convergenceThreshold: 0.001)

Metric.cosine / .l2 / .innerProduct  // Codable, Sendable

SearchResult(id: String, score: Float, internalID: UInt32)  // Sendable
```

### Existing Tests (must not regress)
- Phases 1–4: 36 tests
- Phase 5: 8 tests (PersistenceTests ×3, IncrementalTests ×2, DeletionTests ×3)
- **Total: 44 tests** — all must continue passing

---

## Success Criteria

Phase 6 is done when ALL of the following are true:

- [ ] `ANNSIndex` is a Swift actor with `build`, `insert`, `delete`, `search`, `batchSearch`, `save`, `load`, and `count`
- [ ] Full lifecycle works: init → build → search → insert → delete → save → load → search
- [ ] External string IDs are mapped correctly (results contain external IDs, not empty strings)
- [ ] GPU path used when Metal is available, CPU fallback otherwise
- [ ] Soft-deleted vectors never appear in search results via public API
- [ ] Integration test passes with recall@10 > 0.90 on 500+ vectors
- [ ] BenchmarkRunner prints p50/p95/p99 latency and recall metrics
- [ ] README.md documents the public API with usage examples
- [ ] All new tests pass AND all Phase 1–5 tests still pass (zero regressions)
- [ ] Git history has exactly 24 commits (21 prior + 3 new)
- [ ] `tasks/phase6-todo.md` has all items checked and the completion signal filled in

---

## Execution Instructions

### Before You Start

1. Read `tasks/phase6-todo.md` — this is your checklist.
2. Read `docs/plans/2026-02-25-metalanns-implementation.md` (Tasks 19–21, lines ~2567–2694) — high-level guidance. This prompt provides the detailed spec.
3. Run the full test suite to confirm Phases 1–5 are green: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
4. Complete the **Pre-Flight Checks** in phase6-todo.md.

### For Each Task (19 through 21)

```
1. Read the task's items in tasks/phase6-todo.md
2. Write the test file (check off the "create test" item)
3. Run the test, verify RED (check off the "RED" item)
4. Write the implementation file(s) (check off each file item)
5. Run the test, verify GREEN (check off the "GREEN" item)
6. Run regression check — ALL prior tests still pass
7. Git commit with the specified message (check off the "GIT" item)
8. Update "Last Updated" in phase6-todo.md
9. Write any notes under the task
```

### After All 3 Tasks

1. Run full test suite: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'`
2. Run `git log --oneline` and verify 24 commits
3. Fill in the **"Phase 6 Complete — Signal"** section
4. Do NOT touch the **"Orchestrator Review Checklist"** section

---

## Task-by-Task Reference

### Task 19: ANNSIndex Actor (Public API)

**Purpose**: Create the single public-facing actor that consumers interact with. It wires all internal components together behind an ergonomic API.

**Files to modify/create:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (currently just `import MetalANNSCore`)
- Create: `Tests/MetalANNSTests/ANNSIndexTests.swift`
- Possibly modify: `Sources/MetalANNS/IndexConfiguration.swift` (add `Codable` conformance if not present)

**ANNSIndex actor** (in `MetalANNS` target):

```swift
public actor ANNSIndex {
    // ── Public API ──
    public init(configuration: IndexConfiguration = .default)
    public func build(vectors: [[Float]], ids: [String]) async throws
    public func insert(_ vector: [Float], id: String) async throws
    public func delete(id: String) throws
    public func search(query: [Float], k: Int) async throws -> [SearchResult]
    public func batchSearch(queries: [[Float]], k: Int) async throws -> [[SearchResult]]
    public func save(to url: URL) async throws
    public static func load(from url: URL) async throws -> ANNSIndex
    public var count: Int { get }
}
```

**Internal state the actor must own:**
- `configuration: IndexConfiguration`
- `context: MetalContext?` — optional, nil when no GPU available
- `vectors: VectorBuffer?` — nil until `build` is called
- `graph: GraphBuffer?` — nil until `build` is called
- `idMap: IDMap`
- `softDeletion: SoftDeletion`
- `entryPoint: UInt32`
- `isBuilt: Bool` — guard property to prevent search before build

**Method implementations:**

1. **`init(configuration:)`**:
   - Store configuration
   - Try to create `MetalContext()` — if it throws, set `context = nil` (CPU fallback mode)
   - Initialize empty `IDMap` and `SoftDeletion`
   - `isBuilt = false`

2. **`build(vectors:ids:)`**:
   - Validate: `vectors.count == ids.count`, not empty, all vectors same dimension
   - Assign all IDs via `IDMap` — throw `ANNSError.idAlreadyExists` if duplicate
   - Allocate `VectorBuffer` with **extra capacity** (`vectors.count * 2`) for future inserts
   - Allocate `GraphBuffer` with same extra capacity
   - Batch-insert vectors into `VectorBuffer`, set count
   - **If GPU available** (`context != nil`):
     - Call `NNDescentGPU.build(context:vectors:graph:nodeCount:metric:maxIterations:convergenceThreshold:)`
     - GPU build sets `graph.nodeCount` internally via `graph.setCount(nodeCount)`
     - Pick entry point (see DECISION POINT 19.5)
   - **If CPU fallback** (`context == nil`):
     - Call `NNDescentCPU.build(vectors:degree:metric:maxIterations:convergenceThreshold:)`
     - Transfer CPU graph `[[(UInt32, Float)]]` into `GraphBuffer` via `setNeighbors(of:ids:distances:)`
     - Pad neighbor lists to `degree` with `UInt32.max` / `Float.greatestFiniteMagnitude` if needed
     - Use `cpuResult.entryPoint` as the entry point
     - Set `graph.setCount(vectors.count)`
   - Reset `softDeletion` to fresh `SoftDeletion()`
   - Set `isBuilt = true`

3. **`search(query:k:)`**:
   - Guard: `isBuilt`, vectors/graph not nil, else throw `ANNSError.indexEmpty`
   - Validate: `query.count == vectors.dim`
   - Compute effective ef: `max(configuration.efSearch, k + softDeletion.deletedCount)` — search wider to compensate for deletions
   - Compute effective k: `k + softDeletion.deletedCount` — request extra to ensure k after filtering
   - **If GPU** (`context != nil`):
     - Call `SearchGPU.search(context:query:vectors:graph:entryPoint:k:ef:metric:)`
   - **If CPU**:
     - Extract vectors and graph from buffers to arrays (helper method)
     - Call `BeamSearchCPU.search(query:vectors:graph:entryPoint:k:ef:metric:)`
   - Filter through `softDeletion.filterResults(results)`
   - Map `internalID` → external `id` via `idMap.externalID(for:)` on each result
   - Trim to `k` results
   - Return results

4. **`batchSearch(queries:k:)`**:
   - Loop over queries, call `search(query:k:)` for each
   - Return array of result arrays
   - (Sequential is fine for correctness — parallel optimization is future work)

5. **`insert(_:id:)`**:
   - Guard: `isBuilt`, else throw `ANNSError.indexEmpty`
   - Validate: dimension matches, id not already in `IDMap`
   - Assign ID via `idMap.assign(externalID:)` — throw `ANNSError.idAlreadyExists` if nil
   - Insert vector into `VectorBuffer` at the new internal ID slot
   - Update `vectors.setCount` if needed
   - Call `IncrementalBuilder.insert(vector:at:into:graph:vectors:entryPoint:metric:degree:)`
   - Update `graph.setCount` if needed

6. **`delete(id:)`**:
   - Lookup internal ID via `idMap.internalID(for:)` — throw `ANNSError.idNotFound` if nil
   - Call `softDeletion.markDeleted(internalID)`

7. **`save(to:)`**:
   - Guard: `isBuilt`, vectors/graph not nil
   - Call `IndexSerializer.save(vectors:graph:idMap:entryPoint:metric:to:)`
   - **Also persist SoftDeletion and configuration** (see DECISION POINT 19.6)

8. **`load(from:)`** — `static`:
   - Call `IndexSerializer.load(from:device:)` to get core data
   - Create a new `ANNSIndex` with appropriate configuration
   - Set all internal state from loaded data
   - Return the configured actor
   - **Also load SoftDeletion** if persisted (see DECISION POINT 19.6)

9. **`count`** — computed:
   - Return `idMap.count - softDeletion.deletedCount`
   - (Or just `idMap.count` if you prefer to count total assigned, not active)

**Helper methods you'll likely need:**

```swift
// Extract VectorBuffer contents to [[Float]] for CPU search fallback
private func extractVectors() -> [[Float]] {
    guard let vectors else { return [] }
    return (0..<vectors.count).map { vectors.vector(at: $0) }
}

// Extract GraphBuffer to [[(UInt32, Float)]] for CPU search fallback
private func extractGraph() -> [[(UInt32, Float)]] {
    guard let graph else { return [] }
    return (0..<graph.nodeCount).map { nodeID in
        let ids = graph.neighborIDs(of: nodeID)
        let distances = graph.neighborDistances(of: nodeID)
        return zip(ids, distances)
            .filter { $0.0 != UInt32.max }
            .map { ($0.0, $0.1) }
    }
}
```

**Tests — 5 tests:**

1. `buildAndSearch`:
   - Create `ANNSIndex` with `IndexConfiguration(degree: 8, metric: .cosine)`
   - Build with 100 random 16-dim vectors, string IDs "vec_0" through "vec_99"
   - Search for `vectors[0]` with k=5
   - Assert: `results.count == 5`
   - Assert: `results[0].id == "vec_0"` (exact match should be first)
   - Assert: `results[0].score` is near 0.0 (cosine distance to itself)
   - Assert: all results have non-empty `id` fields

2. `insertAndSearch`:
   - Build index with 50 vectors
   - Insert 5 new vectors with IDs "new_0" through "new_4"
   - Search for each newly inserted vector
   - Assert: each new vector appears as its own top-1 result
   - Assert: `index.count` increased by 5

3. `deleteAndSearch`:
   - Build index with 50 vectors
   - Delete "vec_0" and "vec_5"
   - Search with k=50 (get all)
   - Assert: "vec_0" and "vec_5" are NOT in results
   - Assert: `index.count` decreased by 2

4. `saveAndLoadLifecycle`:
   - Build index with 100 vectors
   - Insert 5 new vectors, delete 2 vectors
   - Save to temp file
   - Load from temp file
   - Search on loaded index — verify results match original index's results
   - Verify `count` matches
   - Clean up temp file

5. `batchSearchReturnsCorrectShape`:
   - Build index with 50 vectors
   - Batch search with 3 queries, k=5
   - Assert: result is array of 3 arrays, each with 5 results

**DECISION POINT (19.5)**: Entry point for GPU builds. `NNDescentGPU.build` doesn't return an entry point. Options:
- (a) Use node 0 — simplest, works well in practice for random data
- (b) Find the node closest to the centroid — more robust but requires computing centroid + distances
- (c) Use the node with the best average neighbor distance
Recommended: **(a) Use node 0** — simple, battle-tested. The NN-Descent graph is well-connected enough that any node works as an entry point. **Document your choice.**

**DECISION POINT (19.6)**: SoftDeletion persistence strategy. `IndexSerializer` currently saves core data (vectors, graph, idMap, entryPoint, metric). SoftDeletion and IndexConfiguration also need persisting. Options:
- (a) Save a companion `.meta.json` file alongside the `.mann` binary
- (b) Extend `IndexSerializer` to accept additional `Codable` payloads appended after the entry point
- (c) Wrap everything in a directory: `index.mann` + `metadata.json`
Recommended: **(a) Companion file** — saves a `{filename}.meta.json` next to the binary file. Contains JSON with `softDeletion`, `configuration`, and any future metadata. Keeps IndexSerializer unchanged. **Document your choice.**

**DECISION POINT (19.7)**: `count` semantics. Should `count` return:
- (a) Total vectors assigned (including deleted) — `idMap.count`
- (b) Active vectors (excluding deleted) — `idMap.count - softDeletion.deletedCount`
Recommended: **(b) Active count** — matches user expectations. Provide separate `totalCount` if needed. **Document your choice.**

**Commit**: `feat: implement ANNSIndex actor as public API facade`

---

### Task 20: Full Integration Test & Benchmark Runner

**Purpose**: Validate the full system end-to-end at realistic scale. Create a benchmark runner for performance measurement.

**Files to create:**
- `Tests/MetalANNSTests/IntegrationTests.swift`
- `Sources/MetalANNSBenchmarks/BenchmarkRunner.swift`
- Modify: `Sources/MetalANNSBenchmarks/main.swift`

**Integration test — 2 tests:**

1. `fullLifecycleIntegration`:
   - Create `ANNSIndex(configuration: IndexConfiguration(degree: 16, metric: .cosine, maxIterations: 15))`
   - Generate 500 random 64-dim vectors with IDs "v_0"..."v_499"
   - `build(vectors:ids:)`
   - Search for 20 random queries, k=10, assert all results have non-empty IDs and count == 10
   - Insert 50 new vectors ("new_0"..."new_49")
   - Search for 5 of the new vectors — verify they appear in own top-5
   - Delete 10 vectors (pick specific IDs)
   - Search and verify deleted IDs never appear
   - Save to temp directory
   - Load from saved file
   - Search on loaded index — verify results are consistent
   - Assert: `count` reflects inserts and deletes
   - Clean up temp files

2. `recallAtTenOverNinetyPercent`:
   - Create `ANNSIndex(configuration: IndexConfiguration(degree: 16, metric: .cosine, efSearch: 64, maxIterations: 15))`
   - Generate 500 random 32-dim vectors
   - Build index
   - 50 random queries, k=10
   - Compute brute-force ground truth for each query (sort all vectors by cosine distance)
   - Compute recall@10 = avg(|intersection(approx_topk, exact_topk)| / k)
   - Assert: recall@10 > 0.90
   - Guard with `guard MTLCreateSystemDefaultDevice() != nil else { return }` (GPU needed for reasonable performance at this scale, but CPU fallback should also work)

**BenchmarkRunner** (in `MetalANNSBenchmarks` target):

```swift
struct BenchmarkRunner {
    struct Config {
        var vectorCount: Int = 1000
        var dim: Int = 128
        var degree: Int = 32
        var queryCount: Int = 100
        var k: Int = 10
        var efSearch: Int = 64
        var metric: Metric = .cosine
    }

    struct Results {
        var buildTimeMs: Double
        var queryLatencyP50Ms: Double
        var queryLatencyP95Ms: Double
        var queryLatencyP99Ms: Double
        var recallAt1: Double
        var recallAt10: Double
        var recallAt100: Double
    }

    static func run(config: Config) async throws -> Results
}
```

**BenchmarkRunner.run implementation:**
1. Generate `config.vectorCount` random vectors of `config.dim` dimensions
2. Create `ANNSIndex`, time the `build` call → `buildTimeMs`
3. Generate `config.queryCount` random queries
4. For each query:
   - Time `search(query:k:)` call → collect latencies
   - Compute brute-force top-k → compare for recall
5. Compute percentiles from latency array
6. Compute recall@1, @10, @100 (adjust k per metric)
7. Return `Results`

**main.swift update:**
```swift
import MetalANNS
import MetalANNSCore

@main
struct BenchmarkApp {
    static func main() async throws {
        print("MetalANNS Benchmark Suite")
        print("========================")

        let config = BenchmarkRunner.Config()
        let results = try await BenchmarkRunner.run(config: config)

        print("Build time:      \(String(format: "%.1f", results.buildTimeMs)) ms")
        print("Query p50:       \(String(format: "%.2f", results.queryLatencyP50Ms)) ms")
        print("Query p95:       \(String(format: "%.2f", results.queryLatencyP95Ms)) ms")
        print("Query p99:       \(String(format: "%.2f", results.queryLatencyP99Ms)) ms")
        print("Recall@1:        \(String(format: "%.3f", results.recallAt1))")
        print("Recall@10:       \(String(format: "%.3f", results.recallAt10))")
        print("Recall@100:      \(String(format: "%.3f", results.recallAt100))")
    }
}
```

Note: If using `@main` attribute, remove the existing `print("MetalANNS Benchmarks")` from main.swift. Or just put the benchmark logic directly in the top-level `main.swift` without `@main`. **Pick whichever compiles cleanly.**

**DECISION POINT (20.3)**: Recall threshold. The original plan says recall@10 > 0.92 on 1000 vectors. At 500 vectors with degree 16, recall > 0.90 is more realistic. **Use 0.90 for the test assertion.** Document if you adjust.

**Commit**: `feat: add integration tests and benchmark runner`

---

### Task 21: README & Release Documentation

**Purpose**: Write user-facing documentation. No new Swift code or tests.

**Files to create:**
- `README.md`
- `BENCHMARKS.md`

**README.md should include:**

1. **Title and badge area**: `# MetalANNS` with one-line description
2. **Features**: Bullet list — GPU-accelerated, CAGRA-style NN-Descent, dual Metal/CPU backend, incremental insert, soft deletion, persistence, Swift 6 strict concurrency
3. **Requirements**: iOS 17+, macOS 14+, visionOS 1.0+, Apple Silicon (for GPU), zero dependencies
4. **Quick Start** code example:
   ```swift
   import MetalANNS

   let index = ANNSIndex()
   try await index.build(vectors: myVectors, ids: myIDs)
   let results = try await index.search(query: queryVector, k: 10)
   for result in results {
       print("\(result.id): \(result.score)")
   }
   ```
5. **Configuration**: Show `IndexConfiguration` parameters with defaults
6. **Incremental Operations**: Insert and delete examples
7. **Persistence**: Save and load examples
8. **Architecture**: Brief description of the two-layer design (MetalANNSCore internal, MetalANNS public API)
9. **Benchmarks**: Link to BENCHMARKS.md
10. **License**: MIT (or leave placeholder)

**BENCHMARKS.md should include:**
- Table of benchmark results from BenchmarkRunner
- Hardware info (Apple Silicon, macOS version)
- Configuration used (vector count, dim, degree, metric)
- Latency numbers (p50, p95, p99)
- Recall numbers (recall@1, @10, @100)
- Comparison note: "Run `swift run MetalANNSBenchmarks` to reproduce on your hardware"

**NOTE**: You must actually run the benchmark to get real numbers for BENCHMARKS.md. Run it via:
```bash
xcodebuild -scheme MetalANNSBenchmarks -destination 'platform=macOS' build 2>&1 | tail -5
```
Then execute the built binary to capture output. If the benchmark takes too long or fails, use reasonable placeholder values and note that they're estimated.

**Final verification before commit:**
1. Run full test suite: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'`
2. Run `git log --oneline` and verify 24 commits (after this commit)
3. Verify no `import XCTest` anywhere
4. Verify all files compile

**Commit**: `docs: add README and benchmark documentation`

---

## Decision Points Summary

| # | Decision | Recommended Approach |
|---|----------|---------------------|
| 19.5 | Entry point for GPU builds | Use node 0 — simple, well-connected graph makes any node viable |
| 19.6 | SoftDeletion persistence strategy | Companion `.meta.json` file alongside `.mann` binary |
| 19.7 | `count` semantics | Active count (excluding deleted) |
| 20.3 | Recall threshold for integration test | recall@10 > 0.90 on 500 vectors |

---

## Common Failure Modes (Read Before Starting)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ANNSIndex` can't be initialized as static func return | Actors need `async` for cross-isolation calls | Make `load(from:)` return the actor via async factory pattern |
| SearchResult has empty `id` field | Forgot to map internalID to externalID via IDMap | Apply `idMap.externalID(for:)` in search method before returning |
| Build succeeds but search returns garbage | GraphBuffer not populated correctly from CPU graph | Verify neighbor padding to degree, verify `graph.setCount()` called |
| GPU not available on CI | No Metal device in CI environment | CPU fallback path must work; guard GPU-specific tests |
| `batchSearch` crashes with concurrent access | Actor isolation should prevent this, but check | All access goes through actor methods — no shared mutable state |
| Benchmark executable doesn't build | Missing dependency on MetalANNS target | Check Package.swift — `MetalANNSBenchmarks` depends on `["MetalANNS", "MetalANNSCore"]` |
| Save works but load can't reconstruct actor | Actor init is not directly callable with all params | Use private init or configuration injection pattern for `load` |
| `delete` then `save` then `load` loses deletions | SoftDeletion not persisted | Ensure companion `.meta.json` includes SoftDeletion |
| Scheme not found | Xcode scheme naming | Use `MetalANNS-Package` for `xcodebuild test` |
| Recall too low in integration test | Degree/efSearch too low for vector count | Use degree=16, efSearch=64 minimum for 500 vectors |
| `IndexConfiguration` not `Codable` | Needs conformance for metadata persistence | Add `: Codable` to IndexConfiguration if not already present |

---

## Reference Files

| File | Purpose |
|------|---------|
| `docs/plans/2026-02-25-metalanns-implementation.md` (lines 2567–2694) | High-level guidance for Tasks 19–21 |
| `Sources/MetalANNS/ANNSIndex.swift` | **Your primary implementation file** |
| `Sources/MetalANNS/IndexConfiguration.swift` | Config struct — may need `Codable` |
| `Sources/MetalANNS/SearchResult.swift` | typealias to MetalANNSCore.SearchResult |
| `Sources/MetalANNSCore/IndexSerializer.swift` | Persistence — you call this from `save`/`load` |
| `Sources/MetalANNSCore/IncrementalBuilder.swift` | Insert — you call this from `insert` |
| `Sources/MetalANNSCore/SoftDeletion.swift` | Deletion — owned as actor state |
| `Sources/MetalANNSCore/SearchGPU.swift` | GPU search — you call this from `search` |
| `Sources/MetalANNSCore/BeamSearchCPU.swift` | CPU search fallback — you call this from `search` |
| `Sources/MetalANNSCore/NNDescentGPU.swift` | GPU construction — you call this from `build` |
| `Sources/MetalANNSCore/NNDescentCPU.swift` | CPU construction fallback — you call this from `build` |
| `Sources/MetalANNSCore/MetalDevice.swift` | MetalContext — attempt init, nil on failure |
| `Sources/MetalANNSBenchmarks/main.swift` | Benchmark entry point — you modify this |
| `Package.swift` | Package structure — verify target dependencies |
| `tasks/phase6-todo.md` | **Your checklist** |
| `tasks/lessons.md` | Record any lessons learned |

---

## Scope Boundary (What NOT To Do)

- Do NOT implement HNSW or any alternative graph algorithm
- Do NOT add batch insert (single-vector insert only through public API)
- Do NOT add hard deletion (removing nodes from graph structure)
- Do NOT add re-indexing, compaction, or graph optimization passes
- Do NOT add async/concurrent batch search (sequential loop is fine)
- Do NOT modify Phase 1–5 source files unless strictly necessary (document any changes)
- Do NOT use XCTest — Swift Testing exclusively
- Do NOT use `swift build` or `swift test` — `xcodebuild` only
- Do NOT tag `v1.0.0` — the orchestrator will handle release tagging after review
- Do NOT modify the Orchestrator Review Checklist in phase6-todo.md
