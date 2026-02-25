# Phase 7 Execution Prompt: CPU Quick Wins

---

## System Context

You are implementing **Phase 7 (Tasks 22–23)** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1–6 are complete. The codebase has dual compute backends, GPU-resident data structures, CPU and GPU NN-Descent construction, CPU and GPU beam search, index serialization, incremental insert, soft deletion, and the `ANNSIndex` public actor API. Git log shows 25 commits. Full test suite has 51 tests, zero failures.

You are building **CPU-side performance improvements** — replacing scalar distance loops with Accelerate vDSP calls and converting sequential batch search into concurrent TaskGroup-based execution.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/phase7-todo.md`

**Your communication contract:**
1. **`tasks/phase7-todo.md` is your shared state.** Check off `[x]` items as you complete them.
2. **Write notes under every task** — especially for decision points and any issues you hit.
3. **Update `Last Updated`** at the top of phase7-todo.md after each task completes.
4. **When done, fill in the "Phase 7 Complete — Signal" section** at the bottom.
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
8. **Do NOT modify Phase 1–6 source files** unless strictly necessary for wiring in the new code. Acceptable reasons: replacing inline distance calls with `SIMDDistance` calls in `BeamSearchCPU.swift`, `IncrementalBuilder.swift`, and `SearchGPU.swift`; modifying `batchSearch` in `ANNSIndex.swift`. Document every change in your notes.

---

## What Already Exists (Phases 1–6 Output)

### Package Structure

```
Sources/
  MetalANNS/                      ← Public API target (depends on MetalANNSCore)
    ANNSIndex.swift               ← Public actor with build/search/insert/delete/save/load
    IndexConfiguration.swift      ← Public config struct (Codable)
    SearchResult.swift            ← typealias to MetalANNSCore.SearchResult
    Errors.swift                  ← typealias to MetalANNSCore.ANNSError
    Metric.swift                  ← typealias to MetalANNSCore.Metric
  MetalANNSCore/                  ← Internal implementation target
    Shaders/
      Distance.metal              ← GPU distance kernels (cosine, L2, inner product)
      NNDescent.metal             ← GPU construction kernels
      Sort.metal                  ← Bitonic sort kernel
    BeamSearchCPU.swift           ← CPU beam search (scalar distance loops at lines 101-130)
    SearchGPU.swift               ← Hybrid CPU/GPU search (scalar distance at lines 185-214)
    NNDescentCPU.swift            ← CPU graph construction
    NNDescentGPU.swift            ← GPU graph construction
    IncrementalBuilder.swift      ← Incremental insert (scalar distance loops at lines 199-228)
    IndexSerializer.swift         ← Binary serialization (.mann format)
    SoftDeletion.swift            ← Soft deletion with filtered results
    VectorBuffer.swift            ← GPU-resident flat Float32 buffer
    GraphBuffer.swift             ← GPU-resident adjacency + distance arrays
    IDMap.swift                   ← Bidirectional String ↔ UInt32 mapping
    MetalDevice.swift             ← MetalContext with PipelineCache
    Errors.swift                  ← ANNSError enum
    Metric.swift                  ← Metric enum (cosine, l2, innerProduct)
    SearchResult.swift            ← SearchResult struct
    MetadataBuffer.swift          ← MetadataBuffer (unused by Tasks 22-23)
    IndexConfiguration.swift      ← IndexConfiguration struct (Codable)
    ComputeBackend.swift          ← ComputeBackend protocol
  MetalANNSBenchmarks/
    BenchmarkRunner.swift         ← Benchmark runner using ANNSIndex
    main.swift                    ← Entry point
Tests/
  MetalANNSTests/                 ← 51 tests across 19 suites
```

### Scalar Distance Functions to Replace

There are **three files** with identical scalar distance loops. Task 22 creates `SIMDDistance` and wires it into all three:

**1. `BeamSearchCPU.swift` — lines 101-130**
```swift
private static func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
    switch metric {
    case .cosine:
        var dot: Float = 0; var normQ: Float = 0; var normV: Float = 0
        for d in 0..<query.count { ... }
    case .l2:
        var sum: Float = 0
        for d in 0..<query.count { ... }
    case .innerProduct:
        var dot: Float = 0
        for d in 0..<query.count { ... }
    }
}
```
Called at line 47 (entry distance) and line 69 (candidate distance).

**2. `IncrementalBuilder.swift` — lines 199-228**
```swift
private static func distance(from lhs: [Float], to rhs: [Float], metric: Metric) -> Float {
    // Same scalar loop pattern
}
```
Called at lines 70, 98, 136, and 158.

**3. `SearchGPU.swift` — lines 185-214**
```swift
private static func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
    // Same scalar loop pattern — used only for entry point distance
}
```
Called at line 40 (entry point distance only — neighbors use GPU batch).

### Current `batchSearch` in `ANNSIndex.swift` — lines 207-215

```swift
public func batchSearch(queries: [[Float]], k: Int) async throws -> [[SearchResult]] {
    var allResults: [[SearchResult]] = []
    allResults.reserveCapacity(queries.count)
    for query in queries {
        let results = try await search(query: query, k: k)
        allResults.append(results)
    }
    return allResults
}
```

Task 23 replaces this with a `TaskGroup`-based concurrent implementation.

### Key API Signatures You'll Need

```swift
// Metric enum
public enum Metric: String, Sendable, Codable { case cosine, l2, innerProduct }

// VectorBuffer — vector(at:) returns [Float]
public func vector(at index: Int) -> [Float]

// SearchResult
public struct SearchResult: Sendable {
    public let id: String
    public let score: Float
    public let internalID: UInt32
}

// ANNSIndex actor (relevant methods)
public func search(query: [Float], k: Int) async throws -> [SearchResult]
public func batchSearch(queries: [[Float]], k: Int) async throws -> [[SearchResult]]

// ANNSIndex.init
public init(configuration: IndexConfiguration = .default)

// IndexConfiguration
public struct IndexConfiguration: Sendable, Codable {
    public var degree: Int           // default: 32
    public var metric: Metric        // default: .cosine
    public var efConstruction: Int   // default: 100
    public var efSearch: Int         // default: 64
    public var maxIterations: Int    // default: 20
    public var useFloat16: Bool      // default: false (unused)
    public var convergenceThreshold: Float // default: 0.001
}
```

---

## Task 22: SIMD CPU Distances via Accelerate

### Goal

Replace all scalar `for d in 0..<dim` distance loops with Accelerate `vDSP` calls for 4-8x speedup on CPU paths. This is a pure performance improvement — results must be numerically identical (within floating-point tolerance).

### What to Create

**`Sources/MetalANNSCore/SIMDDistance.swift`** — stateless enum with static methods:

```swift
import Accelerate

public enum SIMDDistance {
    /// Cosine distance: 1 - (a·b)/(|a||b|)
    public static func cosine(_ a: [Float], _ b: [Float]) -> Float

    /// Cosine distance using raw pointers (avoids array copy overhead)
    public static func cosine(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, dim: Int) -> Float

    /// Squared L2 distance: Σ(a[i] - b[i])²
    public static func l2(_ a: [Float], _ b: [Float]) -> Float

    /// Squared L2 distance using raw pointers
    public static func l2(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, dim: Int) -> Float

    /// Negated inner product: -(a·b)
    public static func innerProduct(_ a: [Float], _ b: [Float]) -> Float

    /// Negated inner product using raw pointers
    public static func innerProduct(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, dim: Int) -> Float

    /// Dispatch to correct metric
    public static func distance(_ a: [Float], _ b: [Float], metric: Metric) -> Float

    /// Dispatch to correct metric (pointer variant)
    public static func distance(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, dim: Int, metric: Metric) -> Float
}
```

**Implementation details:**
- `cosine`: Three `vDSP_dotpr` calls (a·b, a·a, b·b), then `1.0 - dot / (sqrt(normA) * sqrt(normB))`. Guard against zero-magnitude: `denom < 1e-10 ? 1.0 : ...`
- `l2`: Single `vDSP_distancesq` call.
- `innerProduct`: Single `vDSP_dotpr` call, negate result.
- Array variants use `withUnsafeBufferPointer` to get pointers.
- All functions have `precondition(a.count == b.count)` for array variants.

### What to Modify

After `SIMDDistance` is implemented and tested, **replace** the private `distance(...)` functions in:

1. **`BeamSearchCPU.swift`**: Delete lines 101-130 (the `private static func distance(query:vector:metric:)` method). Replace call sites at lines 47 and 69 with `SIMDDistance.distance(query, vector, metric: metric)`.

2. **`IncrementalBuilder.swift`**: Delete lines 199-228 (the `private static func distance(from:to:metric:)` method). Replace call sites at lines 70, 98, 136, and 158 with `SIMDDistance.distance(lhs, rhs, metric: metric)`.

3. **`SearchGPU.swift`**: Delete lines 185-214 (the `private static func distance(query:vector:metric:)` method). Replace call site at line 40 with `SIMDDistance.distance(query, vectors.vector(at: entryPoint), metric: metric)`.

### Tests (4 tests)

**`Tests/MetalANNSTests/SIMDDistanceTests.swift`**:

1. **`cosineMatchesScalar`**: Generate two random 128-dim vectors. Compute cosine distance with SIMDDistance and with an inline scalar reference. Assert `abs(simd - scalar) < 1e-5`.

2. **`l2MatchesScalar`**: Same pattern for L2 squared distance.

3. **`innerProductMatchesScalar`**: Same pattern for negated inner product.

4. **`simdFasterThanScalar`**: Benchmark: 10,000 distance computations at dim=256. Measure `ContinuousClock` durations. Assert `simdTime < scalarTime`. (This is a soft assertion — it validates the optimization has measurable impact. If it fails on a particular machine, the tolerance can be loosened.)

Each test must include its own scalar reference implementation for comparison — do NOT import the scalar versions from other files (those will be deleted).

### Decision Points

- **22.1**: The speed test (`simdFasterThanScalar`) uses a soft assertion (`simdTime < scalarTime`). If this proves flaky on CI, it's acceptable to weaken to `#expect(simdTime < scalarTime * 2)` or remove the strict timing assertion and just verify the SIMD path runs without errors. **Document your choice in notes.**

### Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `SIMDDistance` not found in tests | Missing `public` on enum or methods | Add `public` access modifiers |
| Numerical mismatch > 1e-5 | vDSP accumulates differently than scalar | Increase tolerance to 1e-4 |
| Regression in `BeamSearchCPU` tests | Wiring error — wrong argument order | Check parameter names at call sites |
| Regression in `IncrementalBuilder` tests | Signature mismatch after replacing distance | Ensure `SIMDDistance.distance(lhs, rhs, metric:)` matches old call pattern |
| Import error for `Accelerate` | Missing framework | Accelerate is available on all Apple platforms — just `import Accelerate` |

---

## Task 23: Concurrent Batch Search via TaskGroup

### Goal

Replace the sequential `for query in queries` loop in `ANNSIndex.batchSearch` with a `TaskGroup`-based concurrent implementation. This enables multiple queries to execute in parallel, improving throughput on multi-core machines.

### What to Modify

**`Sources/MetalANNS/ANNSIndex.swift`** — Replace the `batchSearch` method (lines 207-215):

```swift
public func batchSearch(queries: [[Float]], k: Int) async throws -> [[SearchResult]] {
    guard isBuilt else { throw ANNSError.indexEmpty }
    guard !queries.isEmpty else { return [] }

    // Determine concurrency limit
    // GPU backend: limit to 4 to avoid saturating the Metal command queue
    // CPU fallback: use available processor count
    let maxConcurrency = context != nil ? 4 : ProcessInfo.processInfo.activeProcessorCount

    return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
        var results = Array<[SearchResult]?>(repeating: nil, count: queries.count)
        var nextIndex = 0

        // Seed initial batch up to concurrency limit
        for _ in 0..<min(maxConcurrency, queries.count) {
            let idx = nextIndex
            let query = queries[idx]
            nextIndex += 1
            group.addTask { [self] in
                let result = try await self.search(query: query, k: k)
                return (idx, result)
            }
        }

        // Process completions and feed more work
        for try await (idx, result) in group {
            results[idx] = result
            if nextIndex < queries.count {
                let idx = nextIndex
                let query = queries[idx]
                nextIndex += 1
                group.addTask { [self] in
                    let result = try await self.search(query: query, k: k)
                    return (idx, result)
                }
            }
        }

        return results.map { $0! }
    }
}
```

### Important: Actor Reentrancy

Since `ANNSIndex` is an actor, calling `self.search()` from `TaskGroup` child tasks requires re-entering the actor. This is **safe** — Swift actors support reentrancy — but it means GPU search calls are still serialized through the actor. True parallelism happens on the CPU fallback path. The GPU path benefits from pipelining (next query's setup overlaps with previous query's GPU execution).

### Tests (2 tests)

**`Tests/MetalANNSTests/ConcurrentSearchTests.swift`**:

1. **`batchSearchMatchesSequential`**:
   - Create `ANNSIndex` with degree=8, metric=.cosine
   - Build with 100 random 16-dim vectors, IDs `"v_0"..."v_99"`
   - Generate 10 random queries
   - Run sequential: loop over queries calling `search(query:k:)` individually
   - Run concurrent: call `batchSearch(queries:k:)`
   - Assert: same count, same result IDs per query (compare as `Set` since order may vary for tied distances)

2. **`batchSearchHandlesLargeQueryCount`**:
   - Build with 200 random 32-dim vectors
   - Generate 50 random queries
   - Call `batchSearch(queries:k:10)`
   - Assert: 50 result arrays, each with 10 results
   - Assert: no crashes, no empty results

### Decision Points

- **23.1**: The `maxConcurrency` value for GPU backend. The plan uses `4` to avoid Metal command queue saturation. If you observe issues (deadlocks, GPU errors), you can lower to `2` or even `1`. For CPU fallback, `ProcessInfo.processInfo.activeProcessorCount` is used. **Document your choice in notes.**

- **23.2**: Actor reentrancy behavior. If the TaskGroup approach causes compilation warnings about `Sendable` or actor isolation, you may need to add `@Sendable` to the closure or adjust capture semantics. **Document any changes in notes.**

### Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Compilation error: `self captured` | Actor isolation in TaskGroup closure | Use `[self]` capture list; ensure closure is `@Sendable` |
| Results in wrong order | Index not preserved across async boundary | Verify `idx` is captured by value (let binding before closure) |
| Deadlock / hang | Actor reentrancy issue | Check that `search` doesn't hold non-reentrant resources |
| `Sendable` warning on `[SearchResult]` | `SearchResult` not `Sendable` | It already is — check return type annotation |
| Test flakiness on `batchSearchMatchesSequential` | Floating-point tie ordering | Compare result sets, not arrays |

---

## Scope Boundary

**In scope for Phase 7:**
- `SIMDDistance.swift` with Accelerate vDSP (Task 22)
- Wiring `SIMDDistance` into `BeamSearchCPU`, `IncrementalBuilder`, `SearchGPU` (Task 22)
- Concurrent `batchSearch` via TaskGroup (Task 23)
- New test files: `SIMDDistanceTests.swift`, `ConcurrentSearchTests.swift`

**NOT in scope:**
- New Metal shader files (that's Phase 8)
- Float16 (that's Phase 9)
- Any changes to `NNDescentCPU.swift` or `NNDescentGPU.swift` (they use different distance computation patterns)
- Changes to `IndexSerializer`, `SoftDeletion`, `IDMap`, or `IndexConfiguration`
- Changes to `Package.swift`

---

## Reference Files

Read these before starting:
- `tasks/phase7-todo.md` — your shared task checklist
- `Sources/MetalANNSCore/BeamSearchCPU.swift` — scalar distance at lines 101-130, called at lines 47 and 69
- `Sources/MetalANNSCore/IncrementalBuilder.swift` — scalar distance at lines 199-228, called at lines 70, 98, 136, and 158
- `Sources/MetalANNSCore/SearchGPU.swift` — scalar distance at lines 185-214, called at line 40
- `Sources/MetalANNS/ANNSIndex.swift` — `batchSearch` at lines 207-215
- `docs/plans/2026-02-25-metalanns-v2-performance-features.md` — overall v2 plan (Tasks 22-23 section)

---

## Verification Commands

```bash
# Run only SIMDDistance tests (Task 22 RED/GREEN)
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/SIMDDistanceTests 2>&1 | grep -E '(PASS|FAIL|error:)'

# Run only ConcurrentSearch tests (Task 23 RED/GREEN)
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/ConcurrentSearchTests 2>&1 | grep -E '(PASS|FAIL|error:)'

# Full regression suite
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'

# Verify test count (expected: 57 = 51 prior + 4 from Task 22 + 2 from Task 23)
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | tail -3

# Git log check
git log --oneline | head -5
```
