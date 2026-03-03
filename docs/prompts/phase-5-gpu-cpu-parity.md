# Phase 5: GPU-vs-CPU Search Parity Tests

## Role
You are a Swift 6.0 systems engineer writing correctness tests that prove the `FullGPUSearch` Metal kernel agrees with `BeamSearchCPU` on the same graph. You have deep familiarity with the MetalANNS architecture and the Swift Testing framework.

---

## Context

### What Already Exists (Do Not Duplicate)

`Tests/MetalANNSTests/FullGPUSearchTests.swift` has 4 tests that already cover:
- GPU returns k sorted results at 200 nodes
- GPU recall vs **brute-force** (exact top-k) at 200 nodes — recall >= 0.70
- GPU recall vs hybrid (`SearchGPU`) at 500 nodes — recall > 0.80
- GPU correctness above 4096 nodes (5000 nodes)

**These tests use brute-force or hybrid as ground truth.** They do NOT compare GPU to `BeamSearchCPU` on the same graph. That gap is what Phase 5 fills.

### Why This Test Matters
Phase 2 replaced the visited-set implementation in the `beam_search` Metal kernel (from per-search zeroing to generation counters). Without a direct GPU-vs-CPU comparison on the **same graph**, a regression in traversal correctness could go undetected — the brute-force recall tests only catch gross failures, not subtle algorithmic drift.

### What Needs to Be Created

**New file:** `Tests/MetalANNSTests/GPUCPUParityTests.swift`
**Status:** Does NOT exist yet.

This file tests one thing: given the same graph and the same query, do `FullGPUSearch` and `BeamSearchCPU` return result sets that overlap by at least 60%?

### Exact API Signatures (Verified Against Source)

```swift
// NNDescentCPU (Sources/MetalANNSCore/NNDescentCPU.swift:7)
NNDescentCPU.build(
    vectors: [[Float]],
    degree: Int,
    metric: Metric,
    maxIterations: Int = 20,         // use 5 for large scales
    convergenceThreshold: Float = 0.001
) async throws -> (graph: [[(UInt32, Float)]], entryPoint: UInt32)

// BeamSearchCPU [[Float]] overload (Sources/MetalANNSCore/BeamSearchCPU.swift:37)
BeamSearchCPU.search(
    query: [Float],
    vectors: [[Float]],              // pass original [[Float]] directly — no extraction needed
    graph: [[(UInt32, Float)]],      // pass the cpuGraph from NNDescentCPU.build directly
    entryPoint: Int,
    k: Int,
    ef: Int,
    metric: Metric
) async throws -> [SearchResult]

// FullGPUSearch (Sources/MetalANNSCore/FullGPUSearch.swift:7)
FullGPUSearch.search(
    context: MetalContext,
    query: [Float],
    vectors: any VectorStorage,      // pass VectorBuffer
    graph: GraphBuffer,
    entryPoint: Int,
    k: Int,
    ef: Int,
    metric: Metric
) async throws -> [SearchResult]

// VectorBuffer (Sources/MetalANNSCore/VectorBuffer.swift)
VectorBuffer(capacity: Int, dim: Int, device: MTLDevice)
try vectorBuffer.batchInsert(vectors: [[Float]], startingAt: Int)
vectorBuffer.setCount(_ count: Int)

// GraphBuffer (Sources/MetalANNSCore/GraphBuffer.swift)
GraphBuffer(capacity: Int, degree: Int, device: MTLDevice)
try graphBuffer.setNeighbors(of: Int, ids: [UInt32], distances: [Float])
graphBuffer.setCount(_ count: Int)
```

### Key Design Decision: Reuse cpuGraph Directly
`NNDescentCPU.build` returns `(graph: [[(UInt32, Float)]], entryPoint: UInt32)`.
That same `cpuGraph` value feeds both the `GraphBuffer` (for GPU search) and `BeamSearchCPU.search(graph:)` (for CPU search).
**Do NOT extract vectors back from VectorBuffer** — you already have the original `[[Float]]` array.

### SeededGenerator Scope Rule
All 8 existing test files define `SeededGenerator` as `private struct` (file-scoped).
You MUST also declare it `private` in GPUCPUParityTests.swift. Omitting `private` causes a "redeclaration" compile error since all test files compile into the same `MetalANNSTests` module.

---

## Task: Write GPUCPUParityTests.swift

### Step 1: Create the test file

```swift
// Tests/MetalANNSTests/GPUCPUParityTests.swift
import Testing
import Metal
@testable import MetalANNSCore

private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    init(state: UInt64) { self.state = state == 0 ? 1 : state }
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

@Suite("GPU-CPU Search Parity")
struct GPUCPUParityTests {

    // Shared helper: builds VectorBuffer + GraphBuffer from [[Float]] via NNDescentCPU.
    // Returns the raw cpuGraph for direct use with BeamSearchCPU (no extraction step needed).
    private func buildGraph(
        vectors: [[Float]],
        degree: Int,
        maxIterations: Int,
        context: MetalContext
    ) async throws -> (vectorBuffer: VectorBuffer, graphBuffer: GraphBuffer,
                       cpuGraph: [[(UInt32, Float)]], entryPoint: Int) {
        let nodeCount = vectors.count
        let dim = vectors[0].count
        let device = context.device

        let (cpuGraph, cpuEntry) = try await NNDescentCPU.build(
            vectors: vectors,
            degree: degree,
            metric: .cosine,
            maxIterations: maxIterations
        )

        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
        try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
        vectorBuffer.setCount(nodeCount)

        let graphBuffer = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
        for node in 0..<nodeCount {
            let neighbors = cpuGraph[node]
            let ids = neighbors.map(\.0) + Array(
                repeating: UInt32.max, count: max(0, degree - neighbors.count)
            )
            let dists = neighbors.map(\.1) + Array(
                repeating: Float.greatestFiniteMagnitude, count: max(0, degree - neighbors.count)
            )
            try graphBuffer.setNeighbors(
                of: node,
                ids: Array(ids.prefix(degree)),
                distances: Array(dists.prefix(degree))
            )
        }
        graphBuffer.setCount(nodeCount)

        return (vectorBuffer, graphBuffer, cpuGraph, Int(cpuEntry))
    }

    // MARK: - Parameterized parity test across four scales

    @Test(arguments: [
        (nodeCount: 100,  dim: 32,  degree: 8,  k: 5,  ef: 32,  maxIter: 10),
        (nodeCount: 500,  dim: 64,  degree: 16, k: 10, ef: 64,  maxIter: 10),
        (nodeCount: 2000, dim: 128, degree: 32, k: 20, ef: 128, maxIter: 5),
        (nodeCount: 8000, dim: 384, degree: 32, k: 10, ef: 64,  maxIter: 3),
    ])
    func gpuAndCPUSearchAgreeOnSameGraph(
        nodeCount: Int, dim: Int, degree: Int, k: Int, ef: Int, maxIter: Int
    ) async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            print("Skipping: no Metal device available")
            return
        }
        let context = try MetalContext()

        var rng = SeededGenerator(state: UInt64(nodeCount) &* UInt64(dim) &+ 7)
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let (vectorBuffer, graphBuffer, cpuGraph, entryPoint) = try await buildGraph(
            vectors: vectors,
            degree: degree,
            maxIterations: maxIter,
            context: context
        )

        // Test 5 evenly-spaced query vectors from the index
        let stride = max(1, nodeCount / 5)
        var totalOverlap: Float = 0
        var queryCount = 0

        for qi in 0..<5 {
            let queryIndex = qi * stride
            guard queryIndex < nodeCount else { continue }
            let query = vectors[queryIndex]

            let gpuResults = try await FullGPUSearch.search(
                context: context,
                query: query,
                vectors: vectorBuffer,
                graph: graphBuffer,
                entryPoint: entryPoint,
                k: k,
                ef: ef,
                metric: .cosine
            )

            // Pass original [[Float]] and cpuGraph directly — no extraction from buffers needed
            let cpuResults = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: cpuGraph,
                entryPoint: entryPoint,
                k: k,
                ef: ef,
                metric: .cosine
            )

            guard !gpuResults.isEmpty, !cpuResults.isEmpty else {
                continue  // Skip queries where one backend returned nothing
            }

            let gpuIDs = Set(gpuResults.prefix(k).map(\.internalID))
            let cpuIDs = Set(cpuResults.prefix(k).map(\.internalID))
            let denominator = Float(min(k, min(gpuResults.count, cpuResults.count)))
            let overlap = Float(gpuIDs.intersection(cpuIDs).count) / denominator

            totalOverlap += overlap
            queryCount += 1
        }

        guard queryCount > 0 else {
            Issue.record("No queries produced results at n=\(nodeCount) dim=\(dim)")
            return
        }

        let avgOverlap = totalOverlap / Float(queryCount)
        #expect(
            avgOverlap >= 0.6,
            "GPU-vs-CPU avg overlap \(String(format: "%.2f", avgOverlap)) < 0.60 "
            + "at n=\(nodeCount) dim=\(dim) degree=\(degree) k=\(k) ef=\(ef)"
        )
    }

    // MARK: - Determinism test: same query twice gives same GPU results

    @Test
    func gpuSearchIsDeterministic() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let context = try MetalContext()

        var rng = SeededGenerator(state: 1234)
        let nodeCount = 300
        let dim = 64
        let degree = 16
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let (vectorBuffer, graphBuffer, _, entryPoint) = try await buildGraph(
            vectors: vectors,
            degree: degree,
            maxIterations: 5,
            context: context
        )

        let query = vectors[10]

        let run1 = try await FullGPUSearch.search(
            context: context, query: query, vectors: vectorBuffer,
            graph: graphBuffer, entryPoint: entryPoint, k: 10, ef: 32, metric: .cosine
        )
        let run2 = try await FullGPUSearch.search(
            context: context, query: query, vectors: vectorBuffer,
            graph: graphBuffer, entryPoint: entryPoint, k: 10, ef: 32, metric: .cosine
        )

        let ids1 = run1.map(\.internalID)
        let ids2 = run2.map(\.internalID)
        #expect(ids1 == ids2, "FullGPUSearch must return identical results for identical inputs")
    }
}
```

### Step 2: Run the test to verify it compiles and passes

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/GPUCPUParityTests 2>&1 | tail -40
```

**Expected: PASS.** These tests do not require new implementation — they validate existing behaviour. If any test fails, investigate the GPU kernel (do not lower the threshold to make it pass).

**If the 8000-node test times out:** The `maxIter: 3` for that scale should keep NNDescentCPU under 60 seconds. If it still times out, reduce to `maxIter: 1`. The graph just needs to be connected — quality doesn't matter for a parity test.

**If recall falls below 0.6:** This signals a real bug in the generation-counter visited set from Phase 2. File the failure with the query index and scale parameters before investigating.

### Step 3: Run the full suite to confirm no regressions

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30
```

Expected: All previously passing tests still green.

### Step 4: Commit

```bash
git add Tests/MetalANNSTests/GPUCPUParityTests.swift
git commit -m "test: add GPU-vs-CPU beam search parity tests at 4 scales including 8000-node 384-dim"
```

---

## Definition of Done

- [ ] `GPUCPUParityTests.swift` exists with 2 tests
- [ ] Parameterized test runs at all 4 scales: 100, 500, 2000, 8000 nodes
- [ ] avg GPU-CPU overlap >= 0.60 at every scale
- [ ] Determinism test passes (identical query → identical GPU results)
- [ ] Full suite green, no regressions
- [ ] `SeededGenerator` declared `private` (no compile-time redeclaration conflict)

---

## Anti-Patterns to Avoid

- **Do not extract vectors from VectorBuffer.** You already have `vectors: [[Float]]` — pass it to `BeamSearchCPU.search(vectors:)` directly.
- **Do not extract the graph from GraphBuffer.** Pass `cpuGraph` from `NNDescentCPU.build` directly to `BeamSearchCPU.search(graph:)`.
- **Do not use `Float.random(in:)` without a seeded generator.** Recall-threshold tests must be reproducible.
- **Do not declare `struct SeededGenerator` without `private`.** Other test files in the module already define it; `internal` scope causes a redeclaration compile error.
- **Do not duplicate what FullGPUSearchTests already covers.** This file is about GPU-vs-CPU algorithm agreement, not GPU-vs-brute-force recall.
- **Do not lower the 0.60 threshold to make a failing test pass.** A failure below 0.60 means a real kernel bug — investigate it.
- **Do not use `NNDescentGPU.build`.** It mutates the GraphBuffer in-place and doesn't return the raw graph edges needed to drive `BeamSearchCPU.search(graph:)`. Use `NNDescentCPU.build` so you get both outputs.
