# MetalANNS v2: Performance, Scalability & Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform MetalANNS from a functional ANNS library into a production-grade, high-performance vector search engine with full GPU search, Float16 support, filtered search, and million-scale capabilities.

**Architecture:** Six phases of incremental improvement, each building on the last. Phase 7 (CPU quick wins) and Phase 8 (GPU search kernel) provide the largest performance gains. Phase 9 (Float16) halves memory. Phases 10–12 add scalability and user-facing features. Every change is backward-compatible with the existing v1 binary format.

**Tech Stack:** Swift 6, Metal Shading Language, Accelerate framework (vDSP/BLAS), Swift Testing, xcodebuild

---

## Dependency Graph

```
Phase 7 (CPU Quick Wins)                      Phase 8 (GPU Search)
  Task 22: SIMD CPU distances ──────────┐       Task 24: Full GPU beam search kernel
  Task 23: Concurrent batch search      │       Task 25: CAGRA post-processing
                                        │
Phase 9 (Float16)                       │     Phase 10 (Scalability)
  Task 26: Float16 buffers + kernels ───┤       Task 28: Batch insert
  Task 27: Float16 construction/search  │       Task 29: Hard deletion + compaction
                                        │       Task 30: Memory-mapped I/O
                                        │
                                        │     Phase 11 (Advanced Search)
                                        ├───── Task 31: Filtered search
                                        │       Task 32: Range search
                                        │       Task 33: Runtime metric selection
                                        │
                                        │     Phase 12 (Large-Scale)
                                        └───── Task 34: Disk-backed index
                                                Task 35: Sharded indices
```

**Critical path:** Phase 7 → Phase 8 → Phase 9 (the performance trilogy)
**Independent work:** Phases 10, 11, 12 can proceed in parallel after Phase 8

---

## Current Baseline (Phase 6 Complete)

- **24 commits**, 51 tests, zero failures
- **Files**: 25 Swift source files, 3 Metal shaders, 4 targets
- **Performance bottlenecks**:
  - CPU distances: scalar loops (no SIMD)
  - GPU search: hybrid CPU/GPU with per-step buffer round-trips
  - Batch search: sequential loop
  - Float16: `useFloat16` config flag exists but is unused
  - Memory: entire index loaded into RAM via `Data(contentsOf:)`

---

## Phase 7: CPU Quick Wins

**Goal**: 4-8x CPU distance speedup + concurrent query throughput. Low risk, high reward.

### Task 22: SIMD CPU Distances via Accelerate

**Files:**
- Create: `Sources/MetalANNSCore/SIMDDistance.swift`
- Create: `Tests/MetalANNSTests/SIMDDistanceTests.swift`
- Modify: `Sources/MetalANNSCore/BeamSearchCPU.swift` (replace inline distance)
- Modify: `Sources/MetalANNSCore/IncrementalBuilder.swift` (replace inline distance)

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/SIMDDistanceTests.swift
import Testing
import MetalANNSCore

@Suite("SIMD Distance")
struct SIMDDistanceTests {
    @Test func cosineMatchesScalar() async throws {
        let dim = 128
        let a = (0..<dim).map { _ in Float.random(in: -1...1) }
        let b = (0..<dim).map { _ in Float.random(in: -1...1) }

        let simdResult = SIMDDistance.cosine(a, b)
        let scalarResult = scalarCosine(a, b)
        #expect(abs(simdResult - scalarResult) < 1e-5)
    }

    @Test func l2MatchesScalar() async throws {
        let dim = 128
        let a = (0..<dim).map { _ in Float.random(in: -1...1) }
        let b = (0..<dim).map { _ in Float.random(in: -1...1) }

        let simdResult = SIMDDistance.l2(a, b)
        let scalarResult = scalarL2(a, b)
        #expect(abs(simdResult - scalarResult) < 1e-5)
    }

    @Test func innerProductMatchesScalar() async throws {
        let dim = 128
        let a = (0..<dim).map { _ in Float.random(in: -1...1) }
        let b = (0..<dim).map { _ in Float.random(in: -1...1) }

        let simdResult = SIMDDistance.innerProduct(a, b)
        let scalarResult = scalarInnerProduct(a, b)
        #expect(abs(simdResult - scalarResult) < 1e-5)
    }

    @Test func simdFasterThanScalar() async throws {
        // Benchmark: 10000 distance computations, SIMD should be ≥2x faster
        let dim = 256
        let a = (0..<dim).map { _ in Float.random(in: -1...1) }
        let b = (0..<dim).map { _ in Float.random(in: -1...1) }

        let start = ContinuousClock.now
        for _ in 0..<10000 {
            _ = SIMDDistance.cosine(a, b)
        }
        let simdTime = ContinuousClock.now - start

        let start2 = ContinuousClock.now
        for _ in 0..<10000 {
            _ = scalarCosine(a, b)
        }
        let scalarTime = ContinuousClock.now - start2

        // SIMD should be at least 2x faster
        #expect(simdTime < scalarTime)
    }

    // Scalar reference implementations for comparison
    private func scalarCosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, normA: Float = 0, normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom < 1e-10 ? 1.0 : (1.0 - dot / denom)
    }

    private func scalarL2(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count { let d = a[i] - b[i]; sum += d * d }
        return sum
    }

    private func scalarInnerProduct(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }
        return -dot
    }
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/SIMDDistanceTests 2>&1 | grep -E '(PASS|FAIL|error:)'`
Expected: FAIL with "SIMDDistance not found"

**Step 3: Write implementation**

```swift
// Sources/MetalANNSCore/SIMDDistance.swift
import Accelerate

public enum SIMDDistance {
    /// Cosine distance: 1 - (a·b)/(|a||b|)
    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                vDSP_dotpr(aPtr.baseAddress!, 1, bPtr.baseAddress!, 1, &dot, vDSP_Length(a.count))
                vDSP_dotpr(aPtr.baseAddress!, 1, aPtr.baseAddress!, 1, &normA, vDSP_Length(a.count))
                vDSP_dotpr(bPtr.baseAddress!, 1, bPtr.baseAddress!, 1, &normB, vDSP_Length(b.count))
            }
        }

        let denom = sqrt(normA) * sqrt(normB)
        return denom < 1e-10 ? 1.0 : (1.0 - dot / denom)
    }

    /// Cosine distance using raw pointers (avoids array overhead)
    public static func cosine(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        dim: Int
    ) -> Float {
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(dim))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(dim))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(dim))
        let denom = sqrt(normA) * sqrt(normB)
        return denom < 1e-10 ? 1.0 : (1.0 - dot / denom)
    }

    /// Squared L2 distance
    public static func l2(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    /// Squared L2 distance using raw pointers
    public static func l2(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        dim: Int
    ) -> Float {
        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(dim))
        return result
    }

    /// Negated inner product distance
    public static func innerProduct(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                vDSP_dotpr(aPtr.baseAddress!, 1, bPtr.baseAddress!, 1, &dot, vDSP_Length(a.count))
            }
        }
        return -dot
    }

    /// Negated inner product using raw pointers
    public static func innerProduct(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        dim: Int
    ) -> Float {
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(dim))
        return -dot
    }

    /// Dispatch to correct metric
    public static func distance(
        _ a: [Float],
        _ b: [Float],
        metric: Metric
    ) -> Float {
        switch metric {
        case .cosine: cosine(a, b)
        case .l2: l2(a, b)
        case .innerProduct: innerProduct(a, b)
        }
    }

    /// Dispatch to correct metric (pointer variant)
    public static func distance(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        dim: Int,
        metric: Metric
    ) -> Float {
        switch metric {
        case .cosine: cosine(a, b, dim: dim)
        case .l2: l2(a, b, dim: dim)
        case .innerProduct: innerProduct(a, b, dim: dim)
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/SIMDDistanceTests 2>&1 | grep -E '(PASS|FAIL|error:)'`
Expected: PASS

**Step 5: Wire into BeamSearchCPU + IncrementalBuilder**

Replace the `private static func distance(...)` methods in both files:
- `BeamSearchCPU.swift:101-130` → call `SIMDDistance.distance(query, vector, metric: metric)`
- `IncrementalBuilder.swift:199-228` → call `SIMDDistance.distance(from: lhs, to: rhs, metric: metric)`

**Step 6: Regression test**

Run: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
Expected: All 51+ tests pass

**Step 7: Commit**

```bash
git add Sources/MetalANNSCore/SIMDDistance.swift Tests/MetalANNSTests/SIMDDistanceTests.swift Sources/MetalANNSCore/BeamSearchCPU.swift Sources/MetalANNSCore/IncrementalBuilder.swift
git commit -m "perf: replace scalar distance loops with Accelerate vDSP"
```

---

### Task 23: Concurrent Batch Search via TaskGroup

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (modify `batchSearch`)
- Create: `Tests/MetalANNSTests/ConcurrentSearchTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/ConcurrentSearchTests.swift
import Testing
import MetalANNS
import MetalANNSCore

@Suite("Concurrent Search")
struct ConcurrentSearchTests {
    @Test func batchSearchMatchesSequential() async throws {
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        let vectors = (0..<100).map { _ in (0..<16).map { _ in Float.random(in: -1...1) } }
        let ids = (0..<100).map { "v_\($0)" }
        try await index.build(vectors: vectors, ids: ids)

        let queries = (0..<10).map { _ in (0..<16).map { _ in Float.random(in: -1...1) } }

        // Sequential baseline
        var sequential: [[SearchResult]] = []
        for q in queries {
            sequential.append(try await index.search(query: q, k: 5))
        }

        // Concurrent batch
        let batch = try await index.batchSearch(queries: queries, k: 5)

        #expect(batch.count == sequential.count)
        for i in 0..<queries.count {
            #expect(batch[i].count == sequential[i].count)
            // Same results (order may differ due to ties)
            let batchIDs = Set(batch[i].map(\.internalID))
            let seqIDs = Set(sequential[i].map(\.internalID))
            #expect(batchIDs == seqIDs)
        }
    }

    @Test func batchSearchHandlesLargeQueryCount() async throws {
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        let vectors = (0..<200).map { _ in (0..<32).map { _ in Float.random(in: -1...1) } }
        let ids = (0..<200).map { "v_\($0)" }
        try await index.build(vectors: vectors, ids: ids)

        let queries = (0..<50).map { _ in (0..<32).map { _ in Float.random(in: -1...1) } }
        let results = try await index.batchSearch(queries: queries, k: 10)

        #expect(results.count == 50)
        for r in results {
            #expect(r.count == 10)
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/ConcurrentSearchTests 2>&1 | grep -E '(PASS|FAIL|error:)'`
Expected: Tests should compile (batchSearch already exists) but assertions may already pass since current sequential version is correct. If they pass, this task is about **performance improvement**, not correctness change.

**Step 3: Implement concurrent batch search**

In `ANNSIndex.swift`, replace the sequential `batchSearch` loop:

```swift
public func batchSearch(queries: [[Float]], k: Int) async throws -> [[SearchResult]] {
    guard isBuilt else { throw ANNSError.indexEmpty }

    // Determine concurrency limit based on backend
    // For GPU: limit to avoid saturating command queue
    // For CPU: use available cores
    let maxConcurrency = context != nil ? 4 : ProcessInfo.processInfo.activeProcessorCount

    return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
        var results = Array<[SearchResult]?>(repeating: nil, count: queries.count)
        var nextIndex = 0

        // Seed initial batch
        for _ in 0..<min(maxConcurrency, queries.count) {
            let idx = nextIndex
            let query = queries[idx]
            nextIndex += 1
            group.addTask { [self] in
                let result = try await self.search(query: query, k: k)
                return (idx, result)
            }
        }

        // Process as they complete, add more
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

**Important note on actor isolation:** Since `ANNSIndex` is an actor, calling `self.search()` from a `TaskGroup` child task requires re-entering the actor. This is safe (actor reentrance is supported) but serializes GPU access. For truly parallel GPU search, the GPU kernels would need to be dispatched independently — that optimization belongs in Phase 8 (full GPU search kernel).

**Step 4: Run tests**

Run: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
Expected: All tests pass

**Step 5: Commit**

```bash
git add Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/ConcurrentSearchTests.swift
git commit -m "perf: concurrent batch search via TaskGroup"
```

---

## Phase 8: Full GPU Search

**Goal**: Eliminate CPU↔GPU round-trips in search. Single kernel dispatch per query. 5-10x latency reduction.

### Task 24: Full GPU Beam Search Kernel

This is the most complex and highest-value task. The current `SearchGPU` dispatches a separate GPU command per beam step (one per neighbor expansion). The new kernel performs the entire beam traversal in a single kernel dispatch using shared memory.

**Files:**
- Create: `Sources/MetalANNSCore/Shaders/Search.metal`
- Create: `Sources/MetalANNSCore/FullGPUSearch.swift` (Swift-side dispatch)
- Create: `Tests/MetalANNSTests/FullGPUSearchTests.swift`
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (use new search when available)

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/FullGPUSearchTests.swift
import Testing
import MetalANNSCore
import Metal

@Suite("Full GPU Search")
struct FullGPUSearchTests {
    @Test func gpuSearchReturnsK() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let context = try MetalContext()

        let n = 200, dim = 16, degree = 8
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
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

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let results = try await FullGPUSearch.search(
            context: context, query: query, vectors: vb, graph: gb,
            entryPoint: Int(cpuEntry), k: 5, ef: 32, metric: .cosine
        )

        #expect(results.count == 5)
        // Results should be sorted by score ascending
        for i in 1..<results.count {
            #expect(results[i].score >= results[i - 1].score)
        }
    }

    @Test func gpuSearchRecallMatchesHybrid() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let context = try MetalContext()

        let n = 500, dim = 32, degree = 16
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        let (cpuGraph, cpuEntry) = try await NNDescentCPU.build(
            vectors: vectors, degree: degree, metric: .cosine, maxIterations: 15
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

        // Compare recall between hybrid and full-GPU
        var hybridRecall: Float = 0
        var fullGPURecall: Float = 0
        let queryCount = 20
        let k = 10

        for _ in 0..<queryCount {
            let query = (0..<dim).map { _ in Float.random(in: -1...1) }

            // Brute force ground truth
            let bruteForce = vectors.enumerated()
                .map { (UInt32($0.offset), SIMDDistance.cosine(query, $0.element)) }
                .sorted { $0.1 < $1.1 }
                .prefix(k)
                .map(\.0)
            let groundTruth = Set(bruteForce)

            // Hybrid search
            let hybrid = try await SearchGPU.search(
                context: context, query: query, vectors: vb, graph: gb,
                entryPoint: Int(cpuEntry), k: k, ef: 64, metric: .cosine
            )
            hybridRecall += Float(Set(hybrid.map(\.internalID)).intersection(groundTruth).count) / Float(k)

            // Full GPU search
            let full = try await FullGPUSearch.search(
                context: context, query: query, vectors: vb, graph: gb,
                entryPoint: Int(cpuEntry), k: k, ef: 64, metric: .cosine
            )
            fullGPURecall += Float(Set(full.map(\.internalID)).intersection(groundTruth).count) / Float(k)
        }

        hybridRecall /= Float(queryCount)
        fullGPURecall /= Float(queryCount)

        // Full GPU should have comparable recall (within 5% of hybrid)
        #expect(fullGPURecall > hybridRecall - 0.05)
        // Both should be reasonable
        #expect(fullGPURecall > 0.80)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/FullGPUSearchTests 2>&1 | grep -E '(PASS|FAIL|error:)'`
Expected: FAIL with "FullGPUSearch not found"

**Step 3: Write the Metal kernel**

```cpp
// Sources/MetalANNSCore/Shaders/Search.metal
#include <metal_stdlib>
using namespace metal;

// ── Shared memory candidate queue ──
// Each threadgroup processes ONE query.
// Threads cooperate to expand candidates and compute distances.

// Constants
constant uint MAX_EF = 256;     // max beam width
constant uint MAX_VISITED = 4096; // max visited set (open-addressed hash table)

// Candidate entry: packed nodeID + distance
struct CandidateEntry {
    uint nodeID;
    float distance;
};

// ── Distance computation (shared by all kernels) ──
inline float compute_distance(
    device const float* vectors,
    device const float* query,
    uint nodeID,
    uint dim,
    uint metric_type  // 0=cosine, 1=l2, 2=innerProduct
) {
    device const float* vec = vectors + nodeID * dim;

    if (metric_type == 0) {
        // Cosine
        float dot = 0.0f, normQ = 0.0f, normV = 0.0f;
        for (uint d = 0; d < dim; d++) {
            float q = query[d];
            float v = vec[d];
            dot += q * v;
            normQ += q * q;
            normV += v * v;
        }
        float denom = sqrt(normQ) * sqrt(normV);
        return (denom < 1e-10f) ? 1.0f : (1.0f - dot / denom);
    } else if (metric_type == 1) {
        // L2
        float sum = 0.0f;
        for (uint d = 0; d < dim; d++) {
            float diff = query[d] - vec[d];
            sum += diff * diff;
        }
        return sum;
    } else {
        // Inner Product
        float dot = 0.0f;
        for (uint d = 0; d < dim; d++) {
            dot += query[d] * vec[d];
        }
        return -dot;
    }
}

// ── Open-addressed hash set for visited tracking ──
// Returns true if nodeID was NOT already visited (and inserts it)
inline bool try_visit(
    threadgroup atomic_uint* visited_table,
    uint nodeID,
    uint table_size
) {
    uint hash = nodeID % table_size;
    for (uint probe = 0; probe < 32; probe++) {
        uint slot = (hash + probe) % table_size;
        uint expected = UINT_MAX;
        bool exchanged = atomic_compare_exchange_weak_explicit(
            &visited_table[slot],
            &expected,
            nodeID,
            memory_order_relaxed,
            memory_order_relaxed
        );
        if (exchanged) return true;      // we inserted it → not visited before
        if (expected == nodeID) return false; // already there → already visited
        // collision → probe next slot
    }
    return false; // table full or too many probes → treat as visited
}

// ── Main beam search kernel ──
// One threadgroup per query. Threads within the group cooperate on neighbor expansion.
kernel void beam_search(
    device const float* vectors       [[buffer(0)]],  // [nodeCount * dim]
    device const uint*  adjacency     [[buffer(1)]],  // [nodeCount * degree]
    device const float* query         [[buffer(2)]],  // [dim]
    device float*       output_dists  [[buffer(3)]],  // [k] output
    device uint*        output_ids    [[buffer(4)]],  // [k] output
    constant uint&      node_count    [[buffer(5)]],
    constant uint&      degree        [[buffer(6)]],
    constant uint&      dim           [[buffer(7)]],
    constant uint&      k             [[buffer(8)]],
    constant uint&      ef            [[buffer(9)]],
    constant uint&      entry_point   [[buffer(10)]],
    constant uint&      metric_type   [[buffer(11)]],
    uint tid        [[thread_position_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]]
) {
    // ── Shared memory allocation ──
    threadgroup CandidateEntry candidates[MAX_EF];
    threadgroup CandidateEntry results[MAX_EF];
    threadgroup atomic_uint visited[MAX_VISITED];
    threadgroup atomic_uint candidate_count;
    threadgroup atomic_uint result_count;
    threadgroup uint candidate_head;  // next candidate to expand

    // ── Initialize ──
    if (tid == 0) {
        atomic_store_explicit(&candidate_count, 0, memory_order_relaxed);
        atomic_store_explicit(&result_count, 0, memory_order_relaxed);
        candidate_head = 0;
    }

    // Clear visited table
    for (uint i = tid; i < MAX_VISITED; i += tg_size) {
        atomic_store_explicit(&visited[i], UINT_MAX, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Seed with entry point ──
    if (tid == 0) {
        float entryDist = compute_distance(vectors, query, entry_point, dim, metric_type);
        try_visit(visited, entry_point, MAX_VISITED);

        candidates[0] = {entry_point, entryDist};
        results[0] = {entry_point, entryDist};
        atomic_store_explicit(&candidate_count, 1, memory_order_relaxed);
        atomic_store_explicit(&result_count, 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Main beam search loop ──
    uint ef_limit = min(ef, node_count);

    for (uint iter = 0; iter < ef_limit * 2; iter++) {
        // Thread 0 pops the best unvisited candidate
        threadgroup uint current_node;
        threadgroup float current_dist;
        threadgroup bool should_break;
        threadgroup uint num_neighbors;

        if (tid == 0) {
            uint head = candidate_head;
            uint count = atomic_load_explicit(&candidate_count, memory_order_relaxed);
            should_break = (head >= count);

            if (!should_break) {
                current_node = candidates[head].nodeID;
                current_dist = candidates[head].distance;
                candidate_head = head + 1;

                uint rcount = atomic_load_explicit(&result_count, memory_order_relaxed);
                if (rcount >= ef_limit && current_dist > results[rcount - 1].distance) {
                    should_break = true;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (should_break) break;

        // ── Parallel neighbor expansion ──
        // Each thread handles a subset of neighbors
        if (tid == 0) {
            num_neighbors = degree;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint slot = tid; slot < num_neighbors; slot += tg_size) {
            uint neighborID = adjacency[current_node * degree + slot];
            if (neighborID >= node_count) continue;

            // Check visited
            if (!try_visit(visited, neighborID, MAX_VISITED)) continue;

            // Compute distance
            float dist = compute_distance(vectors, query, neighborID, dim, metric_type);

            // Try to insert into results
            uint rcount = atomic_load_explicit(&result_count, memory_order_relaxed);
            bool should_add = (rcount < ef_limit) ||
                              (dist < results[rcount - 1].distance);

            if (should_add) {
                // Sorted insertion into results (thread-safe via sequential section)
                // NOTE: for correctness, only one thread should modify results at a time
                // In practice, we batch neighbor distances and do a single merge step
                // For simplicity in v1: use atomic slot allocation + post-sort

                uint slot_idx = atomic_fetch_add_explicit(&result_count, 1, memory_order_relaxed);
                if (slot_idx < MAX_EF) {
                    results[slot_idx] = {neighborID, dist};
                }

                uint cslot = atomic_fetch_add_explicit(&candidate_count, 1, memory_order_relaxed);
                if (cslot < MAX_EF) {
                    candidates[cslot] = {neighborID, dist};
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Thread 0: sort and trim results + candidates ──
        if (tid == 0) {
            uint rcount = min(atomic_load_explicit(&result_count, memory_order_relaxed), (uint)MAX_EF);
            // Insertion sort (small arrays, ~ef elements)
            for (uint i = 1; i < rcount; i++) {
                CandidateEntry key = results[i];
                int j = (int)i - 1;
                while (j >= 0 && results[j].distance > key.distance) {
                    results[j + 1] = results[j];
                    j--;
                }
                results[j + 1] = key;
            }
            if (rcount > ef_limit) {
                rcount = ef_limit;
            }
            atomic_store_explicit(&result_count, rcount, memory_order_relaxed);

            // Sort candidates too
            uint ccount = min(atomic_load_explicit(&candidate_count, memory_order_relaxed), (uint)MAX_EF);
            for (uint i = 1; i < ccount; i++) {
                CandidateEntry key = candidates[i];
                int j = (int)i - 1;
                while (j >= 0 && candidates[j].distance > key.distance) {
                    candidates[j + 1] = candidates[j];
                    j--;
                }
                candidates[j + 1] = key;
            }
            atomic_store_explicit(&candidate_count, ccount, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write output (thread 0 only) ──
    if (tid == 0) {
        uint rcount = atomic_load_explicit(&result_count, memory_order_relaxed);
        uint output_count = min(k, rcount);
        for (uint i = 0; i < output_count; i++) {
            output_ids[i] = results[i].nodeID;
            output_dists[i] = results[i].distance;
        }
        // Pad remaining with sentinels
        for (uint i = output_count; i < k; i++) {
            output_ids[i] = UINT_MAX;
            output_dists[i] = FLT_MAX;
        }
    }
}
```

**IMPORTANT NOTES ON THE KERNEL:**
- This is a v1 kernel — the sorting bottleneck (thread 0 does insertion sort) can be improved with parallel merge in v2
- `MAX_EF = 256` and `MAX_VISITED = 4096` limit shared memory usage. Verify these fit within `device.maxThreadgroupMemoryLength` (typically 32KB on Apple Silicon)
- Memory: `256 * 8 (candidates) + 256 * 8 (results) + 4096 * 4 (visited) = ~20KB` — fits in 32KB
- The `try_visit` hash table uses open addressing with linear probing, max 32 probes

**Step 4: Write the Swift dispatch wrapper**

```swift
// Sources/MetalANNSCore/FullGPUSearch.swift
import Metal

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
    ) async throws -> [SearchResult] {
        let pipeline = try await context.pipelineCache.pipeline(for: "beam_search")

        let queryBuffer = context.device.makeBuffer(
            bytes: query, length: query.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        let outputDistBuffer = context.device.makeBuffer(
            length: k * MemoryLayout<Float>.stride, options: .storageModeShared
        )!

        let outputIDBuffer = context.device.makeBuffer(
            length: k * MemoryLayout<UInt32>.stride, options: .storageModeShared
        )!

        var nodeCount = UInt32(graph.nodeCount)
        var degree = UInt32(graph.degree)
        var dim = UInt32(vectors.dim)
        var kVal = UInt32(k)
        var efVal = UInt32(min(ef, 256)) // clamp to MAX_EF
        var entry = UInt32(entryPoint)
        var metricType: UInt32 = switch metric {
        case .cosine: 0
        case .l2: 1
        case .innerProduct: 2
        }

        try await context.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ANNSError.searchFailed("Failed to create encoder")
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(vectors.buffer, offset: 0, index: 0)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 1)
            encoder.setBuffer(queryBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputDistBuffer, offset: 0, index: 3)
            encoder.setBuffer(outputIDBuffer, offset: 0, index: 4)
            encoder.setBytes(&nodeCount, length: 4, index: 5)
            encoder.setBytes(&degree, length: 4, index: 6)
            encoder.setBytes(&dim, length: 4, index: 7)
            encoder.setBytes(&kVal, length: 4, index: 8)
            encoder.setBytes(&efVal, length: 4, index: 9)
            encoder.setBytes(&entry, length: 4, index: 10)
            encoder.setBytes(&metricType, length: 4, index: 11)

            // One threadgroup, threads = degree (or suitable width)
            let threadgroupWidth = min(Int(degree), pipeline.maxTotalThreadsPerThreadgroup)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }

        // Read results
        let distPtr = outputDistBuffer.contents().bindMemory(to: Float.self, capacity: k)
        let idPtr = outputIDBuffer.contents().bindMemory(to: UInt32.self, capacity: k)

        var results: [SearchResult] = []
        for i in 0..<k {
            let nodeID = idPtr[i]
            if nodeID == UInt32.max { break }
            results.append(SearchResult(id: "", score: distPtr[i], internalID: nodeID))
        }

        return results
    }
}
```

**Step 5: Run tests, verify GREEN**

Run: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/FullGPUSearchTests 2>&1 | grep -E '(PASS|FAIL|error:)'`
Expected: PASS

**Step 6: Wire into ANNSIndex**

In `ANNSIndex.swift`, update the search method to use `FullGPUSearch` when GPU is available:

```swift
// In search method, replace SearchGPU.search with:
if let ctx = context {
    rawResults = try await FullGPUSearch.search(
        context: ctx, query: query, vectors: vectors!, graph: graph!,
        entryPoint: Int(entryPoint), k: effectiveK, ef: effectiveEf, metric: configuration.metric
    )
}
```

**Step 7: Regression + commit**

```bash
git add Sources/MetalANNSCore/Shaders/Search.metal Sources/MetalANNSCore/FullGPUSearch.swift Tests/MetalANNSTests/FullGPUSearchTests.swift Sources/MetalANNS/ANNSIndex.swift
git commit -m "feat: full GPU beam search kernel with shared-memory candidate queue"
```

---

### Task 25: CAGRA Post-Processing (Graph Pruning)

**Files:**
- Create: `Sources/MetalANNSCore/GraphPruner.swift`
- Create: `Tests/MetalANNSTests/GraphPrunerTests.swift`
- Modify: `Sources/MetalANNSCore/NNDescentGPU.swift` (optional post-processing step)
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (call pruner after build)

**Purpose**: After NN-Descent converges, remove redundant edges (where neighbor A is reachable via neighbor B with shorter total path) and merge reverse edges. Produces a sparser, higher-quality graph.

**Algorithm — Path-based pruning (CAGRA-style):**

For each node `u`:
1. Sort neighbors by distance (already done by bitonic sort)
2. Initialize empty pruned list
3. For each candidate neighbor `v` (in distance order):
   - Check if any already-selected pruned neighbor `w` satisfies: `d(w, v) < d(u, v)`
   - If yes: `v` is redundant (reachable via `w`) — skip it
   - If no: add `v` to pruned list
4. Pad pruned list back to `degree` with sentinel values

**Test design:**
1. `pruningReducesRedundancy` — build graph, prune, verify average neighbor distance improves or stays same while connectivity is maintained
2. `pruningMaintainsRecall` — build graph, prune, search, verify recall doesn't drop more than 2%

**Commit**: `feat: add CAGRA-style graph pruning for higher quality edges`

---

## Phase 9: Float16 Support

**Goal**: Halve memory bandwidth for vector storage while maintaining search quality.

### Task 26: Float16 VectorBuffer + Distance Kernels

**Files:**
- Create: `Sources/MetalANNSCore/Float16VectorBuffer.swift`
- Create: `Sources/MetalANNSCore/Shaders/DistanceFloat16.metal`
- Create: `Tests/MetalANNSTests/Float16Tests.swift`

**Float16VectorBuffer** — same API as VectorBuffer but stores `Float16` (Metal's `half`):
- `init(capacity:dim:device:)` — allocates `capacity * dim * 2` bytes
- `insert(vector:at:)` — converts `[Float]` → `[Float16]` on write
- `vector(at:)` → `[Float]` — converts back to Float32 on read
- `buffer` — MTLBuffer with half-precision data

**DistanceFloat16.metal** — Float16 variants of all three distance kernels:
```cpp
kernel void cosine_distance_f16(
    device const half *query [[buffer(0)]],
    device const half *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],  // output stays Float32
    constant uint &dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
)
```

**Key**: Accumulation in Float32 to avoid precision loss. Only storage and bandwidth are Float16.

**Tests:**
1. `float16DistanceMatchesFloat32` — verify Float16 distance within 1% of Float32
2. `float16RecallComparable` — build with Float16, search, verify recall within 2% of Float32

**Commit**: `feat: add Float16 vector buffer and distance kernels`

### Task 27: Float16 Construction + Search Integration

**Files:**
- Modify: `Sources/MetalANNSCore/NNDescentGPU.swift` (support Float16VectorBuffer)
- Modify: `Sources/MetalANNSCore/FullGPUSearch.swift` (support Float16)
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (use Float16 when `useFloat16 == true`)
- Modify: `Sources/MetalANNSCore/IndexSerializer.swift` (format v2 for Float16)
- Create: `Tests/MetalANNSTests/Float16IntegrationTests.swift`

**Key changes:**
- `IndexSerializer` format v2: header byte at offset 20 encodes `0=float32, 1=float16` instead of just metric
- `ANNSIndex.build` checks `configuration.useFloat16` and creates appropriate buffer type
- All GPU kernels detect buffer type and dispatch to correct kernel variant

**Tests:**
1. `float16FullLifecycle` — build, search, insert, save, load with Float16 enabled
2. `float16SaveLoadPreservesData` — verify round-trip preserves Float16 data

**Commit**: `feat: integrate Float16 into construction, search, and persistence`

---

## Phase 10: Scalability Primitives

### Task 28: Batch Insert

**Files:**
- Create: `Sources/MetalANNSCore/BatchIncrementalBuilder.swift`
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (add `batchInsert` method)
- Create: `Tests/MetalANNSTests/BatchInsertTests.swift`

**Algorithm:**
1. Insert all new vectors into VectorBuffer
2. For each new vector in parallel, find `degree` nearest neighbors via beam search
3. Set all new nodes' neighbor lists
4. Batch reverse-update: for each affected existing node, re-evaluate its neighbor list
5. Single pass is more efficient than N individual inserts

**Public API addition:**
```swift
public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws
```

**Tests:**
1. `batchInsertFasterThanSequential` — time 100 individual inserts vs 1 batch of 100
2. `batchInsertRecall` — verify recall after batch insert matches individual insert quality

**Commit**: `feat: add batch insert for efficient bulk addition`

---

### Task 29: Hard Deletion + Compaction

**Files:**
- Create: `Sources/MetalANNSCore/IndexCompactor.swift`
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (add `compact` method)
- Create: `Tests/MetalANNSTests/CompactionTests.swift`

**Algorithm — Compaction:**
1. Collect all non-deleted internal IDs
2. Create new VectorBuffer and GraphBuffer with reduced capacity
3. Copy non-deleted vectors to new buffer, building old→new ID mapping
4. Rebuild graph using NNDescentGPU on the new (smaller) vector set
5. Rebuild IDMap with remapped internal IDs
6. Reset SoftDeletion (no more deleted nodes)

**Public API addition:**
```swift
public func compact() async throws  // removes soft-deleted nodes, rebuilds graph
```

**Tests:**
1. `compactReducesMemory` — delete 50%, compact, verify buffer capacity decreased
2. `compactMaintainsRecall` — compact after deletions, verify recall is >= pre-deletion recall

**Commit**: `feat: add hard deletion via index compaction`

---

### Task 30: Memory-Mapped I/O

**Files:**
- Create: `Sources/MetalANNSCore/MmapIndexLoader.swift`
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (add `loadMmap` factory method)
- Create: `Tests/MetalANNSTests/MmapTests.swift`

**Key API:**
```swift
MTLDevice.makeBuffer(bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: ...)
```

**Algorithm:**
1. `mmap` the binary file into virtual memory
2. Parse header from mapped memory (same format as IndexSerializer)
3. Create MTLBuffer via `makeBuffer(bytesNoCopy:)` pointing directly at the mapped region
4. Zero-copy: GPU reads directly from mmap'd pages, OS pages in on demand

**Constraints:**
- File must be page-aligned (pad to 4096-byte boundaries)
- Buffer is read-only (mutations require copy-on-write)
- Caller must keep the file handle alive

**Public API addition:**
```swift
public static func loadMmap(from url: URL) async throws -> ANNSIndex
```

**Tests:**
1. `mmapProducesSameResults` — load via mmap, search, compare to normal load
2. `mmapDoesNotLoadEntireFile` — verify resident memory < file size (check via `mach_task_basic_info`)

**Commit**: `feat: add memory-mapped index loading for large indices`

---

## Phase 11: Advanced Search

### Task 31: Filtered Search with Metadata Predicates

**Files:**
- Create: `Sources/MetalANNSCore/MetadataStore.swift`
- Create: `Sources/MetalANNSCore/SearchFilter.swift`
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (add metadata + filtered search)
- Modify: `Sources/MetalANNSCore/IndexSerializer.swift` (persist metadata)
- Create: `Tests/MetalANNSTests/FilteredSearchTests.swift`

**MetadataStore:**
```swift
public struct MetadataStore: Sendable, Codable {
    // Column-oriented storage for efficient predicate evaluation
    private var stringColumns: [String: [UInt32: String]]  // column → (internalID → value)
    private var floatColumns: [String: [UInt32: Float]]
    private var intColumns: [String: [UInt32: Int64]]

    public mutating func set(_ column: String, value: String, for id: UInt32)
    public mutating func set(_ column: String, value: Float, for id: UInt32)
    public mutating func set(_ column: String, value: Int64, for id: UInt32)
    public func matches(id: UInt32, filter: SearchFilter) -> Bool
}
```

**SearchFilter (predicate DSL):**
```swift
public enum SearchFilter: Sendable {
    case equals(column: String, value: String)
    case greaterThan(column: String, value: Float)
    case lessThan(column: String, value: Float)
    case `in`(column: String, values: Set<String>)
    case and([SearchFilter])
    case or([SearchFilter])
    case not(SearchFilter)
}
```

**Public API additions:**
```swift
public func setMetadata(_ column: String, value: String, for id: String) throws
public func search(query: [Float], k: Int, filter: SearchFilter?) async throws -> [SearchResult]
```

**Search strategy**: Post-filter with over-fetch. Search with `ef = k * (1 + deletedRatio + filterEstimate)`, filter results, return top-k. In-traversal filtering is a future optimization.

**Tests:**
1. `filteredSearchReturnsOnlyMatching` — build, set category metadata, search with filter, verify all results match
2. `filteredSearchRecall` — verify recall is reasonable with filter applied
3. `compoundFilterWorks` — test AND/OR/NOT predicates

**Commit**: `feat: add filtered search with metadata predicates`

---

### Task 32: Range Search

**Files:**
- Create: `Sources/MetalANNSCore/RangeSearch.swift`
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (add `rangeSearch` method)
- Create: `Tests/MetalANNSTests/RangeSearchTests.swift`

**Algorithm**: Modified beam search that doesn't terminate at k results but continues until all candidates are beyond the distance threshold.

**Public API:**
```swift
public func rangeSearch(query: [Float], maxDistance: Float, limit: Int = 1000) async throws -> [SearchResult]
```

**Tests:**
1. `rangeSearchReturnsWithinThreshold` — verify all results have `score <= maxDistance`
2. `rangeSearchFindsExactMatch` — insert vector, range search with small threshold, find it

**Commit**: `feat: add range search with distance threshold`

---

### Task 33: Runtime Metric Selection

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (add metric parameter to search)
- Modify: `Sources/MetalANNSCore/FullGPUSearch.swift` (accept runtime metric)
- Create: `Tests/MetalANNSTests/RuntimeMetricTests.swift`

**Key change**: `search(query:k:metric:)` with optional metric override. Default uses index's built metric. Override metric is passed to the search kernel at dispatch time.

**Constraints**: Graph was built with one metric, so search with a different metric will have degraded recall. Document this tradeoff.

**Public API:**
```swift
public func search(query: [Float], k: Int, metric: Metric? = nil) async throws -> [SearchResult]
```

**Tests:**
1. `searchWithDifferentMetric` — build with cosine, search with L2, verify results are valid (sorted by L2 distance)
2. `defaultMetricMatchesBuildMetric` — verify nil metric uses index's configured metric

**Commit**: `feat: support runtime metric selection at query time`

---

## Phase 12: Large-Scale

### Task 34: Disk-Backed Index

**Files:**
- Create: `Sources/MetalANNSCore/DiskBackedVectorBuffer.swift`
- Create: `Sources/MetalANNSCore/DiskBackedGraphBuffer.swift`
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (disk-backed mode)
- Create: `Tests/MetalANNSTests/DiskBackedTests.swift`

**Architecture**: Keep vectors and graph on disk via `mmap`. Page in on-demand during search. Use a small LRU cache for recently accessed pages.

**Key classes:**
```swift
public final class DiskBackedVectorBuffer: @unchecked Sendable {
    // mmap'd file, pages in vectors on demand
    public func vector(at index: Int) -> [Float]  // triggers page-in
    public var dim: Int
    public var count: Int
}
```

**Constraints**: Read-only for disk-backed mode. Mutations require loading into RAM first.

**Tests:**
1. `diskBackedSearchWorks` — save index, open disk-backed, search, verify results
2. `diskBackedMemoryEfficient` — verify RSS stays below file size

**Commit**: `feat: add disk-backed index for memory-constrained devices`

---

### Task 35: Sharded Indices (IVF-Style)

**Files:**
- Create: `Sources/MetalANNS/ShardedIndex.swift`
- Create: `Sources/MetalANNSCore/KMeans.swift` (simple k-means for partitioning)
- Create: `Tests/MetalANNSTests/ShardedIndexTests.swift`

**Architecture:**
1. **Build**: Run k-means on input vectors to create `numShards` centroids
2. **Assign**: Each vector goes to its nearest centroid's shard
3. **Build per-shard**: Each shard gets its own `ANNSIndex`
4. **Search**: Compute distance to all centroids, search top-`nprobe` shards, merge results

**Public API:**
```swift
public actor ShardedIndex {
    public init(numShards: Int = 16, nprobe: Int = 4, configuration: IndexConfiguration = .default)
    public func build(vectors: [[Float]], ids: [String]) async throws
    public func search(query: [Float], k: Int) async throws -> [SearchResult]
}
```

**Tests:**
1. `shardedSearchRecall` — 5000 vectors, 16 shards, verify recall@10 > 0.85
2. `shardedScalesWithShards` — verify search time scales sub-linearly with vector count

**Commit**: `feat: add sharded index with IVF-style partitioning`

---

## Summary

| Phase | Tasks | New Tests | Key Deliverable |
|-------|-------|-----------|----------------|
| 7 (CPU Quick Wins) | 22–23 | ~6 | 4-8x CPU distance speedup, concurrent batch search |
| 8 (GPU Search) | 24–25 | ~4 | 5-10x search latency reduction |
| 9 (Float16) | 26–27 | ~4 | 2x memory reduction |
| 10 (Scalability) | 28–30 | ~6 | Batch insert, compaction, mmap |
| 11 (Advanced Search) | 31–33 | ~7 | Filtered search, range search, runtime metrics |
| 12 (Large-Scale) | 34–35 | ~4 | Disk-backed, sharded indices |
| **Total** | **14 tasks** | **~31 tests** | **Production-grade vector search** |

**Estimated commit range**: 25–38 (14 implementation commits + potential intermediate commits)

---

## Execution Order Recommendation

**Start with Phase 7** — lowest risk, immediate value, validates the testing/integration pipeline for v2 work.

**Then Phase 8** — the defining feature. Full GPU search kernel is the most complex task but provides the largest single performance improvement.

**Phase 9 after Phase 8** — Float16 builds on top of the GPU search kernel changes.

**Phases 10–12 can be parallelized** — batch insert, filtered search, and disk-backed index are independent streams of work.
