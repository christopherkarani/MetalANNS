# Phase 25: GPU ADC Linear Scan — Extraction & Public API

### Mission

Extract the existing GPU ADC implementation from `IVFPQIndex.gpuADCDistances()` into a
standalone, reusable `GPUADCSearch` type in `Sources/MetalANNSCore/`. Expose a two-level
API: a low-level `computeDistances()` (raw `[Float]`) for internal callers like
`IVFPQIndex`, and a high-level `search()` (sorted top-k `[SearchResult]`) for public use.
Then rewire `IVFPQIndex` to delegate to `GPUADCSearch`.

This is an **extraction/refactor** — not a greenfield implementation. The GPU kernels
(`pq_compute_distance_table`, `pq_adc_scan`) and Metal dispatch logic already work
and are tested. The goal is reuse, testability, and a clean public API.

---

### Verified Codebase Facts

Read each file before touching it.

| Fact | Source |
|------|--------|
| `IVFPQIndex` is a Swift actor in `Sources/MetalANNS/IVFPQIndex.swift` | `IVFPQIndex.swift:5` |
| `gpuADCDistances()` at lines 421-538 is a complete GPU ADC implementation | `IVFPQIndex.swift:421-538` |
| It creates 5 Metal buffers: query, codebook, distTable, codes, distances | `IVFPQIndex.swift:459-482` |
| Dispatches `pq_compute_distance_table` (2D: M × Ks) then `pq_adc_scan` (1D: vectorCount) | `IVFPQIndex.swift:500-533` |
| Reads back distances via `UnsafeBufferPointer` | `IVFPQIndex.swift:536-537` |
| `cpuADCDistances()` at lines 393-418 is the CPU reference path | `IVFPQIndex.swift:393-418` |
| `distancesForCluster()` at line 345 dispatches GPU vs CPU based on `forceGPU` and candidate count ≥ 64 | `IVFPQIndex.swift:345-391` |
| `flattenCodebooks()` at line 540 converts PQ codebooks to flat `[Float]` | `IVFPQIndex.swift:540-551` |
| `flattenedCodebooks` is cached as a `private var` on `IVFPQIndex` | `IVFPQIndex.swift:28` |
| `MetalContext` has `execute()` and `executeOnPool()` — both return after GPU completion | `MetalDevice.swift:41,55` |
| `PipelineCache` is an actor; `pipeline(for:)` lazily compiles and caches | `PipelineCache.swift:6,16` |
| `ProductQuantizer` has `numSubspaces`, `centroidsPerSubspace` (always 256), `subspaceDimension`, `codebooks: [[[Float]]]` | `ProductQuantizer.swift:3-7` |
| `PQVectorBuffer.code(at:)` returns `[UInt8]` for one vector | `PQVectorBuffer.swift:74-80` |
| `SearchResult` has `id: String`, `score: Float`, `internalID: UInt32` — lives in MetalANNSCore | `SearchResult.swift:1-11` |
| `pq_compute_distance_table` kernel: buffers 0-5 = query, codebooks, distTable, M, Ks, subspaceDim; 2D grid (M × Ks) | `PQDistance.metal:4-30` |
| `pq_adc_scan` kernel: buffers 0-5 = codes, distTable, distances, M, Ks, vectorCount; threadgroup(0) = tgDistTable; 1D grid (vectorCount) | `PQDistance.metal:32-61` |
| `pq_adc_scan` loads the full M×Ks table into threadgroup memory with stride-32 cooperative load | `PQDistance.metal:48-51` |
| Existing GPU test in `IVFPQGPUTests.swift` validates GPU vs CPU distances within 1e-3 | `IVFPQGPUTests.swift:9-49` |
| `GPUADCSearch` should live in `Sources/MetalANNSCore/` since it depends on `MetalContext`, `ProductQuantizer`, `PipelineCache` | directory structure |

---

### TDD Implementation Order

Work strictly test-first.

**Round 1** — `GPUADCSearch` unit tests
Write `GPUADCSearchTests.swift`. Tests fail to compile. Implement `GPUADCSearch`. Tests pass.

**Round 2** — Rewire `IVFPQIndex`
Replace `gpuADCDistances()` body with a call to `GPUADCSearch.computeDistances()`.
All existing `IVFPQGPUTests` must still pass (no regression).

**Round 3** — High-level `search()` API test
Add a test that exercises `GPUADCSearch.search()` returning sorted top-k `[SearchResult]`.

Run after every step:
```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E "PASS|FAIL|error:"
```

---

### Step 1: Create `GPUADCSearch.swift`

**File**: `Sources/MetalANNSCore/GPUADCSearch.swift`

```swift
import Foundation
import Metal

public enum GPUADCSearch {

    // MARK: - Low-level: raw distances

    /// Compute approximate ADC distances from a query to all provided PQ codes on GPU.
    ///
    /// Returns a `[Float]` of length `codes.count` where `result[i]` is the approximate
    /// distance from `query` to the vector encoded as `codes[i]`.
    ///
    /// - Parameters:
    ///   - context: Metal context (device, queue, pipeline cache).
    ///   - query: Full-precision query vector (dimension = pq.numSubspaces * pq.subspaceDimension).
    ///   - pq: Trained ProductQuantizer (provides codebooks and subspace geometry).
    ///   - codes: PQ codes, one `[UInt8]` per candidate vector (each of length `pq.numSubspaces`).
    ///   - flatCodebooks: Pre-flattened codebook array (M * Ks * subspaceDim floats).
    ///            Pass `nil` to compute on the fly. Callers like `IVFPQIndex` cache this.
    /// - Returns: Array of approximate distances, one per code.
    public static func computeDistances(
        context: MetalContext,
        query: [Float],
        pq: ProductQuantizer,
        codes: [[UInt8]],
        flatCodebooks: [Float]? = nil
    ) async throws -> [Float]

    // MARK: - High-level: sorted top-k results

    /// GPU ADC scan returning the top-k nearest results sorted by ascending distance.
    ///
    /// Convenience wrapper around `computeDistances()` that pairs distances with IDs,
    /// sorts, and returns the top-k.
    ///
    /// - Parameters:
    ///   - context: Metal context.
    ///   - query: Full-precision query vector.
    ///   - pq: Trained ProductQuantizer.
    ///   - codes: PQ codes, one per candidate.
    ///   - ids: External string IDs, parallel to `codes` (ids[i] corresponds to codes[i]).
    ///   - k: Number of results to return.
    ///   - flatCodebooks: Optional pre-flattened codebooks (pass nil to compute internally).
    /// - Returns: Up to `k` results sorted by ascending approximate distance.
    public static func search(
        context: MetalContext,
        query: [Float],
        pq: ProductQuantizer,
        codes: [[UInt8]],
        ids: [String],
        k: Int,
        flatCodebooks: [Float]? = nil
    ) async throws -> [SearchResult]
}
```

---

### Step 2: Implement `computeDistances()`

Extract the GPU dispatch logic from `IVFPQIndex.gpuADCDistances()` (lines 421-538).
The implementation follows the same structure:

1. Validate `query.count == pq.numSubspaces * pq.subspaceDimension`
2. Flatten `codes: [[UInt8]]` into a contiguous `[UInt8]` (candidateCodes)
3. Flatten codebooks (use `flatCodebooks` parameter if non-nil, else compute via `flattenCodebooks(from:)`)
4. Allocate 5 Metal buffers: query, codebooks, distTable, codes, distances
5. Get pipelines via `context.pipelineCache.pipeline(for: "pq_compute_distance_table")` and `"pq_adc_scan"`
6. Dispatch inside `context.execute { commandBuffer in ... }`:
   - Encode `pq_compute_distance_table` — 2D grid `MTLSize(width: M, height: Ks, depth: 1)`, threadgroup `MTLSize(width: 8, height: 8, depth: 1)`
   - Encode `pq_adc_scan` — 1D grid `MTLSize(width: vectorCount, height: 1, depth: 1)`, threadgroup `min(vectorCount, scanPipeline.maxTotalThreadsPerThreadgroup)`, set threadgroup memory length `M * Ks * MemoryLayout<Float>.stride` at index 0
7. Read back distances from `distancesBuffer.contents()`

**Important**: The `flattenCodebooks` helper must also live in `GPUADCSearch` (as a `public static func`) since `IVFPQIndex` will delegate to it.

```swift
public static func flattenCodebooks(from pq: ProductQuantizer) -> [Float] {
    var flattened: [Float] = []
    flattened.reserveCapacity(
        pq.numSubspaces * pq.centroidsPerSubspace * pq.subspaceDimension
    )
    for subspace in 0..<pq.numSubspaces {
        for centroid in 0..<pq.centroidsPerSubspace {
            flattened.append(contentsOf: pq.codebooks[subspace][centroid])
        }
    }
    return flattened
}
```

---

### Step 3: Implement `search()`

```swift
public static func search(
    context: MetalContext,
    query: [Float],
    pq: ProductQuantizer,
    codes: [[UInt8]],
    ids: [String],
    k: Int,
    flatCodebooks: [Float]? = nil
) async throws -> [SearchResult] {
    guard codes.count == ids.count else {
        throw ANNSError.constructionFailed("codes and ids count mismatch")
    }
    guard k > 0 else { return [] }
    guard !codes.isEmpty else { return [] }

    let distances = try await computeDistances(
        context: context,
        query: query,
        pq: pq,
        codes: codes,
        flatCodebooks: flatCodebooks
    )

    // Pair distances with IDs, sort, take top-k
    var results: [SearchResult] = []
    results.reserveCapacity(min(k, distances.count))
    for (i, distance) in distances.enumerated() {
        results.append(SearchResult(id: ids[i], score: distance, internalID: UInt32(i)))
    }
    results.sort { $0.score < $1.score }
    if results.count > k {
        results.removeSubrange(k...)
    }
    return results
}
```

---

### Step 4: Rewire `IVFPQIndex`

**File**: `Sources/MetalANNS/IVFPQIndex.swift`

**4a. Replace `gpuADCDistances()` body**

Replace lines 421-538 with a delegation call:

```swift
private func gpuADCDistances(
    queryResidual: [Float],
    candidateIDs: [UInt32],
    pq: ProductQuantizer,
    vectorBuffer: PQVectorBuffer
) async throws -> [Float] {
    guard let context else {
        throw ANNSError.searchFailed("Metal context unavailable for GPU ADC")
    }
    guard !candidateIDs.isEmpty else { return [] }

    let codes = candidateIDs.map { vectorBuffer.code(at: Int($0)) }

    if flattenedCodebooks.isEmpty {
        flattenedCodebooks = GPUADCSearch.flattenCodebooks(from: pq)
    }

    return try await GPUADCSearch.computeDistances(
        context: context,
        query: queryResidual,
        pq: pq,
        codes: codes,
        flatCodebooks: flattenedCodebooks
    )
}
```

**4b. Replace `flattenCodebooks()` calls**

The private `flattenCodebooks(from:)` method on `IVFPQIndex` (lines 540-551) can either
be removed entirely (delegating all calls to `GPUADCSearch.flattenCodebooks(from:)`) or
kept as a thin forwarding wrapper. The simplest approach:

- In `train()` at line 95: change `flattenCodebooks(from: trainedPQ)` → `GPUADCSearch.flattenCodebooks(from: trainedPQ)`
- In `restore(from:)` at line 561: same replacement
- Delete the private `flattenCodebooks(from:)` method from `IVFPQIndex`

**4c. Do NOT change `distancesForCluster()`**

The dispatch logic in `distancesForCluster()` (lines 345-391) stays unchanged. It still
calls `gpuADCDistances()` / `cpuADCDistances()` as before — only the body of
`gpuADCDistances()` changed internally.

**4d. Do NOT change `cpuADCDistances()`**

The CPU path (lines 393-418) is independent and remains as-is.

---

### Step 5: Write `GPUADCSearchTests.swift`

**File**: `Tests/MetalANNSTests/GPUADCSearchTests.swift`

```swift
import Testing
import Foundation
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("GPU ADC Search Tests")
struct GPUADCSearchTests {

    // TEST 1: gpuDistancesMatchCPU
    // Train PQ on 500 vectors (dim=64, M=8). Encode 200 database vectors.
    // Run GPUADCSearch.computeDistances() and compare against
    // ProductQuantizer.approximateDistance() for each code.
    // Assert all distances match within 1e-3.
    @Test func gpuDistancesMatchCPU() async throws { ... }

    // TEST 2: searchReturnsTopK
    // Train PQ, encode 300 vectors.
    // Run GPUADCSearch.search(k: 10). Assert result count == 10.
    // Assert results are sorted by ascending score.
    // Assert the top-1 result matches a brute-force CPU ADC scan.
    @Test func searchReturnsTopK() async throws { ... }

    // TEST 3: searchKLargerThanCorpus
    // Train PQ, encode 5 vectors.
    // Run GPUADCSearch.search(k: 100). Assert result count == 5.
    @Test func searchKLargerThanCorpus() async throws { ... }

    // TEST 4: emptyCodesReturnsEmpty
    // Run GPUADCSearch.computeDistances() with empty codes array.
    // Assert returns empty array (no crash, no GPU dispatch).
    @Test func emptyCodesReturnsEmpty() async throws { ... }

    // TEST 5: flattenCodebooksCorrectLayout
    // Train a small PQ (M=4, dim=16).
    // Call GPUADCSearch.flattenCodebooks(from:).
    // Assert length == M * 256 * subspaceDim.
    // Assert first subspaceDim floats match pq.codebooks[0][0].
    @Test func flattenCodebooksCorrectLayout() async throws { ... }

    // TEST 6: ivfpqRegressionAfterRewire
    // Full IVFPQIndex round-trip: train, add, search.
    // Run search with forceGPU: true and forceGPU: false.
    // Assert GPU and CPU top-10 results have the same IDs (order may differ slightly).
    // This confirms the IVFPQIndex rewire didn't break anything.
    @Test func ivfpqRegressionAfterRewire() async throws { ... }

    // TEST 7: cachedFlatCodebooksSkipsRecomputation
    // Pre-compute flatCodebooks via GPUADCSearch.flattenCodebooks(from:).
    // Pass it to computeDistances(flatCodebooks:).
    // Compare result against calling with flatCodebooks: nil.
    // Assert identical distances.
    @Test func cachedFlatCodebooksSkipsRecomputation() async throws { ... }
}
```

**Private test helpers:**

```swift
private func trainPQ(dim: Int = 64, M: Int = 8) throws -> ProductQuantizer {
    let vectors = makeRandomVectors(count: 500, dim: dim, seed: 301)
    return try ProductQuantizer.train(
        vectors: vectors,
        numSubspaces: M,
        centroidsPerSubspace: 256,
        maxIterations: 6
    )
}

private func makeRandomVectors(count: Int, dim: Int, seed: UInt64) -> [[Float]] {
    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    return (0..<count).map { _ in
        (0..<dim).map { _ in Float.random(in: -1.0...1.0, using: &rng) }
    }
}

private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
```

**All GPU tests must start with the guard:**
```swift
#if targetEnvironment(simulator)
return
#else
guard let context = try? MetalContext() else { return }
// ... test body ...
#endif
```

---

### Step 6: Verify No Regressions

```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30
```

Critical checks:
- All `IVFPQGPUTests` must still pass — `gpuVsCpuDistances` validates GPU/CPU match within 1e-3
- All `IVFPQTests` (if any) must pass — search, add, save/load
- `GPUADCSearchTests` — all 7 new tests pass
- No other test regressions

---

### Definition of Done

- [ ] `GPUADCSearch` enum exists in `Sources/MetalANNSCore/GPUADCSearch.swift`
- [ ] `computeDistances()` returns `[Float]` with GPU ADC distances matching CPU within 1e-3
- [ ] `search()` returns sorted top-k `[SearchResult]` with correct IDs and scores
- [ ] `flattenCodebooks(from:)` is a public static method on `GPUADCSearch`
- [ ] `IVFPQIndex.gpuADCDistances()` body replaced with `GPUADCSearch.computeDistances()` delegation
- [ ] `IVFPQIndex.flattenCodebooks(from:)` private method deleted; callers use `GPUADCSearch.flattenCodebooks(from:)`
- [ ] `IVFPQIndex.distancesForCluster()` and `cpuADCDistances()` are UNCHANGED
- [ ] All 7 new tests pass
- [ ] All pre-existing `IVFPQGPUTests` and `IVFPQTests` pass — zero regressions
- [ ] `GPUADCSearch` lives in MetalANNSCore (not MetalANNS) since it depends only on Metal types

---

### What Not To Do

- Do not rewrite the Metal shaders — `PQDistance.metal` is correct and tested; do not touch it
- Do not change `pq_adc_scan`'s threadgroup memory strategy — it loads the full M×Ks table cooperatively (stride-32) which is correct for M ≤ 64
- Do not change `cpuADCDistances()` in `IVFPQIndex` — it remains the CPU fallback
- Do not change `distancesForCluster()` dispatch logic — it correctly selects GPU when `candidateIDs.count >= 64`
- Do not add a `MetalContext` or device property to `GPUADCSearch` — it is a stateless enum; the caller passes the context
- Do not add graph traversal to `GPUADCSearch` — it is a flat linear scan only; graph-based search is a separate concern
- Do not change `IVFPQIndex`'s public API — `search(query:k:nprobe:)` signature stays the same
- Do not move `PQVectorBuffer.code(at:)` — `IVFPQIndex` extracts codes before calling `GPUADCSearch`
- Do not use `context.executeOnPool()` — use `context.execute()` for single-query dispatch (pool is for concurrent queue pipelining)
- Do not add `@testable import` for MetalANNSCore in tests that only need public API — but `GPUADCSearch` lives in MetalANNSCore, so `@testable import MetalANNSCore` is needed for tests
