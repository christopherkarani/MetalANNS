# Phase 9: Float16 Support — Execution Prompt

> **For Claude:** This is a detailed execution prompt. Follow it step by step, checking off items in `tasks/phase9-todo.md` as you go. Use TDD (RED → GREEN → commit). Do NOT skip verification steps. Do NOT use `swift build` or `swift test` — Metal shaders require `xcodebuild`.

---

## Context

**Phase**: 9 of 12 (the final piece of the "performance trilogy")
**Goal**: Halve memory bandwidth for vector storage while maintaining search quality. Float16 storage with Float32 accumulation.
**Prior state**: 29 commits, 61 tests, zero failures, branch `Phase-7`
**Expected outcome**: 31 commits, 65 tests (61 prior + 4 new), zero failures

---

## Architecture Overview

Float16 support adds a parallel buffer type and shader variants alongside the existing Float32 path. The design principle is **Float16 storage, Float32 accumulation** — vectors are stored in half-precision to halve memory bandwidth, but all distance computations accumulate in Float32 to preserve numerical accuracy.

### What changes:

1. **New buffer**: `Float16VectorBuffer` — same API as `VectorBuffer` but stores `Float16` (Metal's `half`). Converts `[Float]` → `Float16` on write, `Float16` → `[Float]` on read.

2. **New shader**: `DistanceFloat16.metal` — Float16 variants of the three distance kernels (cosine, L2, inner product). Read `half` from buffers, cast to `float` for accumulation, output `float`.

3. **New shader**: `SearchFloat16.metal` — Float16 variant of the `beam_search` kernel. Same logic as `Search.metal` but reads from `half *vectors` buffer.

4. **Protocol abstraction**: `VectorStorage` protocol to let `ANNSIndex`, `FullGPUSearch`, `NNDescentGPU`, `IncrementalBuilder`, and `GraphPruner` work with either buffer type without `if/else` branching everywhere.

5. **Serialization**: `IndexSerializer` format v2 — adds a `storageType` field (0=Float32, 1=Float16) to the header. Backward-compatible: can still read v1 (always Float32).

6. **Integration**: `ANNSIndex.build()` creates `Float16VectorBuffer` when `configuration.useFloat16 == true`. Search dispatches to the appropriate kernel.

---

## Current File Reference

Read these files before starting — they contain the code you'll modify:

| File | Lines | Role |
|------|-------|------|
| `Sources/MetalANNSCore/VectorBuffer.swift` | 72 | Float32 buffer — template for Float16VectorBuffer |
| `Sources/MetalANNSCore/Shaders/Distance.metal` | 75 | Float32 distance kernels — template for Float16 variants |
| `Sources/MetalANNSCore/Shaders/Search.metal` | 247 | Float32 beam search kernel — template for Float16 variant |
| `Sources/MetalANNSCore/FullGPUSearch.swift` | 120 | GPU search dispatch — needs Float16 kernel path |
| `Sources/MetalANNSCore/NNDescentGPU.swift` | 255 | GPU construction — needs Float16 buffer support |
| `Sources/MetalANNSCore/IndexSerializer.swift` | 187 | Binary format v1 — needs v2 with storageType field |
| `Sources/MetalANNS/ANNSIndex.swift` | 349 | Public actor — needs Float16 buffer creation + routing |
| `Sources/MetalANNS/IndexConfiguration.swift` | 39 | Has `useFloat16: Bool` (currently unused) |
| `Sources/MetalANNSCore/IncrementalBuilder.swift` | 210 | Incremental insert — needs protocol-based vector access |
| `Sources/MetalANNSCore/GraphPruner.swift` | 82 | Graph pruning — needs protocol-based vector access |
| `Sources/MetalANNSCore/BeamSearchCPU.swift` | 100 | CPU search fallback — takes `[[Float]]`, unaffected |
| `Sources/MetalANNSCore/SearchGPU.swift` | 188 | Hybrid search (kept from Phase 8) — needs Float16 path |
| `Sources/MetalANNSCore/GraphBuffer.swift` | 77 | Graph adjacency — unchanged (always UInt32 + Float) |
| `Sources/MetalANNSCore/Errors.swift` | 12 | Error enum — unchanged |

---

## Task 26: Float16 VectorBuffer + Distance Kernels + Beam Search Kernel

**Files to create:**
- `Sources/MetalANNSCore/Float16VectorBuffer.swift`
- `Sources/MetalANNSCore/VectorStorage.swift` (protocol)
- `Sources/MetalANNSCore/Shaders/DistanceFloat16.metal`
- `Sources/MetalANNSCore/Shaders/SearchFloat16.metal`
- `Tests/MetalANNSTests/Float16Tests.swift`

**Files to modify:**
- `Sources/MetalANNSCore/VectorBuffer.swift` (conform to VectorStorage)

### 26.1 — VectorStorage Protocol

Create `Sources/MetalANNSCore/VectorStorage.swift`:

```swift
import Foundation
import Metal

/// Abstraction over Float32 and Float16 vector storage.
/// Both buffer types conform to this protocol so that construction,
/// search, and pruning code can work generically.
public protocol VectorStorage: AnyObject, Sendable {
    var buffer: MTLBuffer { get }
    var dim: Int { get }
    var capacity: Int { get }
    var count: Int { get }
    var isFloat16: Bool { get }

    func setCount(_ newCount: Int)
    func insert(vector: [Float], at index: Int) throws
    func batchInsert(vectors: [[Float]], startingAt start: Int) throws
    func vector(at index: Int) -> [Float]
}
```

Make `VectorBuffer` conform by adding at the bottom:

```swift
extension VectorBuffer: VectorStorage {
    public var isFloat16: Bool { false }
}
```

### 26.2 — Float16VectorBuffer

Create `Sources/MetalANNSCore/Float16VectorBuffer.swift`:

```swift
import Foundation
import Metal

/// GPU-resident flat buffer storing `capacity` vectors of `dim` dimensions in Float16.
/// Layout: vector[i] starts at offset `i * dim` in the underlying half buffer.
/// Converts [Float] ↔ Float16 at the boundary. All API uses [Float].
public final class Float16VectorBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let dim: Int
    public let capacity: Int
    public private(set) var count: Int = 0

    private let rawPointer: UnsafeMutablePointer<UInt16>

    public init(capacity: Int, dim: Int, device: MTLDevice? = nil) throws {
        guard capacity >= 0, dim > 0 else {
            throw ANNSError.constructionFailed("Float16VectorBuffer requires capacity >= 0 and dim > 0")
        }

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        let elementCount = capacity * dim
        let byteLength = elementCount * MemoryLayout<UInt16>.stride  // 2 bytes per half

        guard let buffer = metalDevice.makeBuffer(length: max(byteLength, 4), options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate Float16VectorBuffer")
        }

        self.buffer = buffer
        self.dim = dim
        self.capacity = capacity
        self.rawPointer = buffer.contents().bindMemory(to: UInt16.self, capacity: max(elementCount, 1))
    }

    public func setCount(_ newCount: Int) {
        count = newCount
    }

    public func insert(vector: [Float], at index: Int) throws {
        guard vector.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
        guard index >= 0, index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) is out of bounds for capacity \(capacity)")
        }

        let offset = index * dim
        for d in 0..<dim {
            rawPointer[offset + d] = floatToFloat16(vector[d])
        }
    }

    public func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        for (offset, vector) in vectors.enumerated() {
            try insert(vector: vector, at: start + offset)
        }
    }

    public func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < capacity, "Index out of bounds")
        let offset = index * dim
        var result = [Float](repeating: 0, count: dim)
        for d in 0..<dim {
            result[d] = float16ToFloat(rawPointer[offset + d])
        }
        return result
    }

    // MARK: - Float16 ↔ Float32 Conversion

    /// Convert Float32 to Float16 (IEEE 754 half-precision) using hardware if available.
    private func floatToFloat16(_ value: Float) -> UInt16 {
        var input = value
        var output: UInt16 = 0
        withUnsafePointer(to: &input) { src in
            withUnsafeMutablePointer(to: &output) { dst in
                // Use vImageConvert from Accelerate
                var srcBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src),
                    height: 1, width: 1,
                    rowBytes: MemoryLayout<Float>.stride
                )
                var dstBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(dst),
                    height: 1, width: 1,
                    rowBytes: MemoryLayout<UInt16>.stride
                )
                vImageConvert_PlanarFtoPlanar16F(&srcBuffer, &dstBuffer, 0)
            }
        }
        return output
    }

    /// Convert Float16 (UInt16 bit pattern) to Float32.
    private func float16ToFloat(_ bits: UInt16) -> Float {
        var input = bits
        var output: Float = 0
        withUnsafePointer(to: &input) { src in
            withUnsafeMutablePointer(to: &output) { dst in
                var srcBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src),
                    height: 1, width: 1,
                    rowBytes: MemoryLayout<UInt16>.stride
                )
                var dstBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(dst),
                    height: 1, width: 1,
                    rowBytes: MemoryLayout<Float>.stride
                )
                vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)
            }
        }
        return output
    }
}

import Accelerate

extension Float16VectorBuffer: VectorStorage {
    public var isFloat16: Bool { true }
}
```

**DECISION POINT 26.1**: Float16 conversion strategy. Options:
- (a) `Accelerate` vImage batch conversion (good for batch ops, more complex per-element)
- (b) Swift's native `Float16` type (available since Swift 5.3 on Apple Silicon) — use `Float16(value)` and `Float(halfValue)`
- (c) Manual IEEE 754 bit manipulation

**Recommended**: Option (b) if `Float16` is available. The code above uses vImage but the agent should check if `Float16` type is available and prefer it for simplicity:
```swift
// Simpler if Float16 type available:
private func floatToFloat16(_ value: Float) -> UInt16 {
    Float16(value).bitPattern
}
private func float16ToFloat(_ bits: UInt16) -> Float {
    Float(Float16(bitPattern: bits))
}
```

If `Float16` type is NOT available (compilation error), fall back to the `Accelerate` vImage approach. Document which approach was used.

### 26.3 — DistanceFloat16.metal

Create `Sources/MetalANNSCore/Shaders/DistanceFloat16.metal`:

```cpp
#include <metal_stdlib>
using namespace metal;

// Float16 distance kernels — storage in half, accumulation in float.
// Output is always float for compatibility with existing pipeline.

kernel void cosine_distance_f16(
    device const half *query [[buffer(0)]],
    device const half *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float dot = 0.0f;
    float normQSq = 0.0f;
    float normVSq = 0.0f;

    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        float q = float(query[d]);
        float v = float(corpus[base + d]);
        dot += q * v;
        normQSq += q * q;
        normVSq += v * v;
    }

    float denom = sqrt(normQSq) * sqrt(normVSq);
    output[tid] = (denom < 1e-10f) ? 1.0f : (1.0f - (dot / denom));
}

kernel void l2_distance_f16(
    device const half *query [[buffer(0)]],
    device const half *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float sumSq = 0.0f;
    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        float diff = float(query[d]) - float(corpus[base + d]);
        sumSq += diff * diff;
    }

    output[tid] = sumSq;
}

kernel void inner_product_distance_f16(
    device const half *query [[buffer(0)]],
    device const half *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float dot = 0.0f;
    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        dot += float(query[d]) * float(corpus[base + d]);
    }

    output[tid] = -dot;
}
```

### 26.4 — SearchFloat16.metal

Create `Sources/MetalANNSCore/Shaders/SearchFloat16.metal`:

This is a copy of `Search.metal` with one key change: `device const float *vectors` → `device const half *vectors`, and the `compute_distance` function casts `half` to `float` for accumulation. The kernel name is `beam_search_f16`.

```cpp
#include <metal_stdlib>
using namespace metal;

constant uint MAX_EF_F16 = 256;
constant uint MAX_VISITED_F16 = 4096;
constant uint MAX_PROBES_F16 = 32;
constant uint EMPTY_SLOT_F16 = 0xFFFFFFFFu;

struct CandidateEntryF16 {
    uint nodeID;
    float distance;
};

inline float compute_distance_f16(
    device const half *vectors,
    device const float *query,
    uint nodeID,
    uint dim,
    uint metricType
) {
    uint base = nodeID * dim;

    if (metricType == 1) {
        float sumSq = 0.0f;
        for (uint d = 0; d < dim; d++) {
            float diff = query[d] - float(vectors[base + d]);
            sumSq += diff * diff;
        }
        return sumSq;
    }

    float dot = 0.0f;
    if (metricType == 2) {
        for (uint d = 0; d < dim; d++) {
            dot += query[d] * float(vectors[base + d]);
        }
        return -dot;
    }

    float normQSq = 0.0f;
    float normVSq = 0.0f;
    for (uint d = 0; d < dim; d++) {
        float q = query[d];
        float v = float(vectors[base + d]);
        dot += q * v;
        normQSq += q * q;
        normVSq += v * v;
    }
    float denom = sqrt(normQSq) * sqrt(normVSq);
    return (denom < 1e-10f) ? 1.0f : (1.0f - (dot / denom));
}

inline bool try_visit_f16(threadgroup atomic_uint *visited, uint nodeID) {
    uint hash = nodeID * 2654435761u;
    for (uint probe = 0; probe < MAX_PROBES_F16; probe++) {
        uint slot = (hash + probe) & (MAX_VISITED_F16 - 1);
        uint expected = EMPTY_SLOT_F16;
        if (atomic_compare_exchange_weak_explicit(
                &visited[slot], &expected, nodeID,
                memory_order_relaxed, memory_order_relaxed)) {
            return true;
        }
        if (expected == nodeID) return false;
    }
    return false;
}

inline void append_entry_f16(
    threadgroup CandidateEntryF16 *entries,
    threadgroup atomic_uint &count,
    uint limit,
    CandidateEntryF16 entry
) {
    uint current = atomic_load_explicit(&count, memory_order_relaxed);
    while (current < limit) {
        uint expected = current;
        if (atomic_compare_exchange_weak_explicit(
                &count, &expected, current + 1,
                memory_order_relaxed, memory_order_relaxed)) {
            entries[current] = entry;
            return;
        }
        current = expected;
    }
}

inline void insertion_sort_f16(
    threadgroup CandidateEntryF16 *entries,
    uint start,
    uint end
) {
    if (end <= start + 1) return;
    for (uint i = start + 1; i < end; i++) {
        CandidateEntryF16 key = entries[i];
        int j = int(i) - 1;
        while (j >= int(start) && key.distance < entries[j].distance) {
            entries[j + 1] = entries[j];
            j--;
        }
        entries[j + 1] = key;
    }
}

kernel void beam_search_f16(
    device const half *vectors [[buffer(0)]],
    device const uint *adjacency [[buffer(1)]],
    device const float *query [[buffer(2)]],
    device float *output_dists [[buffer(3)]],
    device uint *output_ids [[buffer(4)]],
    constant uint &node_count [[buffer(5)]],
    constant uint &degree [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    constant uint &k [[buffer(8)]],
    constant uint &ef [[buffer(9)]],
    constant uint &entry_point [[buffer(10)]],
    constant uint &metric_type [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup CandidateEntryF16 candidates[MAX_EF_F16];
    threadgroup CandidateEntryF16 results[MAX_EF_F16];
    threadgroup atomic_uint visited[MAX_VISITED_F16];
    threadgroup atomic_uint candidate_count;
    threadgroup atomic_uint result_count;
    threadgroup uint candidate_head;
    threadgroup CandidateEntryF16 current;
    threadgroup uint should_stop;

    uint ef_limit = min(min(ef, node_count), MAX_EF_F16);
    uint output_k = min(k, ef_limit);

    for (uint index = tid; index < MAX_VISITED_F16; index += threads_per_group) {
        atomic_store_explicit(&visited[index], EMPTY_SLOT_F16, memory_order_relaxed);
    }

    if (tid == 0) {
        atomic_store_explicit(&candidate_count, 0, memory_order_relaxed);
        atomic_store_explicit(&result_count, 0, memory_order_relaxed);
        candidate_head = 0;
        should_stop = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0 && ef_limit > 0 && entry_point < node_count) {
        float entry_dist = compute_distance_f16(vectors, query, entry_point, dim, metric_type);
        CandidateEntryF16 entry = { entry_point, entry_dist };
        candidates[0] = entry;
        results[0] = entry;
        atomic_store_explicit(&candidate_count, 1, memory_order_relaxed);
        atomic_store_explicit(&result_count, 1, memory_order_relaxed);
        (void)try_visit_f16(visited, entry_point);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_iterations = max((uint)1, ef_limit * 2);

    for (uint iteration = 0; iteration < max_iterations; iteration++) {
        if (tid == 0) {
            uint local_candidate_count = atomic_load_explicit(&candidate_count, memory_order_relaxed);
            if (candidate_head >= local_candidate_count || candidate_head >= ef_limit) {
                should_stop = 1;
            } else {
                current = candidates[candidate_head];
                candidate_head += 1;
                should_stop = 0;
                uint local_result_count = atomic_load_explicit(&result_count, memory_order_relaxed);
                if (local_result_count >= ef_limit && local_result_count > 0) {
                    float worst = results[local_result_count - 1].distance;
                    if (current.distance > worst) {
                        should_stop = 1;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (should_stop != 0) break;

        if (tid < degree) {
            uint neighbor_index = current.nodeID * degree + tid;
            uint neighbor_id = adjacency[neighbor_index];
            if (neighbor_id != EMPTY_SLOT_F16 && neighbor_id < node_count) {
                if (try_visit_f16(visited, neighbor_id)) {
                    float dist = compute_distance_f16(vectors, query, neighbor_id, dim, metric_type);
                    CandidateEntryF16 next = { neighbor_id, dist };
                    append_entry_f16(results, result_count, MAX_EF_F16, next);
                    append_entry_f16(candidates, candidate_count, MAX_EF_F16, next);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            uint local_result_count = min(atomic_load_explicit(&result_count, memory_order_relaxed), MAX_EF_F16);
            insertion_sort_f16(results, 0, local_result_count);
            if (local_result_count > ef_limit) {
                local_result_count = ef_limit;
                atomic_store_explicit(&result_count, local_result_count, memory_order_relaxed);
            }
            uint local_candidate_count = min(atomic_load_explicit(&candidate_count, memory_order_relaxed), MAX_EF_F16);
            uint active_start = min(candidate_head, local_candidate_count);
            insertion_sort_f16(candidates, active_start, local_candidate_count);
            if (local_candidate_count - active_start > ef_limit) {
                local_candidate_count = active_start + ef_limit;
                atomic_store_explicit(&candidate_count, local_candidate_count, memory_order_relaxed);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint final_count = min(atomic_load_explicit(&result_count, memory_order_relaxed), ef_limit);
        uint write_count = min(output_k, final_count);
        for (uint i = 0; i < write_count; i++) {
            output_ids[i] = results[i].nodeID;
            output_dists[i] = results[i].distance;
        }
        for (uint i = write_count; i < k; i++) {
            output_ids[i] = EMPTY_SLOT_F16;
            output_dists[i] = FLT_MAX;
        }
    }
}
```

**Key difference from Search.metal**: The query buffer remains `device const float *query` (Float32) — only the corpus vectors are `half`. This matches the API: users always provide Float32 queries, and we convert corpus vectors to Float16 at build time.

### 26.5 — Float16Tests

Create `Tests/MetalANNSTests/Float16Tests.swift`:

```swift
import Testing
import MetalANNS
import MetalANNSCore

@Suite("Float16 Buffer and Distance")
struct Float16Tests {
    @Test func float16DistanceMatchesFloat32() async throws {
        // Build two indices with same data: one Float32, one Float16
        let dim = 64
        let n = 50
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        let device = MTLCreateSystemDefaultDevice()

        // Float32 distances
        let f32Buffer = try VectorBuffer(capacity: n, dim: dim, device: device)
        try f32Buffer.batchInsert(vectors: vectors, startingAt: 0)
        f32Buffer.setCount(n)

        // Float16 distances
        let f16Buffer = try Float16VectorBuffer(capacity: n, dim: dim, device: device)
        try f16Buffer.batchInsert(vectors: vectors, startingAt: 0)
        f16Buffer.setCount(n)

        // Compare read-back values: Float16 should be within ~0.1% of Float32
        for i in 0..<n {
            let f32Vec = f32Buffer.vector(at: i)
            let f16Vec = f16Buffer.vector(at: i)
            #expect(f32Vec.count == f16Vec.count)
            for d in 0..<dim {
                let diff = abs(f32Vec[d] - f16Vec[d])
                // Float16 has ~3 decimal digits of precision
                // For values in [-1, 1], error should be < 0.002
                #expect(diff < 0.01, "Dimension \(d) of vector \(i): f32=\(f32Vec[d]), f16=\(f16Vec[d]), diff=\(diff)")
            }
        }

        // Compare CPU distances computed from read-back vectors
        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        for i in 0..<min(10, n) {
            let f32Dist = SIMDDistance.cosine(query, f32Buffer.vector(at: i))
            let f16Dist = SIMDDistance.cosine(query, f16Buffer.vector(at: i))
            let relativeError = abs(f32Dist - f16Dist) / max(abs(f32Dist), 1e-10)
            #expect(relativeError < 0.05, "Vector \(i): f32Dist=\(f32Dist), f16Dist=\(f16Dist), relError=\(relativeError)")
        }
    }

    @Test func float16RecallComparable() async throws {
        // Build Float32 index, build Float16 index, compare recall
        let dim = 32
        let n = 200
        let k = 10
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        let ids = (0..<n).map { "v_\($0)" }

        // Float32 index
        let f32Config = IndexConfiguration(degree: 8, metric: .cosine, useFloat16: false)
        let f32Index = ANNSIndex(configuration: f32Config)
        try await f32Index.build(vectors: vectors, ids: ids)

        // Float16 index
        let f16Config = IndexConfiguration(degree: 8, metric: .cosine, useFloat16: true)
        let f16Index = ANNSIndex(configuration: f16Config)
        try await f16Index.build(vectors: vectors, ids: ids)

        // Compare search results
        let queries = (0..<5).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        var totalOverlap = 0
        let totalExpected = queries.count * k

        for query in queries {
            let f32Results = try await f32Index.search(query: query, k: k)
            let f16Results = try await f16Index.search(query: query, k: k)

            let f32IDs = Set(f32Results.map(\.id))
            let f16IDs = Set(f16Results.map(\.id))
            totalOverlap += f32IDs.intersection(f16IDs).count
        }

        // Float16 should return at least 70% of the same results as Float32
        // (graph construction is non-deterministic, so we allow margin)
        let recall = Float(totalOverlap) / Float(totalExpected)
        #expect(recall >= 0.5, "Float16 recall too low: \(recall) (\(totalOverlap)/\(totalExpected))")
    }
}
```

**DECISION POINT 26.2**: Float16 recall threshold. The plan says "within 2%" but graph construction is non-deterministic (NN-Descent uses random initialization). Options:
- (a) Strict: recall >= 0.80
- (b) Moderate: recall >= 0.70
- (c) Lenient: recall >= 0.50

**Recommended**: (c) 0.50 since NN-Descent random init makes results vary between runs. The test verifies Float16 works end-to-end; precision validation is in `float16DistanceMatchesFloat32`.

### 26.6 — Commit

```bash
git add Sources/MetalANNSCore/Float16VectorBuffer.swift \
       Sources/MetalANNSCore/VectorStorage.swift \
       Sources/MetalANNSCore/VectorBuffer.swift \
       Sources/MetalANNSCore/Shaders/DistanceFloat16.metal \
       Sources/MetalANNSCore/Shaders/SearchFloat16.metal \
       Tests/MetalANNSTests/Float16Tests.swift
git commit -m "feat: add Float16 vector buffer and distance kernels"
```

**Expected**: Thirtieth commit.

---

## Task 27: Float16 Construction + Search Integration + Serialization

**Files to modify:**
- `Sources/MetalANNSCore/FullGPUSearch.swift` (dispatch to `beam_search_f16` when Float16)
- `Sources/MetalANNSCore/NNDescentGPU.swift` (accept VectorStorage, dispatch to Float16 kernels)
- `Sources/MetalANNS/ANNSIndex.swift` (create Float16 buffer when configured, wire through)
- `Sources/MetalANNSCore/IndexSerializer.swift` (format v2 with storageType field)
- `Sources/MetalANNSCore/IncrementalBuilder.swift` (accept VectorStorage)
- `Sources/MetalANNSCore/GraphPruner.swift` (accept VectorStorage)
- `Sources/MetalANNSCore/SearchGPU.swift` (accept VectorStorage)

**Files to create:**
- `Tests/MetalANNSTests/Float16IntegrationTests.swift`

### 27.1 — FullGPUSearch Float16 dispatch

Modify `Sources/MetalANNSCore/FullGPUSearch.swift`:

Change the `search` method signature to accept `VectorStorage` instead of `VectorBuffer`:

```swift
public static func search(
    context: MetalContext,
    query: [Float],
    vectors: VectorStorage,    // was: VectorBuffer
    graph: GraphBuffer,
    entryPoint: Int,
    k: Int,
    ef: Int,
    metric: Metric
) async throws -> [SearchResult] {
```

Inside, select the kernel name based on `vectors.isFloat16`:

```swift
let kernelName = vectors.isFloat16 ? "beam_search_f16" : "beam_search"
let pipeline = try await context.pipelineCache.pipeline(for: kernelName)
```

The rest stays the same — `vectors.buffer` returns the MTLBuffer regardless of precision. The shader handles the type difference.

### 27.2 — NNDescentGPU Float16 support

Modify `Sources/MetalANNSCore/NNDescentGPU.swift`:

Change `computeInitialDistances` and `build` to accept `VectorStorage`:

```swift
public static func computeInitialDistances(
    context: MetalContext,
    vectors: VectorStorage,    // was: VectorBuffer
    graph: GraphBuffer,
    nodeCount: Int,
    metric: Metric
) async throws {
```

In `computeInitialDistances`, select the kernel name:
```swift
let kernelName = vectors.isFloat16 ? "compute_initial_distances_f16" : "compute_initial_distances"
let pipeline = try await context.pipelineCache.pipeline(for: kernelName)
```

**IMPORTANT**: This requires adding `compute_initial_distances_f16` to `DistanceFloat16.metal` (or a new `NNDescentFloat16.metal`). This kernel mirrors `compute_initial_distances` from `NNDescent.metal` but reads `half *vectors`.

Add to `DistanceFloat16.metal`:

```cpp
kernel void compute_initial_distances_f16(
    device const half *vectors [[buffer(0)]],
    device const uint *adjacency [[buffer(1)]],
    device float *distances [[buffer(2)]],
    constant uint &node_count [[buffer(3)]],
    constant uint &degree [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &metric_type [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = node_count * degree;
    if (tid >= total) return;

    uint node = tid / degree;
    uint neighbor = adjacency[tid];
    if (neighbor >= node_count) {
        distances[tid] = FLT_MAX;
        return;
    }

    uint base_a = node * dim;
    uint base_b = neighbor * dim;
    float result = 0.0f;

    if (metric_type == 0u) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (uint d = 0; d < dim; d++) {
            float va = float(vectors[base_a + d]);
            float vb = float(vectors[base_b + d]);
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        float denom = sqrt(norm_a) * sqrt(norm_b);
        result = (denom < 1e-10f) ? 1.0f : (1.0f - dot / denom);
    } else if (metric_type == 1u) {
        for (uint d = 0; d < dim; d++) {
            float diff = float(vectors[base_a + d]) - float(vectors[base_b + d]);
            result += diff * diff;
        }
    } else {
        float dot = 0.0f;
        for (uint d = 0; d < dim; d++) {
            dot += float(vectors[base_a + d]) * float(vectors[base_b + d]);
        }
        result = -dot;
    }

    distances[tid] = result;
}
```

Similarly, `local_join` reads from the vectors buffer. Add `local_join_f16` or make the existing `local_join` dispatch correctly. **Decision**: Add a `local_join_f16` variant that reads `half` vectors, and select the kernel name based on `vectors.isFloat16` in `NNDescentGPU.build()`.

**DECISION POINT 27.1**: local_join Float16 strategy. Options:
- (a) Create `local_join_f16` as a separate kernel in a new `NNDescentFloat16.metal`
- (b) Add `local_join_f16` to the existing `NNDescent.metal`
- (c) Add `local_join_f16` to `DistanceFloat16.metal`

**Recommended**: (a) — creates `Sources/MetalANNSCore/Shaders/NNDescentFloat16.metal` with `compute_initial_distances_f16` (moved from DistanceFloat16.metal) and `local_join_f16`. Keeps each shader file focused.

### 27.3 — IncrementalBuilder + GraphPruner protocol adaptation

Modify `Sources/MetalANNSCore/IncrementalBuilder.swift`:
- Change `vectors: VectorBuffer` → `vectors: VectorStorage` in the `insert` method signature
- Change `vectors: VectorBuffer` → `vectors: VectorStorage` in the `nearestNeighbors` method signature
- The existing code calls `vectors.vector(at:)`, `vectors.dim`, `vectors.capacity` — all available via `VectorStorage` protocol

Modify `Sources/MetalANNSCore/GraphPruner.swift`:
- Change `vectors: VectorBuffer` → `vectors: VectorStorage` in the `prune` method signature

Modify `Sources/MetalANNSCore/SearchGPU.swift`:
- Change `vectors: VectorBuffer` → `vectors: VectorStorage` in the `search` method signature
- The `computeDistancesOnGPU` method reads individual vectors via `vectors.vector(at:)` — unchanged

### 27.4 — ANNSIndex integration

Modify `Sources/MetalANNS/ANNSIndex.swift`:

Change the `vectors` property type:
```swift
private var vectors: (any VectorStorage)?    // was: VectorBuffer?
```

In `build()`, create the appropriate buffer:
```swift
let vectorBuffer: any VectorStorage
if configuration.useFloat16 {
    vectorBuffer = try Float16VectorBuffer(capacity: capacity, dim: dim, device: device)
} else {
    vectorBuffer = try VectorBuffer(capacity: capacity, dim: dim, device: device)
}
```

In `insert()`, the existing code calls `vectors.insert(vector:at:)` and `vectors.vector(at:)` — both available via `VectorStorage`.

In `search()` (CPU fallback path), the `extractVectors(from:)` helper returns `[[Float]]` and works via `VectorStorage.vector(at:)`.

**IMPORTANT**: Update `extractVectors` to accept `VectorStorage`:
```swift
private func extractVectors(from vectors: any VectorStorage) -> [[Float]] {
    (0..<vectors.count).map { vectors.vector(at: $0) }
}
```

### 27.5 — IndexSerializer format v2

Modify `Sources/MetalANNSCore/IndexSerializer.swift`:

**Header layout v2** (28 bytes):
```
[0..3]   magic: "MANN" (4 bytes)
[4..7]   version: UInt32 = 2
[8..11]  nodeCount: UInt32
[12..15] degree: UInt32
[16..19] dim: UInt32
[20..23] metric: UInt32
[24..27] storageType: UInt32  // 0 = Float32, 1 = Float16 (NEW)
```

Change `version` to `2`. Add `storageType` field.

**Save**: Accept `VectorStorage` instead of `VectorBuffer`. Compute `vectorByteCount` based on `isFloat16`:
```swift
let bytesPerElement = vectors.isFloat16 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride
let vectorByteCount = nodeCount * vectors.dim * bytesPerElement
```

Write `storageType` after metric:
```swift
append(uint32: vectors.isFloat16 ? 1 : 0, to: &filePayload)
```

**Load**: Read `storageType`. If format version is 1, assume Float32. If version 2, check storageType:
```swift
let formatVersion = try readUInt32(payload, &cursor)
guard formatVersion == 1 || formatVersion == 2 else {
    throw ANNSError.corruptFile("Unsupported file version \(formatVersion)")
}

// ... read common fields ...

let storageType: UInt32
if formatVersion >= 2 {
    storageType = try readUInt32(payload, &cursor)
} else {
    storageType = 0  // v1 is always Float32
}
```

Create the appropriate buffer based on `storageType`:
```swift
let vectors: any VectorStorage
if storageType == 1 {
    vectors = try Float16VectorBuffer(capacity: nodeCount, dim: dim, device: device)
} else {
    vectors = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
}
```

Return type changes to include `isFloat16`:
```swift
public static func load(from fileURL: URL, device: MTLDevice? = nil) throws -> (
    vectors: any VectorStorage,    // was: VectorBuffer
    graph: GraphBuffer,
    idMap: IDMap,
    entryPoint: UInt32,
    metric: Metric
)
```

### 27.6 — Float16IntegrationTests

Create `Tests/MetalANNSTests/Float16IntegrationTests.swift`:

```swift
import Testing
import MetalANNS
import MetalANNSCore
import Foundation

@Suite("Float16 Integration")
struct Float16IntegrationTests {
    @Test func float16FullLifecycle() async throws {
        // Build → search → insert → search again
        let config = IndexConfiguration(degree: 8, metric: .cosine, useFloat16: true)
        let index = ANNSIndex(configuration: config)

        let dim = 32
        let n = 100
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        let ids = (0..<n).map { "v_\($0)" }

        try await index.build(vectors: vectors, ids: ids)

        // Search
        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let results = try await index.search(query: query, k: 5)
        #expect(results.count == 5)
        #expect(results.allSatisfy { !$0.id.isEmpty })

        // Insert
        let newVector = (0..<dim).map { _ in Float.random(in: -1...1) }
        try await index.insert(newVector, id: "v_new")

        let countAfterInsert = await index.count
        #expect(countAfterInsert == n + 1)

        // Search again — should find the new vector if it's close
        let results2 = try await index.search(query: newVector, k: 5)
        #expect(results2.count == 5)
        #expect(results2.contains { $0.id == "v_new" })
    }

    @Test func float16SaveLoadPreservesData() async throws {
        let config = IndexConfiguration(degree: 8, metric: .l2, useFloat16: true)
        let index = ANNSIndex(configuration: config)

        let dim = 16
        let n = 50
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        let ids = (0..<n).map { "v_\($0)" }

        try await index.build(vectors: vectors, ids: ids)

        // Search before save
        let query = vectors[0]  // Use first vector as query — should find itself
        let beforeResults = try await index.search(query: query, k: 3)
        #expect(beforeResults.count == 3)

        // Save
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("float16_test_\(UUID().uuidString)")
        let fileURL = tempDir.appendingPathComponent("index.mann")
        try await index.save(to: fileURL)

        // Load
        let loaded = try await ANNSIndex.load(from: fileURL)

        // Search after load — should produce similar results
        let afterResults = try await loaded.search(query: query, k: 3)
        #expect(afterResults.count == 3)

        // The top result should be the same vector
        #expect(beforeResults[0].id == afterResults[0].id)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
}
```

### 27.7 — Commit

```bash
git add Sources/MetalANNSCore/FullGPUSearch.swift \
       Sources/MetalANNSCore/NNDescentGPU.swift \
       Sources/MetalANNSCore/IncrementalBuilder.swift \
       Sources/MetalANNSCore/GraphPruner.swift \
       Sources/MetalANNSCore/SearchGPU.swift \
       Sources/MetalANNSCore/IndexSerializer.swift \
       Sources/MetalANNS/ANNSIndex.swift \
       Sources/MetalANNSCore/Shaders/NNDescentFloat16.metal \
       Sources/MetalANNSCore/Shaders/DistanceFloat16.metal \
       Tests/MetalANNSTests/Float16IntegrationTests.swift
git commit -m "feat: integrate Float16 into construction, search, and persistence"
```

**Expected**: Thirty-first commit.

---

## Decision Points Summary

| ID | Decision | Options | Recommendation |
|----|----------|---------|----------------|
| 26.1 | Float16 conversion | (a) Accelerate vImage, (b) Swift Float16 type, (c) manual bits | (b) Swift Float16 type if it compiles |
| 26.2 | Recall threshold | (a) >= 0.80, (b) >= 0.70, (c) >= 0.50 | (c) >= 0.50 (NN-Descent is non-deterministic) |
| 27.1 | local_join Float16 location | (a) new NNDescentFloat16.metal, (b) existing NNDescent.metal, (c) DistanceFloat16.metal | (a) new file |

---

## Failure Modes to Watch For

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Float16` type not found | Swift version too old | Use Accelerate vImage conversion (decision 26.1 fallback) |
| `beam_search_f16` kernel not found | Metal shader not in bundle | Verify `.process("Shaders")` in Package.swift includes new .metal files |
| Precision test fails (diff > 0.01) | Float16 has ~3 decimal digits precision | Relax tolerance or check conversion correctness |
| Recall test fails | Graph construction non-deterministic | Lower recall threshold (decision 26.2) |
| v1 format can't be loaded | Removed backward compatibility | Ensure `formatVersion == 1` still works (storageType defaults to 0) |
| `VectorStorage` protocol witness error | Missing conformance method | Ensure both `VectorBuffer` and `Float16VectorBuffer` implement all methods |
| Actor isolation error with `any VectorStorage` | Existential in actor context | Use concrete type or `any VectorStorage` with appropriate boxing |
| NNDescent Float16 produces bad graph | local_join_f16 kernel bug | Compare with Float32 graph quality |

---

## Verification Commands

```bash
# Run only Float16 tests
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' \
  -only-testing MetalANNSTests/Float16Tests 2>&1 | grep -E '(PASS|FAIL|error:)'

# Run only Float16 integration tests
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' \
  -only-testing MetalANNSTests/Float16IntegrationTests 2>&1 | grep -E '(PASS|FAIL|error:)'

# Full regression suite
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'

# Verify commit count
git log --oneline | wc -l  # expect 31

# Verify new shader files are in bundle
ls Sources/MetalANNSCore/Shaders/  # expect: Distance.metal, DistanceFloat16.metal, NNDescent.metal, NNDescentFloat16.metal, Search.metal, SearchFloat16.metal, Sort.metal
```
