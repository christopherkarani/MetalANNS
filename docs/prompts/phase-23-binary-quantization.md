# Phase 23: Binary Quantization + Hamming Distance

### Mission

Add 1-bit-per-dimension vector storage (`BinaryVectorBuffer`) and a Hamming distance
metric. A binary index uses 32× less memory than Float32 and enables fast deduplication
and pre-filtering use-cases. The graph search path (NN-Descent + BeamSearch) works
unchanged via a packed → unpacked adapter. A Metal Hamming kernel is provided for
future flat-scan use.

---

### Verified Codebase Facts

Read each file before touching it.

| Fact | Source |
|------|--------|
| `Metric` enum has three cases: `.cosine`, `.l2`, `.innerProduct` | `MetalANNSCore/Metric.swift:1-5` |
| `MetalANNS/Metric.swift` is a typealias — no cases to add there | `MetalANNS/Metric.swift:1-4` |
| `VectorStorage` protocol requires: `buffer: MTLBuffer`, `dim`, `capacity`, `count`, `isFloat16`, `setCount`, `insert`, `batchInsert`, `vector(at:)` | `VectorStorage.swift:6-17` |
| `VectorBuffer.isFloat16` returns `false`; `Float16VectorBuffer.isFloat16` returns `true` | `VectorBuffer.swift:74`, `Float16VectorBuffer.swift` |
| `ANNSIndex.build()` dispatches on `configuration.useFloat16` to pick storage type | `ANNSIndex.swift:96-99` |
| `supportsGPUSearch(for:)` currently returns `!(vectors is DiskBackedVectorBuffer)` | `ANNSIndex.swift:958-960` |
| `SIMDDistance.distance(_:_:metric:)` switches on `Metric` — needs `.hamming` case added | `SIMDDistance.swift:82-91` |
| `IndexSerializer.metricCode(_:)` and `metric(from:)` handle codes 0/1/2 only | `IndexSerializer.swift:236-258` |
| `IndexSerializer.save()` sets `storageType: UInt32 = vectors.isFloat16 ? 1 : 0` | `IndexSerializer.swift:32` |
| `IndexSerializer.load()` rejects `storageType` values other than 0 or 1 | `IndexSerializer.swift:141-143` |
| Metal shaders live in `Sources/MetalANNSCore/Shaders/` — add new `.metal` files there | `Shaders/` directory |
| GPU build path (`NNDescentGPU`) reads `vectors.buffer` as packed Float32 — incompatible with binary | `ANNSIndex.swift:113-122` |
| `DistanceFloat16.metal` kernels use `half` typed buffers — Hamming is not applicable | `DistanceFloat16.metal:1-30` |

---

### TDD Implementation Order

Work strictly test-first. Do not write implementation code before the failing test exists.

**Round 1** — `BinaryVectorBuffer` unit tests (no `ANNSIndex`)
Write `BinaryQuantizationTests.swift` shell. Tests fail to compile. Implement `BinaryVectorBuffer` until they pass.

**Round 2** — `SIMDDistance.hamming` unit tests
Tests fail. Add `.hamming` to `Metric`, implement `SIMDDistance.distance(.hamming)`.

**Round 3** — integration through `ANNSIndex`
Remaining tests fail. Wire `useBinary` into `build()`, `supportsGPUSearch()`, serializer.

Run after every step:
```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E "PASS|FAIL|error:"
```

---

### Step 1: Add `.hamming` to `Metric`

**File**: `Sources/MetalANNSCore/Metric.swift`

```swift
public enum Metric: String, Sendable, Codable {
    case cosine
    case l2
    case innerProduct
    case hamming     // NEW — 1-bit Hamming distance; only valid with BinaryVectorBuffer
}
```

`MetalANNS/Metric.swift` is a typealias — no change needed there.

---

### Step 2: Add `SIMDDistance.hamming`

**File**: `Sources/MetalANNSCore/SIMDDistance.swift`

Add two overloads and extend the `distance(_:_:metric:)` switch:

```swift
/// Hamming distance between two unpacked binary vectors (values must be 0.0 or 1.0).
/// Returns the count of positions where a[i] != b[i] as a Float.
public static func hamming(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    var count: Int = 0
    for i in 0..<a.count {
        if a[i] != b[i] { count += 1 }
    }
    return Float(count)
}

/// Hamming distance between two packed binary byte arrays via XOR + popcount on UInt64 words.
/// `bytesPerVector` must equal a.count and b.count.
public static func hamming(packed a: [UInt8], packed b: [UInt8]) -> Float {
    precondition(a.count == b.count)
    var bits = 0
    let wordCount = a.count / 8
    a.withUnsafeBytes { aRaw in
        b.withUnsafeBytes { bRaw in
            let aWords = aRaw.bindMemory(to: UInt64.self)
            let bWords = bRaw.bindMemory(to: UInt64.self)
            for i in 0..<wordCount {
                bits += (aWords[i] ^ bWords[i]).nonzeroBitCount
            }
        }
    }
    // Handle trailing bytes (when dim is multiple of 8, bytesPerVector*8 = dim, no tail)
    for i in (wordCount * 8)..<a.count {
        bits += Int((a[i] ^ b[i]).nonzeroBitCount)
    }
    return Float(bits)
}
```

Extend `distance(_:_:metric:)`:
```swift
case .hamming:
    hamming(a, b)
```

---

### Step 3: Create `BinaryVectorBuffer.swift`

**File**: `Sources/MetalANNSCore/BinaryVectorBuffer.swift`

```swift
import Foundation
import Metal

/// 1-bit-per-dimension vector storage. Each float dim is binarized: ≥ 0.0 → 1, < 0.0 → 0.
/// Packed into bytes: `bytesPerVector = dim / 8`. `dim` must be a multiple of 8.
/// `vector(at:)` unpacks to [Float] with values {0.0, 1.0} — compatible with SIMDDistance.hamming.
public final class BinaryVectorBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer       // capacity * bytesPerVector bytes
    public let dim: Int                // original float dimension
    public let capacity: Int
    public private(set) var count: Int = 0

    public let bytesPerVector: Int     // dim / 8

    private let rawPointer: UnsafeMutablePointer<UInt8>

    public init(capacity: Int, dim: Int, device: MTLDevice? = nil) throws {
        guard dim > 0, dim % 8 == 0 else {
            throw ANNSError.constructionFailed(
                "BinaryVectorBuffer requires dim > 0 and dim % 8 == 0, got dim=\(dim)"
            )
        }
        guard capacity >= 0 else {
            throw ANNSError.constructionFailed("BinaryVectorBuffer requires capacity >= 0")
        }

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        let bpv = dim / 8
        let byteLength = max(capacity * bpv, 4)

        guard let buf = metalDevice.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate BinaryVectorBuffer")
        }

        self.buffer = buf
        self.dim = dim
        self.capacity = capacity
        self.bytesPerVector = bpv
        self.rawPointer = buf.contents().bindMemory(to: UInt8.self, capacity: max(capacity * bpv, 1))
    }

    /// Packs a float vector to bits: value >= 0.0 → 1, < 0.0 → 0.
    /// MSB of each byte corresponds to the lower-indexed dimension in that group of 8.
    public func insert(vector: [Float], at index: Int) throws {
        guard vector.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
        guard index >= 0, index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) out of bounds for capacity \(capacity)")
        }
        let base = index * bytesPerVector
        for byteIdx in 0..<bytesPerVector {
            var byte: UInt8 = 0
            for bit in 0..<8 {
                let dimIdx = byteIdx * 8 + bit
                if vector[dimIdx] >= 0.0 {
                    byte |= (1 << (7 - bit))
                }
            }
            rawPointer[base + byteIdx] = byte
        }
    }

    /// Returns unpacked [Float] with values {0.0, 1.0}.
    public func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < capacity)
        let base = index * bytesPerVector
        var result = [Float](repeating: 0.0, count: dim)
        for byteIdx in 0..<bytesPerVector {
            let byte = rawPointer[base + byteIdx]
            for bit in 0..<8 {
                let dimIdx = byteIdx * 8 + bit
                result[dimIdx] = (byte >> (7 - bit)) & 1 == 1 ? 1.0 : 0.0
            }
        }
        return result
    }

    /// Returns packed bytes for the vector at `index` — used by the Hamming Metal kernel.
    public func packedVector(at index: Int) -> [UInt8] {
        precondition(index >= 0 && index < capacity)
        let base = index * bytesPerVector
        return Array(UnsafeBufferPointer(start: rawPointer.advanced(by: base), count: bytesPerVector))
    }
}

extension BinaryVectorBuffer: VectorStorage {
    public var isFloat16: Bool { false }

    public func setCount(_ newCount: Int) { count = newCount }

    public func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        for (offset, vector) in vectors.enumerated() {
            try insert(vector: vector, at: start + offset)
        }
    }
}
```

---

### Step 4: Add `HammingDistance.metal`

**File**: `Sources/MetalANNSCore/Shaders/HammingDistance.metal`

```metal
#include <metal_stdlib>
using namespace metal;

/// Compute Hamming distance from a packed binary query to each packed binary corpus vector.
/// `bytesPerVector` = dim / 8. Output[tid] = popcount(XOR) as float.
kernel void hamming_distance(
    device const uchar *query     [[buffer(0)]],
    device const uchar *corpus    [[buffer(1)]],
    device float       *output    [[buffer(2)]],
    constant uint &bytesPerVector [[buffer(3)]],
    constant uint &n              [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    uint dist = 0;
    uint base = tid * bytesPerVector;
    for (uint i = 0; i < bytesPerVector; i++) {
        dist += popcount(query[i] ^ corpus[base + i]);
    }
    output[tid] = float(dist);
}
```

The kernel is compiled as part of the default Metal library. It is not wired into search
in this phase — that is Phase 25 scope. Do not register it in `PipelineCache` yet.

---

### Step 5: Add `useBinary` to `IndexConfiguration`

**File**: `Sources/MetalANNS/IndexConfiguration.swift`

Add `public var useBinary: Bool = false` as a new stored property. Follow the exact
same pattern as `useFloat16`:
- Add to the stored properties
- Add to `static let default` (value: `false`)
- Add to `init(...)` with default `false`
- Add to `init(from decoder: Decoder)` using `decodeIfPresent` with fallback `false`

`useBinary` is only valid when `metric == .hamming`. Validation is enforced in `build()`,
not in `IndexConfiguration` itself.

---

### Step 6: Wire Binary into `ANNSIndex`

**File**: `Sources/MetalANNS/ANNSIndex.swift`

**6a. `build()` — storage dispatch**

Replace the `useFloat16` dispatch block (current lines ~96-99):
```swift
// BEFORE:
if configuration.useFloat16 {
    vectorBuffer = try Float16VectorBuffer(...)
} else {
    vectorBuffer = try VectorBuffer(...)
}

// AFTER:
if configuration.useBinary {
    guard configuration.metric == .hamming else {
        throw ANNSError.constructionFailed("useBinary requires metric == .hamming")
    }
    for vec in inputVectors {
        guard vec.count % 8 == 0 else {
            throw ANNSError.constructionFailed(
                "Binary index requires dim % 8 == 0, got dim=\(vec.count)"
            )
        }
    }
    vectorBuffer = try BinaryVectorBuffer(capacity: capacity, dim: dim, device: device)
} else if configuration.useFloat16 {
    vectorBuffer = try Float16VectorBuffer(capacity: capacity, dim: dim, device: device)
} else {
    vectorBuffer = try VectorBuffer(capacity: capacity, dim: dim, device: device)
}
```

**6b. `build()` — force CPU path for binary**

Binary vectors in `BinaryVectorBuffer.buffer` are packed bits — `NNDescentGPU` reads
the buffer as Float32, producing garbage. Force CPU NN-Descent when `useBinary`:

```swift
// Replace the `if let context { NNDescentGPU... } else { NNDescentCPU... }` block:
let cpuVectors = (0..<inputVectors.count).map { vectorBuffer.vector(at: $0) }
if let context, !configuration.useBinary {
    // ... GPU path (unchanged)
} else {
    // CPU NN-Descent — use unpacked float vectors for distance computation
    let cpuResult = try await NNDescentCPU.build(
        vectors: cpuVectors,
        degree: configuration.degree,
        metric: configuration.metric,
        ...
    )
    // ... existing CPU graph-writing code (unchanged)
}
```

**6c. `supportsGPUSearch(for:)`**

```swift
private func supportsGPUSearch(for vectors: any VectorStorage) -> Bool {
    !(vectors is DiskBackedVectorBuffer) && !(vectors is BinaryVectorBuffer)
}
```

**6d. `applyLoadedState()` — `useBinary` from loaded state**

After loading, set `resolvedConfiguration.useBinary = loaded.vectors is BinaryVectorBuffer`.
This mirrors how `useFloat16` is set from `loaded.vectors.isFloat16`.

---

### Step 7: Update `IndexSerializer`

**File**: `Sources/MetalANNSCore/IndexSerializer.swift`

**7a. `metricCode(_:)` — add hamming**

```swift
case .hamming:
    return 3
```

**7b. `metric(from:)` — add code 3**

```swift
case 3:
    return .hamming
```

**7c. `save()` — handle storageType=2 for binary**

```swift
let storageType: UInt32
let bytesPerElement: Int
let vectorByteCount: Int

if let binaryBuffer = vectors as? BinaryVectorBuffer {
    storageType = 2
    bytesPerElement = 1   // 1 byte per packed group of 8 dims
    vectorByteCount = nodeCount * binaryBuffer.bytesPerVector
} else {
    storageType = vectors.isFloat16 ? 1 : 0
    bytesPerElement = vectors.isFloat16 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride
    vectorByteCount = nodeCount * vectors.dim * bytesPerElement
}
```

Write `storageType` to the header. Write `vectorByteCount` bytes from `vectors.buffer.contents()`.

**7d. `load()` — handle storageType=2**

```swift
guard storageType == 0 || storageType == 1 || storageType == 2 else {
    throw ANNSError.corruptFile("Unsupported storage type \(storageType)")
}

let vectorByteCount: Int
if storageType == 2 {
    guard dim % 8 == 0 else {
        throw ANNSError.corruptFile("Binary index has dim not divisible by 8")
    }
    vectorByteCount = nodeCount * (dim / 8)
} else {
    let bytesPerElement = storageType == 1 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride
    vectorByteCount = nodeCount * dim * bytesPerElement
}
```

Instantiate the correct type on load:
```swift
let vectors: any VectorStorage = switch storageType {
case 2:  try BinaryVectorBuffer(capacity: mutableCapacity, dim: dim, device: device)
case 1:  try Float16VectorBuffer(capacity: mutableCapacity, dim: dim, device: device)
default: try VectorBuffer(capacity: mutableCapacity, dim: dim, device: device)
}
```

---

### Step 8: Write `BinaryQuantizationTests.swift`

**File**: `Tests/MetalANNSTests/BinaryQuantizationTests.swift`

```swift
import Testing
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("Binary Quantization Tests")
struct BinaryQuantizationTests {

    // TEST 1: binaryInsertAndSearch
    // Build ANNSIndex (useBinary: true, metric: .hamming, dim=64, 200 vectors, CPU-only).
    // Search top-10 for 5 queries. Assert results.count == 10 and all IDs are valid.
    @Test func binaryInsertAndSearch() async throws { ... }

    // TEST 2: hammingRecall
    // Build binary index with 500 vectors (dim=64, random {-1,1} floats → binary).
    // For 20 queries, compute recall@10 vs brute-force Hamming ground truth.
    // Ground truth: sort all vectors by hamming distance, take top 10.
    // #expect(recall >= 0.70)  — lower threshold: binarization is lossy
    @Test func hammingRecall() async throws { ... }

    // TEST 3: memoryReduction
    // Create a BinaryVectorBuffer(capacity: 1000, dim: 128).
    // Create a VectorBuffer(capacity: 1000, dim: 128).
    // Assert binaryBuffer.buffer.length * 32 <= vectorBuffer.buffer.length * 1
    // (binary uses dim/8 bytes vs dim*4 bytes → 32× reduction)
    @Test func memoryReduction() throws { ... }

    // TEST 4: packUnpackRoundTrip
    // For 10 random vectors (dim=32): insert into BinaryVectorBuffer.
    // vector(at:) must return [Float] with only 0.0 or 1.0 values.
    // Re-packing the unpacked result must produce the same packed bytes.
    @Test func packUnpackRoundTrip() throws { ... }

    // TEST 5: hammingDistanceCorrectness
    // Known input: a = [1,0,1,0,...] (dim=8), b = [0,0,1,1,...] (dim=8) — 2 differences.
    // SIMDDistance.hamming(a, b) == 2.0
    // SIMDDistance.distance(a, b, metric: .hamming) == 2.0
    @Test func hammingDistanceCorrectness() { ... }

    // TEST 6: persistenceRoundTrip
    // Build binary index with 100 vectors. Save to temp URL. Load back. Search.
    // Assert loaded results match pre-save results (same top-5 IDs).
    @Test func persistenceRoundTrip() async throws { ... }

    // TEST 7: dimensionConstraintEnforced
    // BinaryVectorBuffer(capacity: 10, dim: 7) must throw ANNSError.constructionFailed.
    // IndexConfiguration(useBinary: true, metric: .hamming) + build with dim=7 must throw.
    @Test func dimensionConstraintEnforced() throws { ... }

    // TEST 8: useBinaryRequiresHammingMetric
    // build() with useBinary: true, metric: .cosine must throw ANNSError.constructionFailed.
    @Test func useBinaryRequiresHammingMetric() async throws { ... }
}
```

**Private helpers:**
```swift
// Generate random binary float vectors: values uniformly drawn from {-1.0, 1.0}
private func randomBinaryVectors(count: Int, dim: Int) -> [[Float]] { ... }

// Brute-force Hamming top-k via SIMDDistance.hamming
private func bruteForceHammingTopK(query: [Float], vectors: [[Float]], k: Int) -> Set<Int> { ... }

private func recall(results: [SearchResult], groundTruth: Set<Int>, ids: [String]) -> Double { ... }
```

---

### Step 9: Verify No Regressions

```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30
```

Critical checks:
- `FilteredSearchTests.swift` — float32 indexes unaffected
- `HNSWTests.swift` — HNSW unaffected
- Any serializer tests — existing storageType 0/1 round-trips still work
- `SIMDDistance.distance()` for `.cosine`, `.l2`, `.innerProduct` — unchanged paths

---

### Definition of Done

- [ ] `Metric.hamming` exists in `MetalANNSCore/Metric.swift`
- [ ] `SIMDDistance.hamming([Float],[Float])` and `hamming(packed:[UInt8],packed:[UInt8])` exist; `distance(.hamming)` dispatches correctly
- [ ] `BinaryVectorBuffer` conforms to `VectorStorage`; `dim % 8 != 0` throws; `vector(at:)` returns 0.0/1.0 floats
- [ ] `HammingDistance.metal` kernel compiles (verified by xcodebuild, not `swift build`)
- [ ] `IndexConfiguration.useBinary: Bool = false`; `useBinary + metric != .hamming` throws at build time
- [ ] GPU build and GPU search paths disabled for `BinaryVectorBuffer`
- [ ] `IndexSerializer` reads and writes `storageType=2` and `metricCode=3` correctly
- [ ] All 8 new tests pass including `hammingRecall >= 0.70` and `persistenceRoundTrip`
- [ ] All pre-existing tests pass — zero regressions

---

### What Not To Do

- Do not add `dim % 8 == 0` validation to `IndexConfiguration` — enforce it in `build()` and `BinaryVectorBuffer.init()` only
- Do not wire `hamming_distance` kernel into `PipelineCache` or search paths — the Metal kernel exists for future use (Phase 25 scope)
- Do not use `BinaryVectorBuffer` with `NNDescentGPU` — force CPU NN-Descent when `useBinary == true`
- Do not add `DistanceFloat16.metal` guards for hamming — that file is only invoked when `isFloat16 == true`, which `BinaryVectorBuffer` is not; no change needed there
- Do not normalize Hamming distance — return raw popcount as Float for consistency with how the graph stores distances
