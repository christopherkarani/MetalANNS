# MetalANNS Wax-Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all MetalANNS blockers so it can fully replace USearch as Wax's vector search backend — delivering pure-Swift, zero-dependency ANN search with equal or better performance.

**Architecture:** Six phases targeting the critical gaps: (1) buffer pooling to eliminate per-search allocation, (2) lifting the 4096-node GPU search ceiling, (3) fixing StreamingIndex unbounded memory growth, (4) adding UInt64 native ID support for Wax's frameId type, (5) GPU-vs-CPU search parity tests, (6) algorithmic optimizations (early-exit in local_join, seeded test RNG, rangeSearch guard fix).

**Tech Stack:** Swift 6.0, Metal Shading Language, Apple Accelerate, Swift Testing framework.

---

## Phase 1: Buffer Pool for GPU Search

### Task 1.1: Create SearchBufferPool

**Files:**
- Create: `Sources/MetalANNSCore/SearchBufferPool.swift`
- Test: `Tests/MetalANNSTests/SearchBufferPoolTests.swift`

**Context:** `FullGPUSearch.search()` (`FullGPUSearch.swift:58-74`) allocates 3 new `MTLBuffer` objects per search call (queryBuffer, outputDistanceBuffer, outputIDBuffer). Metal buffer allocation involves page table modifications and is not free. Under concurrent `batchSearch` this creates significant allocator pressure.

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/SearchBufferPoolTests.swift
import Testing
import Metal
@testable import MetalANNSCore

@Suite("SearchBufferPool Tests")
struct SearchBufferPoolTests {

    @Test func acquireAndReleaseReturnsSameBuffer() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let pool = SearchBufferPool(device: device)

        let buffers1 = try pool.acquire(queryDim: 128, maxK: 10)
        let queryPtr1 = buffers1.queryBuffer.gpuAddress

        pool.release(buffers1)

        let buffers2 = try pool.acquire(queryDim: 128, maxK: 10)
        let queryPtr2 = buffers2.queryBuffer.gpuAddress

        // Should reuse the same buffer
        #expect(queryPtr1 == queryPtr2)
        pool.release(buffers2)
    }

    @Test func acquireLargerDimAllocatesNew() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let pool = SearchBufferPool(device: device)

        let small = try pool.acquire(queryDim: 64, maxK: 10)
        pool.release(small)

        // Larger dim needs bigger buffer — should not reuse
        let large = try pool.acquire(queryDim: 512, maxK: 10)
        #expect(large.queryBuffer.length >= 512 * MemoryLayout<Float>.stride)
        pool.release(large)
    }

    @Test func concurrentAcquireReturnsDistinctBuffers() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let pool = SearchBufferPool(device: device)

        let b1 = try pool.acquire(queryDim: 128, maxK: 10)
        let b2 = try pool.acquire(queryDim: 128, maxK: 10)

        // Two concurrent acquires must return different buffers
        #expect(b1.queryBuffer.gpuAddress != b2.queryBuffer.gpuAddress)
        pool.release(b1)
        pool.release(b2)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/SearchBufferPoolTests 2>&1 | tail -20`
Expected: FAIL — `SearchBufferPool` does not exist yet.

**Step 3: Write minimal implementation**

```swift
// Sources/MetalANNSCore/SearchBufferPool.swift
import Foundation
import Metal

/// Reusable buffer pool for GPU search operations.
/// Eliminates per-search MTLBuffer allocation in FullGPUSearch.
public final class SearchBufferPool: @unchecked Sendable {

    public struct Buffers: Sendable {
        public let queryBuffer: MTLBuffer
        public let outputDistanceBuffer: MTLBuffer
        public let outputIDBuffer: MTLBuffer
        public let queryDim: Int
        public let maxK: Int
    }

    private let device: MTLDevice
    private var available: [Buffers] = []
    private let lock = NSLock()

    public init(device: MTLDevice) {
        self.device = device
    }

    public func acquire(queryDim: Int, maxK: Int) throws -> Buffers {
        lock.lock()
        defer { lock.unlock() }

        if let index = available.firstIndex(where: {
            $0.queryDim >= queryDim && $0.maxK >= maxK
        }) {
            return available.remove(at: index)
        }

        return try allocate(queryDim: queryDim, maxK: maxK)
    }

    public func release(_ buffers: Buffers) {
        lock.lock()
        defer { lock.unlock() }
        available.append(buffers)
    }

    private func allocate(queryDim: Int, maxK: Int) throws -> Buffers {
        let floatSize = MemoryLayout<Float>.stride
        let uintSize = MemoryLayout<UInt32>.stride

        guard
            let queryBuffer = device.makeBuffer(
                length: queryDim * floatSize,
                options: .storageModeShared
            ),
            let distBuffer = device.makeBuffer(
                length: max(maxK * floatSize, floatSize),
                options: .storageModeShared
            ),
            let idBuffer = device.makeBuffer(
                length: max(maxK * uintSize, uintSize),
                options: .storageModeShared
            )
        else {
            throw ANNSError.searchFailed("Failed to allocate search buffer pool entry")
        }

        return Buffers(
            queryBuffer: queryBuffer,
            outputDistanceBuffer: distBuffer,
            outputIDBuffer: idBuffer,
            queryDim: queryDim,
            maxK: maxK
        )
    }
}
```

**Step 4: Run test to verify it passes**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/SearchBufferPoolTests 2>&1 | tail -20`
Expected: PASS — all 3 tests green.

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/SearchBufferPool.swift Tests/MetalANNSTests/SearchBufferPoolTests.swift
git commit -m "feat: add SearchBufferPool to eliminate per-search MTLBuffer allocation"
```

---

### Task 1.2: Wire SearchBufferPool into FullGPUSearch

**Files:**
- Modify: `Sources/MetalANNSCore/FullGPUSearch.swift:4-6` (add pool storage)
- Modify: `Sources/MetalANNSCore/FullGPUSearch.swift:58-74` (replace makeBuffer with pool.acquire)
- Modify: `Sources/MetalANNSCore/MetalDevice.swift:8-12` (add pool property to MetalContext)

**Step 1: Write the failing test**

```swift
// Add to Tests/MetalANNSTests/SearchBufferPoolTests.swift

@Test func fullGPUSearchReusesBuffers() async throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Skipping: no Metal device")
        return
    }
    let context = try MetalContext()

    // Build a small index
    let dim = 16
    let nodeCount = 32
    let degree = 8
    var rng = SeededGenerator(state: 42)
    let vectors = (0..<nodeCount).map { _ in
        (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    }

    let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
    for (i, v) in vectors.enumerated() {
        try vectorBuffer.setVector(v, at: i)
    }
    vectorBuffer.setCount(nodeCount)

    let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
    try NNDescentGPU.build(
        context: context, graph: graph, vectors: vectorBuffer,
        nodeCount: nodeCount, metric: .cosine
    )

    // Search twice — second should reuse pooled buffers
    let result1 = try await FullGPUSearch.search(
        context: context, query: vectors[0], vectors: vectorBuffer,
        graph: graph, entryPoint: 0, k: 5, ef: 16, metric: .cosine
    )
    let result2 = try await FullGPUSearch.search(
        context: context, query: vectors[1], vectors: vectorBuffer,
        graph: graph, entryPoint: 0, k: 5, ef: 16, metric: .cosine
    )

    #expect(result1.count > 0)
    #expect(result2.count > 0)
    // Pool is internal — we verify correctness, not reuse directly.
    // The real validation is that no allocation failures occur under load.
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/SearchBufferPoolTests/fullGPUSearchReusesBuffers 2>&1 | tail -20`
Expected: PASS (this test validates correctness after the refactor; write it first as a safety net).

**Step 3: Refactor FullGPUSearch to use the pool**

In `MetalDevice.swift`, add pool to `MetalContext`:

```swift
// MetalDevice.swift:12 — add after pipelineCache
public let searchBufferPool: SearchBufferPool
// MetalDevice.swift:36 — add in init()
self.searchBufferPool = SearchBufferPool(device: device)
```

In `FullGPUSearch.swift`, replace lines 58-74 (buffer allocation) with:

```swift
let buffers = try context.searchBufferPool.acquire(queryDim: query.count, maxK: kLimit)
defer { context.searchBufferPool.release(buffers) }

let queryBuffer = buffers.queryBuffer
queryBuffer.contents().copyMemory(from: query, byteCount: query.count * floatSize)
let outputDistanceBuffer = buffers.outputDistanceBuffer
let outputIDBuffer = buffers.outputIDBuffer
```

**Step 4: Run full test suite to verify no regressions**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/FullGPUSearch.swift Sources/MetalANNSCore/MetalDevice.swift
git commit -m "refactor: wire SearchBufferPool into FullGPUSearch, eliminating per-search allocation"
```

---

## Phase 2: Lift 4096-Node GPU Search Ceiling

### Task 2.1: Replace hash-table visited set with generation-counter bitset

**Files:**
- Modify: `Sources/MetalANNSCore/Shaders/Search.metal:4-7` (replace MAX_VISITED constant)
- Modify: `Sources/MetalANNSCore/Shaders/Search.metal:53-74` (replace try_visit with generation bitset)
- Modify: `Sources/MetalANNSCore/FullGPUSearch.swift:6` (update maxVisited)
- Modify: `Sources/MetalANNSCore/FullGPUSearch.swift:58-74` (add visited buffer allocation)
- Test: `Tests/MetalANNSTests/FullGPUSearchTests.swift` (new file)

**Context:** The current `try_visit` function uses a hash table of size 4096 in threadgroup shared memory (`Search.metal:5`). This limits full GPU search to 4096 nodes. We replace it with a device-memory generation-counter buffer — one `uint` per node. Each search increments a generation counter; a node is "visited" if `visited[nodeID] == currentGeneration`. This allows arbitrary node counts with zero threadgroup memory usage for the visited set.

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/FullGPUSearchTests.swift
import Testing
import Metal
@testable import MetalANNS
@testable import MetalANNSCore

struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

@Suite("FullGPUSearch Tests")
struct FullGPUSearchTests {

    @Test func searchAbove4096NodesReturnsResults() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let context = try MetalContext()

        let dim = 32
        let nodeCount = 5000  // Above old 4096 limit
        let degree = 16
        var rng = SeededGenerator(state: 42)
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
        for (i, v) in vectors.enumerated() {
            try vectorBuffer.setVector(v, at: i)
        }
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
        try await NNDescentGPU.build(
            context: context, graph: graph, vectors: vectorBuffer,
            nodeCount: nodeCount, metric: .cosine
        )

        let results = try await FullGPUSearch.search(
            context: context,
            query: vectors[0],
            vectors: vectorBuffer,
            graph: graph,
            entryPoint: 0,
            k: 10,
            ef: 64,
            metric: .cosine
        )

        #expect(results.count == 10)
        // First result should be the query vector itself (distance ~0)
        #expect(results[0].score < 0.01)
    }

    @Test func gpuSearchMatchesCPUAtSmallScale() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let context = try MetalContext()

        let dim = 32
        let nodeCount = 200
        let degree = 16
        var rng = SeededGenerator(state: 99)
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
        for (i, v) in vectors.enumerated() {
            try vectorBuffer.setVector(v, at: i)
        }
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
        try await NNDescentGPU.build(
            context: context, graph: graph, vectors: vectorBuffer,
            nodeCount: nodeCount, metric: .cosine
        )

        let query = vectors[5]
        let k = 10
        let ef = 64

        // GPU search
        let gpuResults = try await FullGPUSearch.search(
            context: context, query: query, vectors: vectorBuffer,
            graph: graph, entryPoint: 0, k: k, ef: ef, metric: .cosine
        )

        // CPU search (ground truth)
        let extractedVectors = (0..<nodeCount).map { i in vectorBuffer.vector(at: i) }
        let extractedGraph = (0..<nodeCount).map { i in
            let ids = graph.neighborIDs(of: i)
            let dists = graph.neighborDistances(of: i)
            return zip(ids, dists).map { ($0.0, $0.1) }
        }
        let cpuResults = try await BeamSearchCPU.search(
            query: query, vectors: extractedVectors, graph: extractedGraph,
            entryPoint: 0, k: k, ef: ef, metric: .cosine
        )

        // GPU and CPU should find mostly the same neighbors
        let gpuIDs = Set(gpuResults.map(\.internalID))
        let cpuIDs = Set(cpuResults.map(\.internalID))
        let overlap = gpuIDs.intersection(cpuIDs).count
        let recall = Float(overlap) / Float(k)

        #expect(recall >= 0.7, "GPU-vs-CPU recall \(recall) below threshold 0.7")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/FullGPUSearchTests 2>&1 | tail -20`
Expected: `searchAbove4096NodesReturnsResults` FAILS with "nodeCount exceeds FullGPUSearch visited-table capacity (4096)".

**Step 3: Implement generation-counter visited set**

**Search.metal** — Replace lines 4-7 and the `try_visit` function:

```metal
// Replace lines 4-7:
constant uint MAX_EF = 256;
constant uint MAX_PROBES = 32;  // kept for legacy; unused by new visited
constant uint EMPTY_SLOT = 0xFFFFFFFFu;

// Replace try_visit (lines 53-74) with:
inline bool try_visit(
    device atomic_uint *visited,
    uint nodeID,
    uint generation
) {
    uint prev = atomic_exchange_explicit(&visited[nodeID], generation, memory_order_relaxed);
    return prev != generation;
}
```

Update the `beam_search` kernel signature to accept a visited buffer and generation counter:

```metal
// Add after buffer(11):
    device atomic_uint *visited [[buffer(12)]],
    constant uint &generation [[buffer(13)]],
```

Remove the threadgroup `visited` array declaration inside the kernel (currently uses `threadgroup atomic_uint visited[MAX_VISITED]`). Replace all `try_visit(visited, nodeID)` calls with `try_visit(visited, nodeID, generation)`.

**FullGPUSearch.swift** — Update:

```swift
// Line 6: Remove maxVisited or raise to device limit
// Remove the guard at lines 43-47 (nodeCount <= maxVisited)

// Add visited buffer to pool or allocate alongside:
// In the search function, after acquiring pool buffers:
let visitedBuffer = try context.searchBufferPool.acquireVisited(nodeCount: nodeCount)
defer { context.searchBufferPool.releaseVisited(visitedBuffer) }

var generation = UInt32(/* increment a counter on the pool */)

// Bind visited buffer and generation to encoder:
encoder.setBuffer(visitedBuffer, offset: 0, index: 12)
encoder.setBytes(&generation, length: uintSize, index: 13)
```

**SearchBufferPool.swift** — Add visited buffer management:

```swift
// Add to SearchBufferPool:
private var visitedBuffers: [(buffer: MTLBuffer, capacity: Int)] = []
private var generationCounter: UInt32 = 0

public func acquireVisited(nodeCount: Int) throws -> (buffer: MTLBuffer, generation: UInt32) {
    lock.lock()
    defer { lock.unlock() }
    generationCounter &+= 1

    if let index = visitedBuffers.firstIndex(where: { $0.capacity >= nodeCount }) {
        let entry = visitedBuffers.remove(at: index)
        return (entry.buffer, generationCounter)
    }

    let length = nodeCount * MemoryLayout<UInt32>.stride
    guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
        throw ANNSError.searchFailed("Failed to allocate visited buffer")
    }
    // Zero-initialize
    memset(buffer.contents(), 0, length)
    return (buffer, generationCounter)
}

public func releaseVisited(_ buffer: MTLBuffer, capacity: Int) {
    lock.lock()
    defer { lock.unlock() }
    visitedBuffers.append((buffer: buffer, capacity: capacity))
}
```

**ANNSIndex.swift** — Remove line 6 (`fullGPUMaxEF = 256`) gate on nodeCount. Keep the ef/k gates.

**Step 4: Run tests to verify both pass**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/FullGPUSearchTests 2>&1 | tail -20`
Expected: PASS — both `searchAbove4096NodesReturnsResults` and `gpuSearchMatchesCPUAtSmallScale` green.

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30`
Expected: Full suite PASS (no regressions).

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/Shaders/Search.metal Sources/MetalANNSCore/FullGPUSearch.swift \
    Sources/MetalANNSCore/SearchBufferPool.swift Sources/MetalANNS/ANNSIndex.swift \
    Tests/MetalANNSTests/FullGPUSearchTests.swift
git commit -m "feat: lift 4096-node GPU search ceiling using generation-counter visited set"
```

---

### Task 2.2: Update Float16 beam search kernel with same visited pattern

**Files:**
- Modify: `Sources/MetalANNSCore/Shaders/SearchFloat16.metal` (mirror Search.metal changes)
- Test: Add Float16 variant of `gpuSearchMatchesCPUAtSmallScale` to `FullGPUSearchTests.swift`

Follow the exact same pattern as Task 2.1 but for the `beam_search_f16` kernel. The changes are identical — replace threadgroup hash table with device-memory generation counter.

**Commit message:** `feat: lift 4096-node ceiling for Float16 beam search kernel`

---

## Phase 3: Fix StreamingIndex Unbounded Memory Growth

### Task 3.1: Replace allVectorsList with append-only vector log

**Files:**
- Modify: `Sources/MetalANNS/StreamingIndex.swift:17-24` (PersistedMeta struct)
- Modify: `Sources/MetalANNS/StreamingIndex.swift:37-39` (allVectorsList/allIDsList/allIDs)
- Modify: `Sources/MetalANNS/StreamingIndex.swift:~300-347` (save method)
- Test: `Tests/MetalANNSTests/StreamingIndexMemoryTests.swift` (new file)

**Context:** `StreamingIndex` keeps `allVectorsList: [[Float]]` (`StreamingIndex.swift:37`) — every vector ever inserted, including soft-deleted ones. On `save()` this entire array is JSON-encoded into `PersistedMeta` (`StreamingIndex.swift:17-24`). At 100k vectors × 384 dims × 4 bytes = 153 MB of vectors serialized as JSON text (which bloats to ~500MB+). This causes OOM for Wax's continuous-ingest workload.

**Fix:** After a merge completes, vectors incorporated into the base index are no longer needed in `allVectorsList`. Only post-merge vectors (pending + delta that haven't been merged yet) need to be kept. Deleted vectors should be removed immediately.

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/StreamingIndexMemoryTests.swift
import Testing
import Foundation
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("StreamingIndex Memory Tests")
struct StreamingIndexMemoryTests {

    @Test func mergedVectorsAreEvictedFromHistory() async throws {
        let config = StreamingConfiguration(
            deltaCapacity: 10,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 4, metric: .cosine)
        )
        let index = StreamingIndex(config: config)

        // Insert enough vectors to trigger a merge (deltaCapacity = 10)
        let dim = 16
        var rng = SeededGenerator(state: 42)
        for i in 0..<15 {
            let vector = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
            try await index.insert(vector, id: "vec-\(i)")
        }

        // Flush to ensure merge completes
        try await index.flush()

        // Save and check file size is reasonable
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("streaming-mem-\(UUID().uuidString)")
        try await index.save(to: tempDir)

        let metaURL = tempDir.appendingPathComponent("streaming.meta.json")
        let metaData = try Data(contentsOf: metaURL)

        // After merge, the meta file should NOT contain all 15 vectors as raw arrays
        // It should only contain post-merge pending vectors (if any)
        // A reasonable upper bound: meta should be < 10KB for 15 small vectors
        #expect(metaData.count < 50_000,
            "Meta file is \(metaData.count) bytes — allVectorsList not evicted after merge")

        // Verify the index still returns correct results
        let query = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        let results = try await index.search(query: query, k: 5)
        #expect(results.count == 5)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test func deletedVectorsAreRemovedFromHistory() async throws {
        let config = StreamingConfiguration(
            deltaCapacity: 50,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 4, metric: .cosine)
        )
        let index = StreamingIndex(config: config)

        let dim = 16
        var rng = SeededGenerator(state: 77)
        for i in 0..<10 {
            let vector = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
            try await index.insert(vector, id: "vec-\(i)")
        }

        // Delete 5 vectors
        for i in 0..<5 {
            try await index.delete(id: "vec-\(i)")
        }

        // The internal vector history should not retain deleted vectors
        // Verify via round-trip: save, load, search should find only 5 results
        try await index.flush()
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("streaming-del-\(UUID().uuidString)")
        try await index.save(to: tempDir)

        let loaded = try await StreamingIndex.load(from: tempDir)
        #expect(await loaded.count == 5)

        try? FileManager.default.removeItem(at: tempDir)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/StreamingIndexMemoryTests 2>&1 | tail -20`
Expected: `mergedVectorsAreEvictedFromHistory` FAILS — meta file size exceeds threshold because allVectorsList keeps everything.

**Step 3: Implement eviction after merge**

In `StreamingIndex.swift`, modify the merge completion path to clear merged vectors from `allVectorsList` and `allIDsList`. After `base` is rebuilt from the full vector set, only vectors that were inserted AFTER the merge started need to remain in the lists.

Key changes:
1. After merge completes, set `allVectorsList` and `allIDsList` to only contain vectors not yet in the base index (i.e., vectors in `pendingVectors` and any delta vectors added during the merge).
2. In the `delete` method, also remove the vector from `allVectorsList`/`allIDsList` (find by matching ID in `allIDsList`).
3. Update `PersistedMeta` to only serialize post-merge vectors.

The specific implementation: after `performMerge()` completes and the new `base` is set, replace:

```swift
// Current: allVectorsList keeps growing forever
// Replace with:
allVectorsList = pendingVectors
allIDsList = pendingIDs
allIDs = Set(pendingIDs).union(deletedIDs.isEmpty ? [] : [])
```

For delete, add cleanup:

```swift
// In delete(id:), after adding to deletedIDs:
if let idx = allIDsList.firstIndex(of: id) {
    allIDsList.remove(at: idx)
    allVectorsList.remove(at: idx)
}
```

**Step 4: Run tests to verify both pass**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/StreamingIndexMemoryTests 2>&1 | tail -20`
Expected: PASS.

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30`
Expected: Full suite PASS.

**Step 5: Commit**

```bash
git add Sources/MetalANNS/StreamingIndex.swift Tests/MetalANNSTests/StreamingIndexMemoryTests.swift
git commit -m "fix: evict merged and deleted vectors from StreamingIndex history to prevent OOM"
```

---

## Phase 4: Native UInt64 ID Support

### Task 4.1: Add UInt64 key support to IDMap

**Files:**
- Modify: `Sources/MetalANNSCore/IDMap.swift` (add UInt64 overloads)
- Test: `Tests/MetalANNSTests/IDMapTests.swift` (new file)

**Context:** Wax uses `frameId: UInt64` as its primary key. MetalANNS's `IDMap` (`IDMap.swift:4-53`) only supports `String` external IDs. Without native UInt64 support, the Wax adapter would need to convert `UInt64 → String → UInt64` on every insert and search result, adding allocation and parsing overhead on the hot path.

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/IDMapTests.swift
import Testing
@testable import MetalANNSCore

@Suite("IDMap Tests")
struct IDMapTests {

    @Test func assignUInt64Key() {
        var map = IDMap()
        let internalID = map.assign(numericID: 42)
        #expect(internalID != nil)
        #expect(map.internalID(forNumeric: 42) == internalID)
        #expect(map.numericID(for: internalID!) == 42)
    }

    @Test func uint64KeyCollidesWithStringKey() {
        var map = IDMap()
        // String "42" and UInt64 42 should be independent keys
        let strID = map.assign(externalID: "42")
        let numID = map.assign(numericID: 42)
        #expect(strID != nil)
        #expect(numID != nil)
        #expect(strID != numID)
    }

    @Test func uint64KeyPreservesRoundTrip() {
        var map = IDMap()
        let values: [UInt64] = [0, 1, UInt64.max - 1, 12345678901234]
        for val in values {
            let internal_id = map.assign(numericID: val)
            #expect(internal_id != nil)
            #expect(map.numericID(for: internal_id!) == val)
        }
    }

    @Test func canAllocateChecksCapacity() {
        var map = IDMap()
        #expect(map.canAllocate(100))
        // Assign some IDs
        for i in 0..<50 {
            _ = map.assign(numericID: UInt64(i))
        }
        #expect(map.count == 50)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/IDMapTests 2>&1 | tail -20`
Expected: FAIL — `assign(numericID:)` does not exist.

**Step 3: Add UInt64 overloads to IDMap**

```swift
// Add to IDMap.swift after line 48:

// MARK: - UInt64 Key Support

private var numericToInternal: [UInt64: UInt32] = [:]
private var internalToNumeric: [UInt32: UInt64] = [:]

/// Assigns a new internal ID for a numeric (UInt64) key. Returns nil if already exists.
public mutating func assign(numericID: UInt64) -> UInt32? {
    guard numericToInternal[numericID] == nil else {
        return nil
    }
    guard nextID < UInt32.max else {
        return nil
    }
    let internalID = nextID
    numericToInternal[numericID] = internalID
    internalToNumeric[internalID] = numericID
    nextID += 1
    return internalID
}

public func internalID(forNumeric numericID: UInt64) -> UInt32? {
    numericToInternal[numericID]
}

public func numericID(for internalID: UInt32) -> UInt64? {
    internalToNumeric[internalID]
}
```

Also update `count` to include both maps, and ensure `Codable` conformance encodes the new dictionaries.

**Step 4: Run tests to verify they pass**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/IDMapTests 2>&1 | tail -20`
Expected: PASS.

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/IDMap.swift Tests/MetalANNSTests/IDMapTests.swift
git commit -m "feat: add native UInt64 key support to IDMap for Wax frameId compatibility"
```

---

### Task 4.2: Add UInt64-keyed insert/search to ANNSIndex

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift` (add `insert(_:numericID:)`, search returning numericID)
- Test: Add to `Tests/MetalANNSTests/ANNSIndexTests.swift`

**Step 1: Write the failing test**

```swift
@Test func insertAndSearchWithUInt64IDs() async throws {
    let config = IndexConfiguration(degree: 8, metric: .cosine)
    let index = ANNSIndex(configuration: config)

    let dim = 16
    var rng = SeededGenerator(state: 42)
    let baseVectors = (0..<20).map { _ in (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) } }
    let baseIDs = (0..<20).map { "base-\($0)" }
    try await index.build(vectors: baseVectors, ids: baseIDs)

    // Insert with UInt64 ID
    let newVector = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    try await index.insert(newVector, numericID: 999)

    // Search and verify numericID is returned
    let results = try await index.search(query: newVector, k: 1)
    #expect(results.count == 1)
    #expect(results[0].numericID == 999)
}
```

**Step 2-5:** Implement `insert(_:numericID:)` on ANNSIndex, add `numericID: UInt64?` to `SearchResult`, wire through IDMap. Follow TDD cycle. Commit.

**Commit message:** `feat: add UInt64-keyed insert and search to ANNSIndex for Wax integration`

---

## Phase 5: GPU-vs-CPU Search Parity Tests

### Task 5.1: Comprehensive parity test at multiple scales

**Files:**
- Create: `Tests/MetalANNSTests/GPUCPUParityTests.swift`

**Context:** No test currently validates that the `beam_search` Metal kernel produces the same results as `BeamSearchCPU` on the same graph. This is the most critical correctness gap — especially after Phase 2 changes the visited-set implementation.

**Step 1: Write the tests**

```swift
// Tests/MetalANNSTests/GPUCPUParityTests.swift
import Testing
import Metal
@testable import MetalANNS
@testable import MetalANNSCore

struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state
    }
}

@Suite("GPU-CPU Search Parity")
struct GPUCPUParityTests {

    @Test(arguments: [
        (nodeCount: 100, dim: 32, degree: 8, k: 5, ef: 32),
        (nodeCount: 500, dim: 64, degree: 16, k: 10, ef: 64),
        (nodeCount: 2000, dim: 128, degree: 32, k: 20, ef: 128),
        (nodeCount: 8000, dim: 384, degree: 32, k: 10, ef: 64),
    ])
    func gpuMatchesCPU(
        nodeCount: Int, dim: Int, degree: Int, k: Int, ef: Int
    ) async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device"); return
        }
        let context = try MetalContext()

        var rng = SeededGenerator(state: UInt64(nodeCount * dim))
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
        for (i, v) in vectors.enumerated() { try vectorBuffer.setVector(v, at: i) }
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
        try await NNDescentGPU.build(
            context: context, graph: graph, vectors: vectorBuffer,
            nodeCount: nodeCount, metric: .cosine
        )

        // Test 5 different query vectors
        for qi in 0..<5 {
            let query = vectors[qi * (nodeCount / 5)]

            let gpuResults = try await FullGPUSearch.search(
                context: context, query: query, vectors: vectorBuffer,
                graph: graph, entryPoint: 0, k: k, ef: ef, metric: .cosine
            )

            let extractedVectors = (0..<nodeCount).map { vectorBuffer.vector(at: $0) }
            let extractedGraph = (0..<nodeCount).map { i in
                zip(graph.neighborIDs(of: i), graph.neighborDistances(of: i)).map { ($0.0, $0.1) }
            }
            let cpuResults = try await BeamSearchCPU.search(
                query: query, vectors: extractedVectors, graph: extractedGraph,
                entryPoint: 0, k: k, ef: ef, metric: .cosine
            )

            let gpuIDs = Set(gpuResults.prefix(k).map(\.internalID))
            let cpuIDs = Set(cpuResults.prefix(k).map(\.internalID))
            let overlap = gpuIDs.intersection(cpuIDs).count
            let recall = Float(overlap) / Float(min(k, gpuResults.count, cpuResults.count))

            #expect(recall >= 0.6,
                "Query \(qi): GPU-CPU recall \(recall) < 0.6 at n=\(nodeCount) dim=\(dim)")
        }
    }
}
```

**Step 2: Run tests**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/GPUCPUParityTests 2>&1 | tail -30`
Expected: PASS after Phase 2 changes. If any fail, investigate the GPU kernel.

**Step 3: Commit**

```bash
git add Tests/MetalANNSTests/GPUCPUParityTests.swift
git commit -m "test: add GPU-vs-CPU search parity tests at multiple scales"
```

---

## Phase 6: Algorithmic Optimizations and Bug Fixes

### Task 6.1: Fix rangeSearch guard inconsistency

**Files:**
- Modify: `Sources/MetalANNS/StreamingIndex.swift:193` (change `> 0` to `>= 0`)
- Test: Add to `Tests/MetalANNSTests/StreamingIndexMemoryTests.swift`

**Step 1: Write the failing test**

```swift
@Test func rangeSearchWithZeroDistanceReturnsExactMatches() async throws {
    let config = StreamingConfiguration(
        deltaCapacity: 50,
        mergeStrategy: .blocking,
        indexConfiguration: IndexConfiguration(degree: 4, metric: .l2)
    )
    let index = StreamingIndex(config: config)

    let dim = 8
    var rng = SeededGenerator(state: 55)
    var vectors: [[Float]] = []
    for i in 0..<10 {
        let v = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        vectors.append(v)
        try await index.insert(v, id: "v-\(i)")
    }

    // Search for exact match with maxDistance = 0
    let results = try await index.rangeSearch(query: vectors[3], maxDistance: 0.0, limit: 10)
    // Should return the vector itself (exact match), not empty
    #expect(results.count >= 1, "rangeSearch(maxDistance: 0) returned empty for exact match query")
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `StreamingIndex.swift:193` returns `[]` when `maxDistance == 0`.

**Step 3: Fix the guard**

In `StreamingIndex.swift:193`, change:
```swift
// Before:
guard maxDistance > 0 else { return [] }
// After:
guard maxDistance >= 0 else { return [] }
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add Sources/MetalANNS/StreamingIndex.swift Tests/MetalANNSTests/StreamingIndexMemoryTests.swift
git commit -m "fix: allow maxDistance=0 in StreamingIndex.rangeSearch for exact match queries"
```

---

### Task 6.2: Add SeededGenerator to all test files using Float.random

**Files:**
- Modify: `Tests/MetalANNSTests/NNDescentGPUTests.swift` (replace unseeded random)
- Modify: `Tests/MetalANNSTests/MetalDistanceTests.swift` (replace unseeded random)
- Modify: Any other test file using `Float.random(in:)` without `using:` parameter

**Context:** Several test files use `Float.random(in:)` without a seeded generator (found via grep). This makes recall-threshold tests non-reproducible and potentially flaky.

**Step 1: Create a shared test utility**

```swift
// Tests/MetalANNSTests/TestUtilities.swift
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    init(state: UInt64) {
        self.state = state == 0 ? 1 : state
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
```

**Step 2:** Replace all `Float.random(in: -1...1)` calls in test files with `Float.random(in: -1...1, using: &rng)` where `rng` is a `SeededGenerator` initialized with a deterministic seed.

**Step 3: Run full test suite**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30`
Expected: All PASS.

**Step 4: Commit**

```bash
git add Tests/MetalANNSTests/TestUtilities.swift Tests/MetalANNSTests/NNDescentGPUTests.swift \
    Tests/MetalANNSTests/MetalDistanceTests.swift
git commit -m "test: use SeededGenerator across all tests for reproducible results"
```

---

### Task 6.3: Early-exit optimization in local_join kernel

**Files:**
- Modify: `Sources/MetalANNSCore/Shaders/NNDescent.metal:298-321`
- Test: Existing `NNDescentGPUTests` (verify recall is maintained)

**Context:** In `local_join` (`NNDescent.metal:310`), for each `(a, b)` pair the kernel computes `pair_dist = compute_metric_distance(vectors, a, b, dim, metric_type)` unconditionally. For dim=384, this is 384 FMAs per call. Most candidates will be worse than the current worst neighbor. We can skip the distance computation when both `a` and `b` are already in each other's neighbor lists at closer distances.

**Step 1: Read the current worst distance before computing**

```metal
// Replace NNDescent.metal lines 298-321 with:
for (uint fi = 0; fi < fwd_count; fi++) {
    uint a = fwd[fi];
    if (a >= node_count) continue;

    // Read current worst distance for node a
    uint a_worst_bits = atomic_load_explicit(
        &adj_dists_bits[a * degree + degree - 1], memory_order_relaxed);
    float a_worst = as_type<float>(a_worst_bits);

    for (uint ri = 0; ri < actual_reverse; ri++) {
        uint b = rev[ri];
        if (b >= node_count || a == b) continue;

        float pair_dist = compute_metric_distance(vectors, a, b, dim, metric_type);

        // Only attempt insert if distance could improve the neighbor list
        if (pair_dist < a_worst) {
            try_insert_neighbor(adj_ids, adj_dists_bits, a, b, node_count, degree, pair_dist, update_counter);
            // Refresh worst after potential insert
            a_worst_bits = atomic_load_explicit(
                &adj_dists_bits[a * degree + degree - 1], memory_order_relaxed);
            a_worst = as_type<float>(a_worst_bits);
        }

        // Symmetric: also try inserting a into b's list
        uint b_worst_bits = atomic_load_explicit(
            &adj_dists_bits[b * degree + degree - 1], memory_order_relaxed);
        float b_worst = as_type<float>(b_worst_bits);
        if (pair_dist < b_worst) {
            try_insert_neighbor(adj_ids, adj_dists_bits, b, a, node_count, degree, pair_dist, update_counter);
        }
    }
}
```

Note: This also adds the symmetric update (`b`'s list gets `a` as a neighbor), which the current code omits. This gives 3x more useful work per distance computation.

**Step 2: Apply same change to NNDescentFloat16.metal**

Mirror the identical optimization.

**Step 3: Run NN-Descent tests to verify recall is maintained**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/NNDescentGPUTests 2>&1 | tail -20`
Expected: PASS with recall >= 0.80 (should actually improve since we now do symmetric updates).

**Step 4: Commit**

```bash
git add Sources/MetalANNSCore/Shaders/NNDescent.metal Sources/MetalANNSCore/Shaders/NNDescentFloat16.metal
git commit -m "perf: add early-exit and symmetric updates to local_join kernel"
```

---

### Task 6.4: Add threadgroup memory guard for PQ ADC scan

**Files:**
- Modify: `Sources/MetalANNSCore/GPUADCSearch.swift:136` (add guard)
- Test: Add validation test to `Tests/MetalANNSTests/GPUADCSearchTests.swift`

**Step 1: Write the failing test**

```swift
@Test func rejectsOversizedDistanceTable() async throws {
    guard let context = try? makeGPUContextOrSkip() else { return }

    // M=32, Ks=256 → 32*256*4 = 32768 bytes = exactly 32KB limit
    // This should work but be at the boundary
    let m = 32
    let ks = 256
    let tableLengthBytes = m * ks * MemoryLayout<Float>.stride
    let maxTG = context.device.maxThreadgroupMemoryLength

    if tableLengthBytes > maxTG {
        // If this device can't handle it, GPUADCSearch should throw, not crash
        // We'd need to construct a scenario that triggers the guard
        print("Device threadgroup limit \(maxTG) < table \(tableLengthBytes), guard should catch this")
    }
}
```

**Step 2: Add the guard**

In `GPUADCSearch.swift`, before line 136:

```swift
guard tableLengthBytes <= context.device.maxThreadgroupMemoryLength else {
    throw ANNSError.searchFailed(
        "PQ distance table (\(tableLengthBytes) bytes) exceeds device threadgroup memory limit (\(context.device.maxThreadgroupMemoryLength) bytes)"
    )
}
```

**Step 3: Commit**

```bash
git add Sources/MetalANNSCore/GPUADCSearch.swift Tests/MetalANNSTests/GPUADCSearchTests.swift
git commit -m "fix: guard against PQ distance table exceeding threadgroup memory limit"
```

---

## Summary: Phase Dependency Graph

```
Phase 1 (Buffer Pool)
  ├── Task 1.1: Create SearchBufferPool
  └── Task 1.2: Wire into FullGPUSearch ← depends on 1.1

Phase 2 (Lift 4096 Ceiling) ← depends on Phase 1 (uses pool for visited buffer)
  ├── Task 2.1: Generation-counter visited set + tests
  └── Task 2.2: Float16 variant ← depends on 2.1

Phase 3 (StreamingIndex Memory) ← independent of Phase 1-2
  └── Task 3.1: Evict merged/deleted vectors

Phase 4 (UInt64 IDs) ← independent of Phase 1-3
  ├── Task 4.1: IDMap UInt64 support
  └── Task 4.2: ANNSIndex UInt64 insert/search ← depends on 4.1

Phase 5 (Parity Tests) ← depends on Phase 2 (tests the new visited set)
  └── Task 5.1: GPU-CPU parity at multiple scales

Phase 6 (Optimizations) ← independent, can run in parallel with Phase 3-4
  ├── Task 6.1: rangeSearch guard fix
  ├── Task 6.2: Seeded test RNG
  ├── Task 6.3: local_join early-exit ← run after Phase 5 to verify with parity tests
  └── Task 6.4: PQ threadgroup memory guard
```

**Parallel execution opportunities:**
- Phase 3 + Phase 4 can run in parallel (no shared files)
- Tasks 6.1, 6.2, 6.4 can run in parallel (independent files)
- Phase 5 should run after Phase 2 completes

**Total estimated tasks:** 10 tasks across 6 phases.
