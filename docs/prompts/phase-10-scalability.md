# Phase 10: Scalability Primitives — Execution Prompt

> **Context**: You are implementing Phase 10 of MetalANNS, a GPU-native ANNS library for Apple Silicon.
> Phase 9 (Float16) is complete: 32 commits, 65 tests, branch `Phase-10`.
> This phase adds three scalability features: batch insert, hard deletion via compaction, and memory-mapped I/O.

---

## Build & Test Commands

```bash
# Build
xcodebuild build -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | tail -5

# Run ALL tests
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|PASS|FAIL|error:)'

# Run specific test file
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/BatchInsertTests 2>&1 | grep -E '(PASS|FAIL|error:)'
```

**Testing framework**: Swift Testing (`@Suite`, `@Test`, `#expect`). NOT XCTest.

---

## Task 28: Batch Insert

### Goal
Add `BatchIncrementalBuilder` that inserts multiple vectors more efficiently than N sequential `insert()` calls. Public API: `ANNSIndex.batchInsert(_:ids:)`.

### Files to Create
- `Sources/MetalANNSCore/BatchIncrementalBuilder.swift`
- `Tests/MetalANNSTests/BatchInsertTests.swift`

### Files to Modify
- `Sources/MetalANNS/ANNSIndex.swift` — add `batchInsert` method

### Current State of `IncrementalBuilder.insert()` (Sources/MetalANNSCore/IncrementalBuilder.swift)

The existing single-vector insert:
1. Validates bounds and dimension
2. Calls `nearestNeighbors()` — beam search on existing graph to find `degree` nearest neighbors
3. Sets the new node's neighbor list via `graph.setNeighbors()`
4. Reverse-updates: for each found neighbor, checks if new node is closer than their worst neighbor, replaces if so
5. Fallback: if no reverse-update succeeded, forces an edge from the entry point

### Algorithm for `BatchIncrementalBuilder`

```swift
// Sources/MetalANNSCore/BatchIncrementalBuilder.swift
import Foundation

public enum BatchIncrementalBuilder {
    /// Insert multiple vectors at once, more efficiently than sequential single inserts.
    /// 1. Insert all vectors into VectorStorage
    /// 2. For each new vector, find nearest neighbors via beam search on existing+previously-inserted nodes
    /// 3. Set each new node's neighbor list
    /// 4. Batch reverse-update: for each affected existing node, re-evaluate neighbor list
    public static func batchInsert(
        vectors newVectors: [[Float]],
        startingAt startSlot: Int,
        into graph: GraphBuffer,
        vectorStorage: any VectorStorage,
        entryPoint: UInt32,
        metric: Metric,
        degree: Int
    ) throws {
        // Implementation details below
    }
}
```

**Key insight**: After inserting vector[i], subsequent vectors[i+1..N] can find vector[i] as a neighbor. Process sequentially but defer all reverse-updates to one batch pass at the end.

**Step-by-step**:
1. For each new vector (in order), run beam search on the current graph to find `degree` nearest neighbors. Set that new node's neighbor list. Update `graph.nodeCount` after each.
2. After all forward neighbor lists are set, do a single batch reverse-update pass: for each new node, try to insert it into each of its neighbors' lists (replacing the worst neighbor if closer).

This avoids the N^2 cost of N individual inserts each doing their own reverse-update mid-stream, while still allowing later vectors to discover earlier ones.

### ANNSIndex.batchInsert API

```swift
// Add to ANNSIndex.swift
public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws {
    guard isBuilt, let vectorStorage = self.vectors, let graph else {
        throw ANNSError.indexEmpty
    }
    guard vectors.count == ids.count else {
        throw ANNSError.constructionFailed("Vector and ID counts do not match")
    }
    guard !vectors.isEmpty else { return }

    let dim = vectorStorage.dim
    for vector in vectors {
        guard vector.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
    }

    // Check for duplicate IDs (within batch and against existing)
    var seenIDs = Set<String>()
    for id in ids {
        if !seenIDs.insert(id).inserted {
            throw ANNSError.idAlreadyExists(id)
        }
        if idMap.internalID(for: id) != nil {
            throw ANNSError.idAlreadyExists(id)
        }
    }

    // Check capacity
    let startSlot = idMap.count
    guard startSlot + vectors.count <= vectorStorage.capacity,
          startSlot + vectors.count <= graph.capacity else {
        throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
    }

    // Assign IDs
    var slots: [Int] = []
    for id in ids {
        guard let assignedID = idMap.assign(externalID: id) else {
            throw ANNSError.idAlreadyExists(id)
        }
        slots.append(Int(assignedID))
    }

    // Insert vectors into storage
    for (i, vector) in vectors.enumerated() {
        try vectorStorage.insert(vector: vector, at: slots[i])
    }
    let newMaxCount = (slots.last ?? 0) + 1
    if vectorStorage.count < newMaxCount {
        vectorStorage.setCount(newMaxCount)
    }

    // Batch insert into graph
    try BatchIncrementalBuilder.batchInsert(
        vectors: vectors,
        startingAt: startSlot,
        into: graph,
        vectorStorage: vectorStorage,
        entryPoint: entryPoint,
        metric: configuration.metric,
        degree: configuration.degree
    )

    if graph.nodeCount < newMaxCount {
        graph.setCount(newMaxCount)
    }
}
```

### Tests

```swift
// Tests/MetalANNSTests/BatchInsertTests.swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Batch Insert Tests")
struct BatchInsertTests {

    @Test("Batch insert produces searchable results")
    func batchInsertRecall() async throws {
        // 1. Build index with 100 vectors (dim=32)
        // 2. Batch insert 50 more vectors
        // 3. Search for each inserted vector
        // 4. Verify recall >= 0.70 (each vector should find itself in top-k)
    }

    @Test("Batch insert matches sequential insert quality")
    func batchInsertMatchesSequential() async throws {
        // 1. Build two identical indexes with same 100 base vectors
        // 2. Index A: batchInsert 50 vectors
        // 3. Index B: insert same 50 vectors one by one
        // 4. Run same queries on both
        // 5. Verify batch recall is within 10% of sequential recall
    }
}
```

### Decision Point 28.1
**How to handle graph updates for batch?**
- **Option A (Recommended): Sequential forward, batched reverse** — Process each new vector's forward neighbors sequentially (so later vectors discover earlier ones), but defer all reverse-updates to one pass at the end. Simple, correct, still faster than N inserts.
- **Option B: Fully parallel** — All new vectors find neighbors in parallel on the pre-insert graph only. Simpler but new vectors can't discover each other. Lower quality.

**Default**: Option A

### Commit
```bash
git add Sources/MetalANNSCore/BatchIncrementalBuilder.swift Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/BatchInsertTests.swift
git commit -m "feat: add batch insert for efficient bulk addition"
```

---

## Task 29: Hard Deletion + Compaction

### Goal
Add `IndexCompactor` that rebuilds the index without soft-deleted nodes. Public API: `ANNSIndex.compact()`.

### Files to Create
- `Sources/MetalANNSCore/IndexCompactor.swift`
- `Tests/MetalANNSTests/CompactionTests.swift`

### Files to Modify
- `Sources/MetalANNSCore/SoftDeletion.swift` — add `allDeletedIDs` accessor
- `Sources/MetalANNSCore/IDMap.swift` — no changes needed (rebuild from scratch)
- `Sources/MetalANNS/ANNSIndex.swift` — add `compact` method

### Current State

**SoftDeletion** (Sources/MetalANNSCore/SoftDeletion.swift):
```swift
public struct SoftDeletion: Sendable, Codable {
    private var deletedIDs: Set<UInt32> = []
    public mutating func markDeleted(_ internalID: UInt32)
    public func isDeleted(_ internalID: UInt32) -> Bool
    public var deletedCount: Int
    public func filterResults(_ results: [SearchResult]) -> [SearchResult]
}
```
**Problem**: `deletedIDs` is private. Compaction needs to enumerate which IDs are deleted vs alive.

**Solution**: Add a public accessor:
```swift
/// All internal IDs currently marked as deleted.
public var allDeletedIDs: Set<UInt32> { deletedIDs }
```

**IDMap** (Sources/MetalANNSCore/IDMap.swift):
```swift
public struct IDMap: Sendable, Codable {
    private var externalToInternal: [String: UInt32] = [:]
    private var internalToExternal: [UInt32: String] = [:]
    private var nextID: UInt32 = 0
}
```
**Problem**: No remove or remap capability. Compaction needs contiguous internal IDs.

**Solution**: Rebuild IDMap from scratch during compaction. No modifications to IDMap needed.

### Algorithm for `IndexCompactor`

```swift
// Sources/MetalANNSCore/IndexCompactor.swift
import Foundation
import Metal

public enum IndexCompactor {
    public struct CompactionResult {
        public let vectors: any VectorStorage
        public let graph: GraphBuffer
        public let idMap: IDMap
        public let entryPoint: UInt32
    }

    /// Compact the index by:
    /// 1. Collect all non-deleted (internal ID, external ID) pairs
    /// 2. Create new VectorBuffer/GraphBuffer sized to surviving count
    /// 3. Copy surviving vectors to new buffer with contiguous IDs
    /// 4. Rebuild graph via NNDescentGPU (or CPU fallback) on the new compact vector set
    /// 5. Return new IDMap, new buffers, new entry point
    public static func compact(
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        softDeletion: SoftDeletion,
        metric: Metric,
        degree: Int,
        context: MetalContext?,
        maxIterations: Int,
        convergenceThreshold: Float,
        useFloat16: Bool
    ) async throws -> CompactionResult {
        // Implementation details below
    }
}
```

**Step-by-step**:
1. Enumerate all internal IDs from 0..<vectors.count
2. Filter out those in `softDeletion.allDeletedIDs`
3. Surviving IDs get remapped to 0..<survivingCount contiguously
4. Create new VectorStorage (Float16 or Float32 based on `useFloat16`) with capacity = `max(2, survivingCount * 2)`
5. Copy each surviving vector: `newVectors.insert(vector: oldVectors.vector(at: oldID), at: newID)`
6. Build new IDMap from surviving external IDs (in new order)
7. Rebuild graph using NNDescentGPU.build() (or NNDescentCPU fallback) on the new vector set
8. Run GraphPruner.prune() on the new graph
9. Return CompactionResult

### ANNSIndex.compact API

```swift
// Add to ANNSIndex.swift
public func compact() async throws {
    guard isBuilt, let vectors, let graph else {
        throw ANNSError.indexEmpty
    }
    guard softDeletion.deletedCount > 0 else {
        return // nothing to compact
    }

    let result = try await IndexCompactor.compact(
        vectors: vectors,
        graph: graph,
        idMap: idMap,
        softDeletion: softDeletion,
        metric: configuration.metric,
        degree: configuration.degree,
        context: context,
        maxIterations: configuration.maxIterations,
        convergenceThreshold: configuration.convergenceThreshold,
        useFloat16: configuration.useFloat16
    )

    self.vectors = result.vectors
    self.graph = result.graph
    self.idMap = result.idMap
    self.entryPoint = result.entryPoint
    self.softDeletion = SoftDeletion() // reset — no more deleted nodes
}
```

### Tests

```swift
// Tests/MetalANNSTests/CompactionTests.swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Compaction Tests")
struct CompactionTests {

    @Test("Compact reduces node count after deletions")
    func compactReducesMemory() async throws {
        // 1. Build index with 100 vectors
        // 2. Delete 50 vectors
        // 3. Verify count == 50 (soft deletion)
        // 4. Call compact()
        // 5. Verify count == 50 still
        // 6. Verify search still works for remaining vectors
        // 7. Verify deleted IDs throw idNotFound on search
    }

    @Test("Compact maintains search recall")
    func compactMaintainsRecall() async throws {
        // 1. Build index with 200 vectors (dim=32)
        // 2. Search for 10 queries, record baseline recall
        // 3. Delete 100 vectors (not the query vectors)
        // 4. Compact
        // 5. Search same queries
        // 6. Verify post-compaction recall >= 0.80 * baseline recall
    }
}
```

### Decision Point 29.1
**How to rebuild the graph during compaction?**
- **Option A (Recommended): Full NNDescentGPU rebuild** — Treat compaction as building a fresh index on the surviving vectors. Uses existing NNDescentGPU.build() + GraphPruner.prune(). Produces optimal graph quality. Cost: O(N * degree * iterations) but N is smaller post-deletion.
- **Option B: Remap existing graph** — Copy and remap neighbor IDs using old→new mapping. Faster but produces suboptimal graph (broken edges from deleted nodes become holes).

**Default**: Option A (quality > speed for compaction, which is an infrequent operation)

### Commit
```bash
git add Sources/MetalANNSCore/IndexCompactor.swift Sources/MetalANNSCore/SoftDeletion.swift Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/CompactionTests.swift
git commit -m "feat: add hard deletion via index compaction"
```

---

## Task 30: Memory-Mapped I/O

### Goal
Add `MmapIndexLoader` that loads an index file via `mmap()` and creates MTLBuffers via `makeBuffer(bytesNoCopy:)` for zero-copy GPU access. Public API: `ANNSIndex.loadMmap(from:)`.

### Files to Create
- `Sources/MetalANNSCore/MmapIndexLoader.swift`
- `Tests/MetalANNSTests/MmapTests.swift`

### Files to Modify
- `Sources/MetalANNSCore/IndexSerializer.swift` — add `saveMmapCompatible()` variant with page alignment
- `Sources/MetalANNS/ANNSIndex.swift` — add `loadMmap` static method

### Current IndexSerializer Format (v2)

```
Header (28 bytes):
  [0..3]   magic "MANN"
  [4..7]   version (2)
  [8..11]  nodeCount
  [12..15] degree
  [16..19] dim
  [20..23] metricCode
  [24..27] storageType (0=float32, 1=float16)

Body:
  [28..]   vectorData (nodeCount * dim * bytesPerElement)
  [..]     adjacencyData (nodeCount * degree * 4)
  [..]     distanceData (nodeCount * degree * 4)
  [..]     idMapByteCount (4 bytes) + idMapData
  [..]     entryPoint (4 bytes)
```

### Page Alignment Requirements

`makeBuffer(bytesNoCopy:)` requires the pointer to be page-aligned (4096 bytes on Apple Silicon). The solution:

1. **Save path**: After the 28-byte header, pad to the next page boundary (4096). Then write vector data starting at that page boundary. Similarly pad between sections.
2. **Load path**: Parse header, compute page-aligned offsets, create MTLBuffers pointing directly at the mmap'd regions.

### Mmap-Compatible File Format (v3)

```
Header (28 bytes):  same as v2
Padding:            zeros to reach 4096 byte boundary
Section 1:          vectorData (page-aligned start)
Padding:            zeros to reach next page boundary
Section 2:          adjacencyData (page-aligned start)
Padding:            zeros to reach next page boundary
Section 3:          distanceData (page-aligned start)
Padding:            zeros to reach next page boundary
Trailer:            idMapByteCount(4) + idMapData + entryPoint(4)
```

**Important**: The trailer (IDMap + entryPoint) is NOT page-aligned because it's read into CPU memory, not GPU buffers.

### Algorithm for `MmapIndexLoader`

```swift
// Sources/MetalANNSCore/MmapIndexLoader.swift
import Foundation
import Metal

public enum MmapIndexLoader {
    public struct MmapLoadResult {
        public let vectors: any VectorStorage
        public let graph: GraphBuffer
        public let idMap: IDMap
        public let entryPoint: UInt32
        public let metric: Metric
        // Keep reference to prevent deallocation
        let fileHandle: FileHandle
        let mappedData: UnsafeMutableRawPointer
        let mappedLength: Int
    }

    public static func load(from fileURL: URL, device: MTLDevice? = nil) throws -> MmapLoadResult {
        // 1. Open file, get file size
        // 2. mmap the entire file
        // 3. Parse 28-byte header from mmap'd memory
        // 4. Compute page-aligned offsets for each section
        // 5. Create MTLBuffer via device.makeBuffer(bytesNoCopy:length:options:deallocator:)
        //    - deallocator: nil (we manage the mmap lifetime via MmapLoadResult)
        // 6. Parse trailer for IDMap + entryPoint
        // 7. Wrap MTLBuffers in MmapVectorStorage/MmapGraphBuffer or reuse existing types
    }
}
```

**Key constraint**: `makeBuffer(bytesNoCopy:)` requires:
- Pointer must be page-aligned
- Length must be page-aligned (round up)
- `.storageModeShared` option
- deallocator manages lifetime

### Decision Point 30.1
**How to expose mmap'd buffers?**
- **Option A (Recommended): Wrapper types** — Create `MmapVectorBuffer` that conforms to `VectorStorage` but wraps a read-only MTLBuffer from mmap. The buffer is not owned by the wrapper (mmap owns it). The `insert()` method throws an error (read-only).
- **Option B: Reuse existing types** — Copy mmap'd data into regular VectorBuffer/GraphBuffer. Simpler but defeats the purpose of mmap (still allocates full memory).

**Default**: Option A. Create read-only wrappers.

### Decision Point 30.2
**Page-aligned save format?**
- **Option A (Recommended): New saveMmap method** — Add `IndexSerializer.saveMmapCompatible()` that writes with page padding. Bump version to 3. The existing `save()` and `load()` remain unchanged (v2 format).
- **Option B: Always page-align** — Change the existing save to always pad. Simpler but wastes space for small indexes.

**Default**: Option A. Keep v2 for normal save, add v3 for mmap-compatible save.

### ANNSIndex.loadMmap API

```swift
// Add to ANNSIndex.swift
public static func loadMmap(from url: URL) async throws -> ANNSIndex {
    let index = ANNSIndex()
    let device = await index.currentDevice()

    let loaded = try MmapIndexLoader.load(from: url, device: device)

    var resolvedConfiguration = IndexConfiguration.default
    resolvedConfiguration.metric = loaded.metric
    resolvedConfiguration.useFloat16 = loaded.vectors.isFloat16

    await index.applyLoadedState(
        configuration: resolvedConfiguration,
        vectors: loaded.vectors,
        graph: loaded.graph,
        idMap: loaded.idMap,
        entryPoint: loaded.entryPoint,
        softDeletion: SoftDeletion()
    )

    return index
}
```

Also add `saveMmapCompatible(to:)`:
```swift
public func saveMmapCompatible(to url: URL) async throws {
    guard isBuilt, let vectors, let graph else {
        throw ANNSError.indexEmpty
    }
    try IndexSerializer.saveMmapCompatible(
        vectors: vectors,
        graph: graph,
        idMap: idMap,
        entryPoint: entryPoint,
        metric: configuration.metric,
        to: url
    )
}
```

### Tests

```swift
// Tests/MetalANNSTests/MmapTests.swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Memory-Mapped I/O Tests")
struct MmapTests {

    @Test("Mmap load produces same search results as normal load")
    func mmapProducesSameResults() async throws {
        // 1. Build index with 200 vectors (dim=32)
        // 2. Save with saveMmapCompatible
        // 3. Load via loadMmap
        // 4. Load via normal load
        // 5. Run same 10 queries on both
        // 6. Verify results match exactly (same IDs, same order)
    }

    @Test("Mmap save and load roundtrip preserves all data")
    func mmapRoundtrip() async throws {
        // 1. Build index with 100 vectors (dim=16)
        // 2. saveMmapCompatible to temp file
        // 3. loadMmap from temp file
        // 4. Verify count matches
        // 5. Search for known vectors, verify they're found
    }
}
```

### IndexSerializer Changes

Add to `IndexSerializer`:
```swift
private static let mmapVersion: UInt32 = 3
private static let pageSize = 4096

public static func saveMmapCompatible(
    vectors: any VectorStorage,
    graph: GraphBuffer,
    idMap: IDMap,
    entryPoint: UInt32,
    metric: Metric,
    to fileURL: URL
) throws {
    // Write 28-byte header
    // Pad to 4096 boundary
    // Write vectorData, pad to 4096
    // Write adjacencyData, pad to 4096
    // Write distanceData, pad to 4096
    // Write trailer (idMapByteCount + idMapData + entryPoint)
}
```

Update `load()` to handle version 3:
```swift
guard formatVersion == 1 || formatVersion == 2 || formatVersion == version || formatVersion == mmapVersion else {
    throw ANNSError.corruptFile("Unsupported file version \(formatVersion)")
}
// If version 3, skip padding between sections
```

### Commit
```bash
git add Sources/MetalANNSCore/MmapIndexLoader.swift Sources/MetalANNSCore/IndexSerializer.swift Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/MmapTests.swift
git commit -m "feat: add memory-mapped index loading for large indices"
```

---

## Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `batchInsert` produces zero recall | Reverse-update pass skipped or broken | Check that batch reverse-update iterates all new nodes against their neighbors |
| Compaction crash on empty deletion set | Missing early-return guard | Add `guard softDeletion.deletedCount > 0` check |
| `makeBuffer(bytesNoCopy:)` returns nil | Pointer not page-aligned | Verify padding calculation: `(4096 - (cursor % 4096)) % 4096` |
| Mmap test fails with EXC_BAD_ACCESS | File handle closed before GPU reads | Ensure `MmapLoadResult` keeps `FileHandle` alive |
| Compaction changes recall dramatically | Graph rebuild quality varies | Use same configuration (degree, iterations, convergence) as original build |
| `idAlreadyExists` during batch insert | Duplicate detection within batch missing | Check `seenIDs` set for within-batch duplicates |

---

## Verification Checklist

After completing all three tasks:

```bash
# 1. All tests pass
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|PASS|FAIL|error:)'

# 2. Expected: 71+ tests (65 existing + 6 new), 0 failures

# 3. Verify batch insert test
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/BatchInsertTests 2>&1 | grep -E '(PASS|FAIL)'

# 4. Verify compaction test
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/CompactionTests 2>&1 | grep -E '(PASS|FAIL)'

# 5. Verify mmap test
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/MmapTests 2>&1 | grep -E '(PASS|FAIL)'

# 6. Verify no regressions
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/Float16IntegrationTests 2>&1 | grep -E '(PASS|FAIL)'
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/PersistenceTests 2>&1 | grep -E '(PASS|FAIL)'
```

---

## Summary

| Task | New Files | Modified Files | New Tests | Commit |
|------|-----------|----------------|-----------|--------|
| 28: Batch Insert | `BatchIncrementalBuilder.swift`, `BatchInsertTests.swift` | `ANNSIndex.swift` | 2 | `feat: add batch insert for efficient bulk addition` |
| 29: Compaction | `IndexCompactor.swift`, `CompactionTests.swift` | `ANNSIndex.swift`, `SoftDeletion.swift` | 2 | `feat: add hard deletion via index compaction` |
| 30: Mmap I/O | `MmapIndexLoader.swift`, `MmapTests.swift` | `ANNSIndex.swift`, `IndexSerializer.swift` | 2 | `feat: add memory-mapped index loading for large indices` |

**Expected end state**: 35 commits, 71+ tests, all passing.
