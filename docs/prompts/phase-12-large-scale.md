# Phase 12: Large-Scale (Combined Phase 11+12) — Execution Prompt

> **Context**: You are implementing Phase 12 of MetalANNS, a GPU-native ANNS library for Apple Silicon.
> Phase 10 (Scalability Primitives) is complete. **Phase 11 (Advanced Search) was NOT implemented** — its API surface is included here as prerequisite Tasks 31–33, followed by the original Phase 12 Tasks 34–35.
> This combined phase adds: filtered search, range search, runtime metric selection (Phase 11 APIs), then disk-backed index loading and sharded (IVF-style) partitioned search (Phase 12 features).

---

## Known Baseline Issue: MmapTests

`MmapTests` currently has a failing test (`mmapLoadedIndexIsReadOnly` or similar) related to an index-capacity issue on insert-after-load. **This is a pre-existing issue from Phase 10.** Do NOT attempt to fix it in this phase. Treat any MmapTests failure as baseline. Focus Phase 12 acceptance on new tests + non-regression of other existing tests.

---

## Build & Test Commands

```bash
# Build
xcodebuild build -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | tail -5

# Run ALL tests
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|PASS|FAIL|error:)'

# Run specific test file
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/FilteredSearchTests 2>&1 | grep -E '(PASS|FAIL|error:)'
```

**Testing framework**: Swift Testing (`@Suite`, `@Test`, `#expect`). NOT XCTest.

---

## Current Codebase State (Post Phase 10)

### ANNSIndex (Sources/MetalANNS/ANNSIndex.swift)

```swift
public actor ANNSIndex {
    private var configuration: IndexConfiguration
    private var context: MetalContext?
    private var vectors: (any VectorStorage)?
    private var graph: GraphBuffer?
    private var idMap: IDMap
    private var softDeletion: SoftDeletion
    // NO metadataStore — will be added in Task 31
    private var entryPoint: UInt32
    private var isBuilt: Bool
    private var isReadOnlyLoadedIndex: Bool
    private var mmapLifetime: AnyObject?

    // Current public API:
    // build(vectors:ids:), insert(_:id:), batchInsert(_:ids:), delete(id:), compact()
    // search(query:k:)  — NO filter, NO metric override
    // batchSearch(queries:k:)  — NO filter, NO metric override
    // save(to:), saveMmapCompatible(to:), load(from:), loadMmap(from:)
    // count (computed property)
    //
    // MISSING (to be added in Tasks 31-33):
    //   - metadataStore property
    //   - setMetadata() methods
    //   - filter: SearchFilter? parameter on search/batchSearch
    //   - metric: Metric? parameter on search/batchSearch
    //   - rangeSearch() method
    //   - MetadataStore in PersistedMetadata

    private struct PersistedMetadata: Codable, Sendable {
        let configuration: IndexConfiguration
        let softDeletion: SoftDeletion
        // NO metadataStore — will be added in Task 31
    }

    private func applyLoadedState(
        configuration: IndexConfiguration,
        vectors: any VectorStorage,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        softDeletion: SoftDeletion,
        isReadOnlyLoadedIndex: Bool = false,
        mmapLifetime: AnyObject? = nil
    ) { ... }
}
```

### VectorStorage Protocol (Sources/MetalANNSCore/VectorStorage.swift)

```swift
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

### GraphBuffer (Sources/MetalANNSCore/GraphBuffer.swift)

```swift
public final class GraphBuffer: @unchecked Sendable {
    public let adjacencyBuffer: MTLBuffer
    public let distanceBuffer: MTLBuffer
    public let degree: Int
    public let capacity: Int
    public private(set) var nodeCount: Int = 0

    // Regular init: allocates new buffers
    public init(capacity: Int, degree: Int, device: MTLDevice? = nil) throws

    // Mmap-compatible init: wraps pre-existing buffers
    public init(adjacencyBuffer: MTLBuffer, distanceBuffer: MTLBuffer, capacity: Int, degree: Int, nodeCount: Int) throws

    public func neighborIDs(of nodeID: Int) -> [UInt32]
    public func neighborDistances(of nodeID: Int) -> [Float]
}
```

### Search Architecture

- **GPU path**: `FullGPUSearch.search()` passes all buffers to a Metal kernel. `metricType` is already a runtime parameter (`buffer(11)`). No shader changes needed.
- **CPU path**: `BeamSearchCPU.search()` already accepts `metric: Metric`. ANNSIndex calls `extractVectors()` and `extractGraph()` which eagerly materialize all data.
- **SIMDDistance**: `SIMDDistance.distance(_ a: [Float], _ b: [Float], metric: Metric) -> Float` — used for centroid distance.

### IndexSerializer (Sources/MetalANNSCore/IndexSerializer.swift)

Binary formats:
- **v1**: 24-byte header (no storageType), then vectorData + adjacencyData + distanceData + idMap + entryPoint
- **v2**: 28-byte header (adds storageType), same layout as v1 after header
- **v3**: 28-byte header, page-aligned sections with padding between them

`IndexSerializer.load()` handles all three formats. `IndexSerializer.saveMmapCompatible()` writes v3.

### MmapIndexLoader (Sources/MetalANNSCore/MmapIndexLoader.swift)

- `MmapRegion` (private): Opens file, `mmap()`s, `munmap()`s on `deinit`
- `MmapVectorStorage` (private): Read-only `VectorStorage` wrapper over `MTLBuffer` via `makeBuffer(bytesNoCopy:)`
- Loads v3 (page-aligned) format files only

### SoftDeletion (Sources/MetalANNSCore/SoftDeletion.swift)

```swift
public struct SoftDeletion: Sendable, Codable {
    private var deletedIDs: Set<UInt32> = []
    public mutating func markDeleted(_ internalID: UInt32)
    public func isDeleted(_ internalID: UInt32) -> Bool
    public var deletedCount: Int
    public var allDeletedIDs: Set<UInt32>
    public func filterResults(_ results: [SearchResult]) -> [SearchResult]
}
```

### SearchResult (Sources/MetalANNSCore/SearchResult.swift)

```swift
public struct SearchResult: Sendable {
    public let id: String
    public let score: Float
    public let internalID: UInt32
}
```

### Metric (Sources/MetalANNSCore/Metric.swift)

```swift
public enum Metric: String, Sendable, Codable { case cosine, l2, innerProduct }
```

### IndexConfiguration (Sources/MetalANNS/IndexConfiguration.swift)

```swift
public struct IndexConfiguration: Sendable, Codable {
    public var degree: Int           // 32
    public var metric: Metric        // .cosine
    public var efConstruction: Int   // 100
    public var efSearch: Int         // 64
    public var maxIterations: Int    // 20
    public var useFloat16: Bool      // false
    public var convergenceThreshold: Float // 0.001
}
```

---

## Task 31: Filtered Search with Metadata Predicates

### Goal
Add a `MetadataStore` for per-vector metadata (string, float, int columns) and a `SearchFilter` predicate DSL. Support filtered search via post-filter with over-fetch.

### Files to Create
- `Sources/MetalANNSCore/MetadataStore.swift`
- `Sources/MetalANNSCore/SearchFilter.swift`
- `Tests/MetalANNSTests/FilteredSearchTests.swift`

### Files to Modify
- `Sources/MetalANNS/ANNSIndex.swift` — add metadata methods + filtered search

### SearchFilter Design

Recursive enum for composable predicates:

```swift
// Sources/MetalANNSCore/SearchFilter.swift
import Foundation

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

**Note**: `SearchFilter` must be `Sendable`. Since all cases use `String`, `Float`, `Set<String>`, and recursive `[SearchFilter]` — all of which are `Sendable` — this works automatically. Do NOT make it `Codable` unless needed.

### MetadataStore Design

Column-oriented storage. Each column is a dictionary from internalID to value. Three value types: String, Float, Int64.

```swift
// Sources/MetalANNSCore/MetadataStore.swift
import Foundation

public struct MetadataStore: Sendable, Codable {
    private var stringColumns: [String: [UInt32: String]] = [:]
    private var floatColumns: [String: [UInt32: Float]] = [:]
    private var intColumns: [String: [UInt32: Int64]] = [:]

    public init() {}

    // MARK: - Setters

    public mutating func set(_ column: String, stringValue: String, for id: UInt32) {
        stringColumns[column, default: [:]][id] = stringValue
    }

    public mutating func set(_ column: String, floatValue: Float, for id: UInt32) {
        floatColumns[column, default: [:]][id] = floatValue
    }

    public mutating func set(_ column: String, intValue: Int64, for id: UInt32) {
        intColumns[column, default: [:]][id] = intValue
    }

    // MARK: - Getters

    public func getString(_ column: String, for id: UInt32) -> String? {
        stringColumns[column]?[id]
    }

    public func getFloat(_ column: String, for id: UInt32) -> Float? {
        floatColumns[column]?[id]
    }

    public func getInt(_ column: String, for id: UInt32) -> Int64? {
        intColumns[column]?[id]
    }

    // MARK: - Filter evaluation

    public func matches(id: UInt32, filter: SearchFilter) -> Bool {
        switch filter {
        case .equals(let column, let value):
            return stringColumns[column]?[id] == value
        case .greaterThan(let column, let value):
            if let floatVal = floatColumns[column]?[id] { return floatVal > value }
            if let intVal = intColumns[column]?[id] { return Float(intVal) > value }
            return false
        case .lessThan(let column, let value):
            if let floatVal = floatColumns[column]?[id] { return floatVal < value }
            if let intVal = intColumns[column]?[id] { return Float(intVal) < value }
            return false
        case .in(let column, let values):
            guard let val = stringColumns[column]?[id] else { return false }
            return values.contains(val)
        case .and(let filters):
            return filters.allSatisfy { matches(id: id, filter: $0) }
        case .or(let filters):
            return filters.contains { matches(id: id, filter: $0) }
        case .not(let inner):
            return !matches(id: id, filter: inner)
        }
    }

    // MARK: - Compaction support

    public func remapped(using mapping: [UInt32: UInt32]) -> MetadataStore {
        var result = MetadataStore()
        for (col, values) in stringColumns {
            for (oldID, val) in values {
                if let newID = mapping[oldID] {
                    result.stringColumns[col, default: [:]][newID] = val
                }
            }
        }
        for (col, values) in floatColumns {
            for (oldID, val) in values {
                if let newID = mapping[oldID] {
                    result.floatColumns[col, default: [:]][newID] = val
                }
            }
        }
        for (col, values) in intColumns {
            for (oldID, val) in values {
                if let newID = mapping[oldID] {
                    result.intColumns[col, default: [:]][newID] = val
                }
            }
        }
        return result
    }

    public mutating func remove(id: UInt32) {
        for col in stringColumns.keys { stringColumns[col]?[id] = nil }
        for col in floatColumns.keys { floatColumns[col]?[id] = nil }
        for col in intColumns.keys { intColumns[col]?[id] = nil }
    }

    public var isEmpty: Bool {
        stringColumns.isEmpty && floatColumns.isEmpty && intColumns.isEmpty
    }
}
```

### ANNSIndex Changes for Task 31

Add to ANNSIndex:

```swift
// New private property (add alongside softDeletion)
private var metadataStore: MetadataStore

// In init():
self.metadataStore = MetadataStore()

// New public methods:
public func setMetadata(_ column: String, value: String, for id: String) throws {
    guard let internalID = idMap.internalID(for: id) else {
        throw ANNSError.idNotFound(id)
    }
    metadataStore.set(column, stringValue: value, for: internalID)
}

public func setMetadata(_ column: String, value: Float, for id: String) throws {
    guard let internalID = idMap.internalID(for: id) else {
        throw ANNSError.idNotFound(id)
    }
    metadataStore.set(column, floatValue: value, for: internalID)
}

public func setMetadata(_ column: String, value: Int64, for id: String) throws {
    guard let internalID = idMap.internalID(for: id) else {
        throw ANNSError.idNotFound(id)
    }
    metadataStore.set(column, intValue: value, for: internalID)
}

// Modified search signature (backward compatible via default nil):
public func search(query: [Float], k: Int, filter: SearchFilter? = nil) async throws -> [SearchResult] {
    // ... existing guards ...

    let hasFilter = filter != nil
    let deletedCount = softDeletion.deletedCount
    let effectiveK: Int
    if hasFilter {
        effectiveK = min(vectors.count, k * 4 + deletedCount)  // over-fetch for filter
    } else {
        effectiveK = min(vectors.count, k + deletedCount)
    }
    let effectiveEf = max(configuration.efSearch, effectiveK)

    // ... existing beam search dispatch (GPU or CPU) ...

    var filtered = softDeletion.filterResults(rawResults)

    // Apply metadata filter
    if let filter {
        filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
    }

    let mapped = filtered.compactMap { result -> SearchResult? in
        guard let externalID = idMap.externalID(for: result.internalID) else { return nil }
        return SearchResult(id: externalID, score: result.score, internalID: result.internalID)
    }
    return Array(mapped.prefix(k))
}
```

### Persistence: MetadataStore in Save/Load

Update `PersistedMetadata`:
```swift
private struct PersistedMetadata: Codable, Sendable {
    let configuration: IndexConfiguration
    let softDeletion: SoftDeletion
    let metadataStore: MetadataStore?  // Optional for backward compat
}
```

In `save()`: include `metadataStore` in the JSON.
In `load()`: read `metadataStore` from JSON (default to empty if nil for backward compat).

Also update `applyLoadedState` to accept `metadataStore`:
```swift
private func applyLoadedState(
    configuration: IndexConfiguration,
    vectors: any VectorStorage,
    graph: GraphBuffer,
    idMap: IDMap,
    entryPoint: UInt32,
    softDeletion: SoftDeletion,
    metadataStore: MetadataStore = MetadataStore(),
    isReadOnlyLoadedIndex: Bool = false,
    mmapLifetime: AnyObject? = nil
) { ... }
```

**No changes to IndexSerializer's binary format.** Metadata lives in the `.meta.json` sidecar file.

### Tests

```swift
// Tests/MetalANNSTests/FilteredSearchTests.swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Filtered Search Tests")
struct FilteredSearchTests {

    @Test("Filtered search returns only matching results")
    func filteredSearchReturnsOnlyMatching() async throws {
        // 1. Build index with 100 vectors (dim=16), IDs "v0"..."v99"
        // 2. Set category metadata: "v0"..."v49" → category="A", "v50"..."v99" → category="B"
        // 3. Search with filter: .equals(column: "category", value: "A")
        // 4. Verify ALL returned results have IDs in "v0"..."v49"
        // 5. Verify no result has ID in "v50"..."v99"
    }

    @Test("Filtered search maintains reasonable recall")
    func filteredSearchRecall() async throws {
        // 1. Build index with 200 vectors (dim=32), IDs "v0"..."v199"
        // 2. Set category: even IDs → "even", odd IDs → "odd"
        // 3. Use even-indexed vectors as queries
        // 4. Search with k=5, filter: .equals(column: "category", value: "even")
        // 5. Compute recall: for each query, does the query vector appear in results?
        // 6. Expect recall >= 0.50
    }

    @Test("Compound filters work correctly")
    func compoundFilterWorks() async throws {
        // 1. Build index with 100 vectors (dim=16)
        // 2. Set metadata: category (string) and score (float)
        //    "v0"..."v24": category="A", score=1.0
        //    "v25"..."v49": category="A", score=5.0
        //    "v50"..."v74": category="B", score=1.0
        //    "v75"..."v99": category="B", score=5.0
        // 3. Test AND: .and([.equals(column: "category", value: "A"), .greaterThan(column: "score", value: 3.0)])
        //    → only "v25"..."v49"
        // 4. Test OR: .or([.equals(column: "category", value: "A"), .greaterThan(column: "score", value: 3.0)])
        //    → "v0"..."v49" + "v75"..."v99"
        // 5. Test NOT: .not(.equals(column: "category", value: "A"))
        //    → only "v50"..."v99"
    }
}
```

### Commit
```bash
git add Sources/MetalANNSCore/MetadataStore.swift Sources/MetalANNSCore/SearchFilter.swift Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/FilteredSearchTests.swift
git commit -m "feat: add filtered search with metadata predicates"
```

---

## Task 32: Range Search

### Goal
Add range search: return all vectors within a given distance threshold, up to a limit.

### Files to Create
- `Tests/MetalANNSTests/RangeSearchTests.swift`

### Files to Modify
- `Sources/MetalANNS/ANNSIndex.swift` — add `rangeSearch` method

### ANNSIndex.rangeSearch API

```swift
public func rangeSearch(
    query: [Float],
    maxDistance: Float,
    limit: Int = 1000,
    filter: SearchFilter? = nil
) async throws -> [SearchResult] {
    guard isBuilt, let vectors, let graph else {
        throw ANNSError.indexEmpty
    }
    guard query.count == vectors.dim else {
        throw ANNSError.dimensionMismatch(expected: vectors.dim, got: query.count)
    }
    guard maxDistance > 0 else { return [] }
    guard limit > 0 else { return [] }

    let deletedCount = softDeletion.deletedCount
    let searchK = min(vectors.count, limit + deletedCount)
    let searchEf = min(vectors.count, max(configuration.efSearch, searchK * 2))

    let rawResults: [SearchResult]
    if let context {
        rawResults = try await FullGPUSearch.search(
            context: context, query: query, vectors: vectors, graph: graph,
            entryPoint: Int(entryPoint), k: max(1, searchK),
            ef: max(1, searchEf), metric: configuration.metric
        )
    } else {
        rawResults = try await BeamSearchCPU.search(
            query: query, vectors: extractVectors(from: vectors),
            graph: extractGraph(from: graph), entryPoint: Int(entryPoint),
            k: max(1, searchK), ef: max(1, searchEf), metric: configuration.metric
        )
    }

    var filtered = softDeletion.filterResults(rawResults)
    if let filter {
        filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
    }
    let withinRange = filtered.filter { $0.score <= maxDistance }

    let mapped = withinRange.compactMap { result -> SearchResult? in
        guard let externalID = idMap.externalID(for: result.internalID) else { return nil }
        return SearchResult(id: externalID, score: result.score, internalID: result.internalID)
    }
    return Array(mapped.prefix(limit))
}
```

### Tests

```swift
// Tests/MetalANNSTests/RangeSearchTests.swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Range Search Tests")
struct RangeSearchTests {

    @Test("Range search returns only results within threshold")
    func rangeSearchReturnsWithinThreshold() async throws {
        // 1. Build index with 100 vectors (dim=16, cosine metric)
        // 2. Pick a query vector
        // 3. Range search with maxDistance = 0.5
        // 4. Verify ALL returned results have score <= 0.5
        // 5. Verify count > 0
    }

    @Test("Range search finds exact match with tight threshold")
    func rangeSearchFindsExactMatch() async throws {
        // 1. Build index with 50 vectors (dim=16, cosine metric)
        // 2. Insert a known vector with ID "target"
        // 3. Range search using that exact vector as query, maxDistance = 0.01
        // 4. Verify "target" is in the results
    }
}
```

### Commit
```bash
git add Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/RangeSearchTests.swift
git commit -m "feat: add range search with distance threshold"
```

---

## Task 33: Runtime Metric Selection

### Goal
Allow search queries to override the metric used during search. The graph was built with one metric, so search with a different metric will have degraded recall — but the distances will be correct for the chosen metric.

### Files to Modify
- `Sources/MetalANNS/ANNSIndex.swift` — add optional `metric` parameter to `search()`, `rangeSearch()`, `batchSearch()`

### No Shader Changes Needed

The GPU beam search kernel already accepts `metricType` as a runtime parameter. `BeamSearchCPU.search()` already accepts `metric: Metric`. The only change is in ANNSIndex — the public API needs an optional metric override.

### ANNSIndex Changes

Update `search` (already modified in Task 31 to add `filter`):

```swift
public func search(
    query: [Float],
    k: Int,
    filter: SearchFilter? = nil,
    metric: Metric? = nil        // NEW: optional runtime metric override
) async throws -> [SearchResult] {
    let searchMetric = metric ?? configuration.metric
    // Pass searchMetric instead of configuration.metric to GPU/CPU search
}
```

Similarly for `rangeSearch`:
```swift
public func rangeSearch(
    query: [Float],
    maxDistance: Float,
    limit: Int = 1000,
    filter: SearchFilter? = nil,
    metric: Metric? = nil        // NEW
) async throws -> [SearchResult] {
    let searchMetric = metric ?? configuration.metric
    // use searchMetric in search dispatch
}
```

And `batchSearch`:
```swift
public func batchSearch(
    queries: [[Float]],
    k: Int,
    filter: SearchFilter? = nil,
    metric: Metric? = nil        // NEW
) async throws -> [[SearchResult]] {
    // calls self.search(query:k:filter:metric:) internally
}
```

### Tests

```swift
// Tests/MetalANNSTests/RuntimeMetricTests.swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Runtime Metric Selection Tests")
struct RuntimeMetricTests {

    @Test("Search with different metric returns valid results")
    func searchWithDifferentMetric() async throws {
        // 1. Build index with 100 vectors (dim=16), metric: .cosine
        // 2. Search with metric: .l2
        // 3. Verify results are non-empty
        // 4. Verify results are sorted by L2 distance (ascending)
        // 5. Verify distances are valid L2 values (non-negative)
    }

    @Test("Default metric matches build metric")
    func defaultMetricMatchesBuildMetric() async throws {
        // 1. Build index with 100 vectors (dim=16), metric: .cosine
        // 2. Search with metric: nil (default)
        // 3. Search with metric: .cosine (explicit)
        // 4. Verify both return identical results (same IDs, same scores)
    }
}
```

### Commit
```bash
git add Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/RuntimeMetricTests.swift
git commit -m "feat: support runtime metric selection at query time"
```

---

## Task 34: Disk-Backed Index

### Goal
Add `DiskBackedVectorBuffer` that serves vector data from mmap'd files on demand. Unlike `loadMmap` (which requires v3 page-aligned format and creates GPU-accessible `bytesNoCopy` buffers), the disk-backed approach works with any saved index file, reads vectors on demand from mmap, and avoids loading the entire vector set into RAM. Graph data is loaded into regular `GraphBuffer` (much smaller than vectors).

### Key Difference from loadMmap

| | `loadMmap` (Phase 10) | `loadDiskBacked` (Phase 12) |
|---|---|---|
| Format | v3 only (page-aligned) | v1, v2, or v3 |
| GPU buffers | `bytesNoCopy` (zero-copy) | Regular MTLBuffer placeholder (CPU-only search) |
| Memory usage | OS pages in whole mmap sections | Only materializes individual vectors on access via LRU cache |
| CPU search | Still extracts all data | Reads vectors on demand |
| Mutability | Read-only | Read-only |
| Use case | Fast GPU search on large indices | Memory-constrained devices (iOS) with any format |

For a 1M vector index with dim=128, degree=32:
- Vectors: 1M × 128 × 4 = 512 MB (**kept on disk**, read on demand via LRU cache)
- Graph adjacency: 1M × 32 × 4 = 128 MB (loaded into RAM — acceptable)
- Graph distances: 1M × 32 × 4 = 128 MB (loaded into RAM)

So the disk-backed approach saves ~512 MB by keeping vectors on disk.

### Files to Create
- `Sources/MetalANNSCore/DiskBackedVectorBuffer.swift` (includes `DiskBackedIndexLoader`)
- `Tests/MetalANNSTests/DiskBackedTests.swift`

### Files to Modify
- `Sources/MetalANNS/ANNSIndex.swift` — add `loadDiskBacked(from:)` static method

### DiskBackedVectorBuffer Design

```swift
// Sources/MetalANNSCore/DiskBackedVectorBuffer.swift
import Foundation
import Metal

/// Disk-backed vector storage that reads vectors on demand from an mmap'd region.
/// Conforms to VectorStorage but is read-only.
/// Uses a simple LRU cache to avoid repeated page-in for hot vectors.
public final class DiskBackedVectorBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer  // Minimal placeholder for protocol conformance
    public let dim: Int
    public let capacity: Int
    public private(set) var count: Int
    public let isFloat16: Bool

    private let mmapPointer: UnsafeRawPointer
    private let dataOffset: Int   // byte offset to vector data within the mmap'd file
    private let bytesPerVector: Int

    // Simple LRU cache
    private var cache: [Int: [Float]] = [:]
    private var cacheOrder: [Int] = []
    private let cacheCapacity: Int

    public init(
        mmapPointer: UnsafeRawPointer,
        dataOffset: Int,
        dim: Int,
        count: Int,
        isFloat16: Bool,
        device: MTLDevice,
        cacheCapacity: Int = 1024
    ) throws {
        self.dim = dim
        self.capacity = count
        self.count = count
        self.isFloat16 = isFloat16
        self.mmapPointer = mmapPointer
        self.dataOffset = dataOffset
        self.bytesPerVector = dim * (isFloat16 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride)
        self.cacheCapacity = cacheCapacity

        guard let buf = device.makeBuffer(length: max(4, bytesPerVector), options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate staging buffer")
        }
        self.buffer = buf
    }

    private func readVector(at index: Int) -> [Float] {
        let byteOffset = dataOffset + index * bytesPerVector
        if isFloat16 {
            let ptr = mmapPointer.advanced(by: byteOffset).assumingMemoryBound(to: UInt16.self)
            var result = [Float](repeating: 0, count: dim)
            for d in 0..<dim {
                result[d] = Float(Float16(bitPattern: ptr[d]))
            }
            return result
        } else {
            let ptr = mmapPointer.advanced(by: byteOffset).assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: ptr, count: dim))
        }
    }

    private func cachedVector(at index: Int) -> [Float] {
        if let cached = cache[index] {
            if let orderIndex = cacheOrder.firstIndex(of: index) {
                cacheOrder.remove(at: orderIndex)
            }
            cacheOrder.append(index)
            return cached
        }

        let vector = readVector(at: index)
        cache[index] = vector
        cacheOrder.append(index)

        if cacheOrder.count > cacheCapacity {
            let evicted = cacheOrder.removeFirst()
            cache[evicted] = nil
        }
        return vector
    }
}

extension DiskBackedVectorBuffer: VectorStorage {
    public func setCount(_ newCount: Int) {
        // No-op for disk-backed (read-only, count is fixed)
    }

    public func insert(vector: [Float], at index: Int) throws {
        throw ANNSError.constructionFailed("Disk-backed vectors are read-only")
    }

    public func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        throw ANNSError.constructionFailed("Disk-backed vectors are read-only")
    }

    public func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < count, "Index out of bounds")
        return cachedVector(at: index)
    }
}
```

### DiskBackedIndexLoader

Add to `DiskBackedVectorBuffer.swift` (same file):

```swift
public enum DiskBackedIndexLoader {
    public struct LoadResult {
        public let vectors: DiskBackedVectorBuffer
        public let graph: GraphBuffer
        public let idMap: IDMap
        public let entryPoint: UInt32
        public let metric: Metric
        public let mmapLifetime: AnyObject
    }

    /// Load an index file with disk-backed vectors.
    /// Graph data is loaded into RAM (smaller), vectors remain on disk (larger).
    /// Works with v1, v2, and v3 format files.
    public static func load(from fileURL: URL, device: MTLDevice? = nil) throws -> LoadResult {
        // 1. mmap the file (reuse MmapRegion pattern — create a local MmapRegion class
        //    or extract it. Simplest: duplicate the small MmapRegion class here)
        // 2. Parse header (same as IndexSerializer):
        //    - Read magic, version, nodeCount, degree, dim, metricCode
        //    - Read storageType (for version >= 2, else default 0)
        // 3. Compute offsets for vector/adjacency/distance sections:
        //    - For v3: account for page padding (same alignedOffset math)
        //    - For v1/v2: sections are contiguous after header
        // 4. Create DiskBackedVectorBuffer pointing at vector data offset in mmap
        // 5. Create regular GraphBuffer, copy adjacency + distance data from mmap into it
        // 6. Parse trailer for IDMap + entryPoint (after distance section)
        // 7. Return LoadResult with MmapRegion as mmapLifetime
    }
}
```

**Important**: The `MmapRegion` class from `MmapIndexLoader.swift` is private. Either:
- Duplicate a small mmap helper in `DiskBackedVectorBuffer.swift`, or
- Extract `MmapRegion` to its own public file

The simplest approach: duplicate a minimal `DiskBackedMmapRegion` class (~20 lines).

### ANNSIndex.loadDiskBacked

```swift
// Add to ANNSIndex.swift
public static func loadDiskBacked(from url: URL) async throws -> ANNSIndex {
    let persistedMetadata = try loadPersistedMetadataIfPresent(from: url)
    let initialConfiguration = persistedMetadata?.configuration ?? .default
    let index = ANNSIndex(configuration: initialConfiguration)

    let diskBacked = try DiskBackedIndexLoader.load(from: url, device: await index.currentDevice())

    var resolvedConfiguration = persistedMetadata?.configuration ?? .default
    resolvedConfiguration.metric = diskBacked.metric
    resolvedConfiguration.useFloat16 = diskBacked.vectors.isFloat16

    await index.applyLoadedState(
        configuration: resolvedConfiguration,
        vectors: diskBacked.vectors,
        graph: diskBacked.graph,
        idMap: diskBacked.idMap,
        entryPoint: diskBacked.entryPoint,
        softDeletion: persistedMetadata?.softDeletion ?? SoftDeletion(),
        metadataStore: persistedMetadata?.metadataStore ?? MetadataStore(),
        isReadOnlyLoadedIndex: true,
        mmapLifetime: diskBacked.mmapLifetime
    )

    return index
}
```

### Tests

```swift
// Tests/MetalANNSTests/DiskBackedTests.swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Disk-Backed Index Tests")
struct DiskBackedTests {

    @Test("Disk-backed search produces correct results")
    func diskBackedSearchWorks() async throws {
        // 1. Build index with 200 vectors (dim=32)
        // 2. Save to temp file (normal v2 format)
        // 3. Load via ANNSIndex.loadDiskBacked(from:)
        // 4. Also load via normal ANNSIndex.load(from:)
        // 5. Run same 10 queries on both
        // 6. Verify disk-backed results match normal results
    }

    @Test("Disk-backed load works with v3 mmap format")
    func diskBackedWorksWithV3() async throws {
        // 1. Build index with 100 vectors (dim=16)
        // 2. Save with saveMmapCompatible (v3 format)
        // 3. Load via loadDiskBacked
        // 4. Search and verify results are valid
    }
}
```

### Commit
```bash
git add Sources/MetalANNSCore/DiskBackedVectorBuffer.swift Sources/MetalANNS/ANNSIndex.swift Tests/MetalANNSTests/DiskBackedTests.swift
git commit -m "feat: add disk-backed index for memory-constrained devices"
```

---

## Task 35: Sharded Indices (IVF-Style)

### Goal
Add `ShardedIndex` that partitions vectors across multiple `ANNSIndex` shards via k-means clustering. At search time, only the top-`nprobe` shards are searched, providing sub-linear scaling.

### Files to Create
- `Sources/MetalANNSCore/KMeans.swift`
- `Sources/MetalANNS/ShardedIndex.swift`
- `Tests/MetalANNSTests/ShardedIndexTests.swift`

### Decision Points

**35.1: Should ShardedIndex support save/load?**
- **Option A (Recommended): No persistence in v1** — ShardedIndex is rebuilt each time. Defer to future phase.

**35.2: Should ShardedIndex support insert/delete?**
- **Option A (Recommended): Build-only in v1** — No incremental insert or delete. Users rebuild when data changes.

### KMeans Implementation

Simple Lloyd's algorithm with k-means++ init. CPU-only.

```swift
// Sources/MetalANNSCore/KMeans.swift
import Foundation
import Accelerate

public enum KMeans {
    public struct Result {
        public let centroids: [[Float]]
        public let assignments: [Int]
    }

    public static func cluster(
        vectors: [[Float]],
        k: Int,
        maxIterations: Int = 20,
        metric: Metric = .l2,
        seed: UInt64 = 42
    ) throws -> Result {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot cluster empty vectors")
        }
        guard k > 0, k <= vectors.count else {
            throw ANNSError.constructionFailed("k must be between 1 and vector count")
        }

        let dim = vectors[0].count
        for v in vectors where v.count != dim {
            throw ANNSError.dimensionMismatch(expected: dim, got: v.count)
        }

        var centroids = initializeCentroids(vectors: vectors, k: k, metric: metric, seed: seed)
        var assignments = [Int](repeating: 0, count: vectors.count)

        for _ in 0..<maxIterations {
            var changed = false
            for i in 0..<vectors.count {
                var bestCluster = 0
                var bestDistance = Float.greatestFiniteMagnitude
                for c in 0..<k {
                    let dist = SIMDDistance.distance(vectors[i], centroids[c], metric: metric)
                    if dist < bestDistance {
                        bestDistance = dist
                        bestCluster = c
                    }
                }
                if assignments[i] != bestCluster {
                    assignments[i] = bestCluster
                    changed = true
                }
            }

            if !changed { break }

            var sums = [[Float]](repeating: [Float](repeating: 0, count: dim), count: k)
            var counts = [Int](repeating: 0, count: k)
            for i in 0..<vectors.count {
                let cluster = assignments[i]
                counts[cluster] += 1
                for d in 0..<dim {
                    sums[cluster][d] += vectors[i][d]
                }
            }
            for c in 0..<k {
                if counts[c] > 0 {
                    let scale = 1.0 / Float(counts[c])
                    for d in 0..<dim {
                        centroids[c][d] = sums[c][d] * scale
                    }
                }
                // Empty clusters keep their previous centroid
            }
        }

        return Result(centroids: centroids, assignments: assignments)
    }

    /// K-means++ initialization
    private static func initializeCentroids(
        vectors: [[Float]],
        k: Int,
        metric: Metric,
        seed: UInt64
    ) -> [[Float]] {
        var rng = RandomNumberGenerator64(seed: seed)
        var centroids: [[Float]] = []

        let firstIndex = Int.random(in: 0..<vectors.count, using: &rng)
        centroids.append(vectors[firstIndex])

        for _ in 1..<k {
            var distances = [Float](repeating: 0, count: vectors.count)
            var totalDistance: Float = 0
            for i in 0..<vectors.count {
                var minDist = Float.greatestFiniteMagnitude
                for centroid in centroids {
                    let dist = SIMDDistance.distance(vectors[i], centroid, metric: metric)
                    minDist = min(minDist, dist)
                }
                distances[i] = minDist
                totalDistance += minDist
            }

            if totalDistance <= 0 {
                let idx = Int.random(in: 0..<vectors.count, using: &rng)
                centroids.append(vectors[idx])
            } else {
                var threshold = Float.random(in: 0..<totalDistance, using: &rng)
                var picked = vectors.count - 1
                for i in 0..<vectors.count {
                    threshold -= distances[i]
                    if threshold <= 0 {
                        picked = i
                        break
                    }
                }
                centroids.append(vectors[picked])
            }
        }

        return centroids
    }
}

/// Simple seeded RNG for reproducibility.
private struct RandomNumberGenerator64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 1 : seed
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
```

### ShardedIndex Design

```swift
// Sources/MetalANNS/ShardedIndex.swift
import Foundation
import MetalANNSCore

public actor ShardedIndex {
    private let numShards: Int
    private let nprobe: Int
    private let configuration: IndexConfiguration
    private var centroids: [[Float]] = []
    private var shards: [ANNSIndex] = []
    private var shardIDMappings: [[String]] = []
    private var isBuilt: Bool = false

    public init(
        numShards: Int = 16,
        nprobe: Int = 4,
        configuration: IndexConfiguration = .default
    ) {
        self.numShards = max(1, numShards)
        self.nprobe = max(1, min(nprobe, numShards))
        self.configuration = configuration
    }

    public func build(vectors: [[Float]], ids: [String]) async throws {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot build sharded index with empty vectors")
        }
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }

        let effectiveShards = min(numShards, vectors.count)

        let kmResult = try KMeans.cluster(
            vectors: vectors,
            k: effectiveShards,
            maxIterations: 20,
            metric: configuration.metric
        )

        var shardVectors: [[[Float]]] = Array(repeating: [], count: effectiveShards)
        var shardIDs: [[String]] = Array(repeating: [], count: effectiveShards)
        for i in 0..<vectors.count {
            let shard = kmResult.assignments[i]
            shardVectors[shard].append(vectors[i])
            shardIDs[shard].append(ids[i])
        }

        var builtShards: [ANNSIndex] = []
        for shardIndex in 0..<effectiveShards {
            guard !shardVectors[shardIndex].isEmpty else { continue }
            let shard = ANNSIndex(configuration: configuration)
            try await shard.build(
                vectors: shardVectors[shardIndex],
                ids: shardIDs[shardIndex]
            )
            builtShards.append(shard)
        }

        self.centroids = kmResult.centroids
        self.shards = builtShards
        self.shardIDMappings = shardIDs.filter { !$0.isEmpty }
        self.isBuilt = true
    }

    public func search(
        query: [Float],
        k: Int,
        filter: SearchFilter? = nil,
        metric: Metric? = nil
    ) async throws -> [SearchResult] {
        guard isBuilt, !shards.isEmpty else { throw ANNSError.indexEmpty }
        guard k > 0 else { return [] }

        let searchMetric = metric ?? configuration.metric

        let centroidDistances = centroids.enumerated().map { (index, centroid) in
            (index, SIMDDistance.distance(query, centroid, metric: searchMetric))
        }.sorted { $0.1 < $1.1 }

        let probeCount = min(nprobe, shards.count)
        let probeIndices = centroidDistances.prefix(probeCount).map { $0.0 }

        var allResults: [SearchResult] = []
        for shardIndex in probeIndices {
            guard shardIndex < shards.count else { continue }
            let shardResults = try await shards[shardIndex].search(
                query: query, k: k, filter: filter, metric: metric
            )
            allResults.append(contentsOf: shardResults)
        }

        allResults.sort { $0.score < $1.score }
        return Array(allResults.prefix(k))
    }

    public var count: Int {
        get async {
            var total = 0
            for shard in shards { total += await shard.count }
            return total
        }
    }
}
```

### Tests

```swift
// Tests/MetalANNSTests/ShardedIndexTests.swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Sharded Index Tests")
struct ShardedIndexTests {

    @Test("Sharded search achieves reasonable recall")
    func shardedSearchRecall() async throws {
        // 1. Generate 500 random vectors (dim=32)
        // 2. Build ShardedIndex with numShards=8, nprobe=3
        // 3. Use first 20 vectors as queries
        // 4. For each query, check if the query vector's ID appears in top-10 results
        // 5. Expect recall >= 0.70
    }

    @Test("Sharded index distributes vectors across shards")
    func shardedDistribution() async throws {
        // 1. Generate 200 random vectors (dim=16)
        // 2. Build ShardedIndex with numShards=4
        // 3. Verify total count across all shards equals 200
        // 4. Verify each shard has at least 1 vector (with well-separated data)
    }
}
```

### Commit
```bash
git add Sources/MetalANNSCore/KMeans.swift Sources/MetalANNS/ShardedIndex.swift Tests/MetalANNSTests/ShardedIndexTests.swift
git commit -m "feat: add sharded index with IVF-style partitioning"
```

---

## Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Filtered search returns 0 results | Over-fetch multiplier too low | Increase from `k * 4` to `k * 8` or use `min(vectors.count, ...)` |
| `MetadataStore.matches` always returns false | Column name mismatch between set and filter | Ensure exact string match for column names |
| `PersistedMetadata` fails to decode after adding `metadataStore` | Missing backward compat | Make `metadataStore` optional in `PersistedMetadata` with `?` |
| Compound filter `.or` never matches | Using `allSatisfy` instead of `contains` | `.or` should use `filters.contains { matches(id:filter:) }` |
| `rangeSearch` returns empty for known vectors | maxDistance threshold too tight for the metric | Cosine distance is [0,2], L2 is [0,inf). Choose appropriate thresholds |
| Runtime metric changes recall dramatically | Expected — graph was built for different metric | Document: "recall may degrade when override metric differs from build metric" |
| Disk-backed reads garbage data | Wrong dataOffset calculation | Verify offset matches header size + section offsets for the file version |
| Disk-backed v3 format fails | Page padding not accounted for | Use same alignedOffset logic as MmapIndexLoader |
| Disk-backed EXC_BAD_ACCESS | MmapRegion deallocated before use | Ensure `mmapLifetime` ref kept alive in ANNSIndex |
| KMeans returns empty cluster | Random init picked duplicate centroid | K-means++ avoids this; empty cluster keeps previous centroid |
| Sharded recall very low | nprobe too small | Increase nprobe (e.g., nprobe=numShards/2 for high recall) |
| `DiskBackedVectorBuffer.buffer` used in GPU search | Placeholder buffer passed to Metal | GPU search path should check and error; disk-backed is CPU-only search |
| MmapTests failures | **KNOWN BASELINE ISSUE** from Phase 10 | Do NOT fix in this phase. Document and skip |

---

## Verification Checklist

After completing all five tasks:

```bash
# 1. All tests pass (except known MmapTests baseline failure)
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|PASS|FAIL|error:)'

# 2. Expected: 89+ tests (78 existing + 3 filtered + 2 range + 2 runtime + 2 disk-backed + 2 sharded), 0 new failures

# 3. Verify filtered search
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/FilteredSearchTests 2>&1 | grep -E '(PASS|FAIL)'

# 4. Verify range search
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/RangeSearchTests 2>&1 | grep -E '(PASS|FAIL)'

# 5. Verify runtime metric
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/RuntimeMetricTests 2>&1 | grep -E '(PASS|FAIL)'

# 6. Verify disk-backed
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/DiskBackedTests 2>&1 | grep -E '(PASS|FAIL)'

# 7. Verify sharded
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/ShardedIndexTests 2>&1 | grep -E '(PASS|FAIL)'

# 8. Verify no regressions (skip MmapTests — known baseline issue)
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/ANNSIndexTests 2>&1 | grep -E '(PASS|FAIL)'
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/FullGPUSearchTests 2>&1 | grep -E '(PASS|FAIL)'
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/PersistenceTests 2>&1 | grep -E '(PASS|FAIL)'
```

---

## Summary

| Task | New Files | Modified Files | New Tests | Commit |
|------|-----------|----------------|-----------|--------|
| 31: Filtered Search | `MetadataStore.swift`, `SearchFilter.swift`, `FilteredSearchTests.swift` | `ANNSIndex.swift` | 3 | `feat: add filtered search with metadata predicates` |
| 32: Range Search | `RangeSearchTests.swift` | `ANNSIndex.swift` | 2 | `feat: add range search with distance threshold` |
| 33: Runtime Metric | `RuntimeMetricTests.swift` | `ANNSIndex.swift` | 2 | `feat: support runtime metric selection at query time` |
| 34: Disk-Backed | `DiskBackedVectorBuffer.swift` | `ANNSIndex.swift` | 2 | `feat: add disk-backed index for memory-constrained devices` |
| 35: Sharded Index | `KMeans.swift`, `ShardedIndex.swift`, `ShardedIndexTests.swift` | — | 2 | `feat: add sharded index with IVF-style partitioning` |

**Expected end state**: 38 commits, 89+ tests, all passing (except known MmapTests baseline issue).

---

## Phase 12 Marks the End of the v2 Implementation Plan

After Phase 12, MetalANNS has:
- GPU-accelerated NN-Descent construction + beam search (Phases 1-8)
- Float16 support with 2x memory reduction (Phase 9)
- Batch insert, compaction, memory-mapped I/O (Phase 10)
- Filtered search, range search, runtime metric selection (Phase 11 APIs, built here)
- Disk-backed index for iOS, sharded indices for million-scale (Phase 12 features)

**Total across all v2 phases**: ~38 commits, ~89 tests, production-grade vector search engine.
