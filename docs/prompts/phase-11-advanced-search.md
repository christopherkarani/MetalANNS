# Phase 11: Advanced Search — Execution Prompt

> **Context**: You are implementing Phase 11 of MetalANNS, a GPU-native ANNS library for Apple Silicon.
> Phase 10 (Scalability Primitives) should be complete before starting this phase.
> This phase adds three search capabilities: filtered search with metadata predicates, range search, and runtime metric selection.

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

## Current Search Architecture

### ANNSIndex.search() (Sources/MetalANNS/ANNSIndex.swift)

```swift
public func search(query: [Float], k: Int) async throws -> [SearchResult] {
    guard isBuilt, let vectors, let graph else { throw ANNSError.indexEmpty }
    guard query.count == vectors.dim else { throw ANNSError.dimensionMismatch(...) }
    guard k > 0 else { return [] }

    let deletedCount = softDeletion.deletedCount
    let effectiveK = min(vectors.count, k + deletedCount)
    let effectiveEf = max(configuration.efSearch, effectiveK)

    let rawResults: [SearchResult]
    if let context {
        rawResults = try await FullGPUSearch.search(
            context: context, query: query, vectors: vectors, graph: graph,
            entryPoint: Int(entryPoint), k: max(1, effectiveK),
            ef: max(1, effectiveEf), metric: configuration.metric
        )
    } else {
        rawResults = try await BeamSearchCPU.search(
            query: query, vectors: extractVectors(from: vectors),
            graph: extractGraph(from: graph), entryPoint: Int(entryPoint),
            k: max(1, effectiveK), ef: max(1, effectiveEf), metric: configuration.metric
        )
    }

    let filtered = softDeletion.filterResults(rawResults)
    let mapped = filtered.compactMap { result -> SearchResult? in
        guard let externalID = idMap.externalID(for: result.internalID) else { return nil }
        return SearchResult(id: externalID, score: result.score, internalID: result.internalID)
    }
    return Array(mapped.prefix(k))
}
```

### FullGPUSearch (Sources/MetalANNSCore/FullGPUSearch.swift)

Single-threadgroup beam search kernel. Parameters passed via `setBytes`:
- `buffer(0)`: vectors (Float32 or Float16 buffer)
- `buffer(1)`: adjacency graph
- `buffer(2)`: query vector
- `buffer(3)`: output distances
- `buffer(4)`: output IDs
- `buffer(5-11)`: nodeCount, degree, dim, k, ef, entryPoint, metricType

The metric is already passed as `metricType: UInt32` (0=cosine, 1=l2, 2=innerProduct). **No shader changes needed for runtime metric selection** — just pass a different metric value at dispatch time.

### BeamSearchCPU (Sources/MetalANNSCore/BeamSearchCPU.swift)

CPU beam search. Already accepts `metric: Metric` parameter. Uses `SIMDDistance.distance()` for distance computation.

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
public enum Metric: String, Sendable, Codable {
    case cosine, l2, innerProduct
}
```

### SoftDeletion (Sources/MetalANNSCore/SoftDeletion.swift)

```swift
public struct SoftDeletion: Sendable, Codable {
    private var deletedIDs: Set<UInt32> = []
    public mutating func markDeleted(_ internalID: UInt32)
    public func isDeleted(_ internalID: UInt32) -> Bool
    public var deletedCount: Int
    public func filterResults(_ results: [SearchResult]) -> [SearchResult]
    // Phase 10 added: public var allDeletedIDs: Set<UInt32>
}
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
- `Sources/MetalANNSCore/IndexSerializer.swift` — persist metadata in save/load

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

    /// Remap internal IDs during compaction. Returns new MetadataStore with remapped keys.
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

    /// Remove metadata for a specific internal ID.
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

**Note**: `SearchFilter` must be `Sendable`. Since all cases use `String`, `Float`, `Set<String>`, and recursive `[SearchFilter]` — all of which are `Sendable` — this works automatically. Do NOT make it `Codable` unless needed (recursive enums with associated values require custom Codable conformance).

### Filtered Search Strategy: Post-Filter with Over-Fetch

The search flow changes to:

1. Estimate over-fetch factor: `overFetchMultiplier = max(2, totalCount / max(1, totalCount - deletedCount - estimatedFilteredOut))`
2. For simplicity, use a fixed over-fetch: `effectiveK = min(vectors.count, k * 4 + deletedCount)` when a filter is provided
3. Run the standard beam search with the inflated `effectiveK`
4. Filter results through `softDeletion.filterResults()` (as before)
5. Filter results through `metadataStore.matches(id:filter:)` (new)
6. Map to external IDs, take top-k

### ANNSIndex Changes

Add to ANNSIndex:

```swift
// New private property
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

**PersistedMetadata** already exists in ANNSIndex:
```swift
private struct PersistedMetadata: Codable, Sendable {
    let configuration: IndexConfiguration
    let softDeletion: SoftDeletion
}
```

Add `metadataStore` to it:
```swift
private struct PersistedMetadata: Codable, Sendable {
    let configuration: IndexConfiguration
    let softDeletion: SoftDeletion
    let metadataStore: MetadataStore?  // Optional for backward compat
}
```

In `save()`: include `metadataStore` in the JSON.
In `load()`: read `metadataStore` from JSON (default to empty if nil for backward compat).

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
        // 6. Expect recall >= 0.50 (relaxed because over-fetch may miss some)
    }

    @Test("Compound filters work correctly")
    func compoundFilterWorks() async throws {
        // 1. Build index with 100 vectors (dim=16)
        // 2. Set metadata: category (string) and score (float)
        //    - "v0"..."v24": category="A", score=1.0
        //    - "v25"..."v49": category="A", score=5.0
        //    - "v50"..."v74": category="B", score=1.0
        //    - "v75"..."v99": category="B", score=5.0
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

### Algorithm

Range search is a modified beam search that:
1. Runs the standard beam search with a large `ef` (to explore broadly)
2. Filters results to only those with `score <= maxDistance`
3. Returns up to `limit` results (default 1000), sorted by distance

The key difference from k-NN search: we don't know how many results we'll get. We over-fetch using a large ef and filter.

### ANNSIndex.rangeSearch API

```swift
// Add to ANNSIndex.swift
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
    guard maxDistance > 0 else {
        return []
    }
    guard limit > 0 else {
        return []
    }

    // Use a large ef to explore broadly
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

    // Apply metadata filter if present
    if let filter {
        filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
    }

    // Apply distance threshold
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
        // 5. Verify count > 0 (should find at least some)
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
- `Sources/MetalANNS/ANNSIndex.swift` — add optional `metric` parameter to `search()` and `rangeSearch()`

### No Shader Changes Needed

The GPU beam search kernel already accepts `metricType` as a runtime parameter (buffer index 11). The Swift code in `FullGPUSearch.search()` already converts `Metric` to `UInt32` and passes it via `setBytes`. Similarly, `BeamSearchCPU.search()` already accepts `metric: Metric`.

The only change is in `ANNSIndex` — the public API needs an optional metric override.

### ANNSIndex Changes

Modify the `search` signature (already modified in Task 31 to add `filter`):

```swift
public func search(
    query: [Float],
    k: Int,
    filter: SearchFilter? = nil,
    metric: Metric? = nil        // NEW: optional runtime metric override
) async throws -> [SearchResult] {
    // ... existing guards ...

    let searchMetric = metric ?? configuration.metric  // Use override or default

    // Pass searchMetric instead of configuration.metric to the search dispatch:
    if let context {
        rawResults = try await FullGPUSearch.search(
            ..., metric: searchMetric
        )
    } else {
        rawResults = try await BeamSearchCPU.search(
            ..., metric: searchMetric
        )
    }
    // ... rest unchanged ...
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
    // ... use searchMetric in search dispatch ...
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
    // ... calls self.search(query:k:filter:metric:) internally ...
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

## Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Filtered search returns 0 results | Over-fetch multiplier too low | Increase from `k * 4` to `k * 8` or use `min(vectors.count, ...)` |
| `MetadataStore.matches` always returns false | Column name mismatch between set and filter | Ensure exact string match for column names |
| `SearchFilter` causes infinite recursion | Circular .and/.or nesting | Not possible with enum (no reference cycles), but test compound depth |
| `rangeSearch` returns empty for known vectors | maxDistance threshold too tight for the metric | Cosine distance is [0,2], L2 is [0,inf). Choose appropriate thresholds. |
| Runtime metric changes recall dramatically | Expected — graph was built for different metric | Document in API: "recall may degrade when override metric differs from build metric" |
| `PersistedMetadata` fails to decode after adding `metadataStore` | Missing backward compat | Make `metadataStore` optional in `PersistedMetadata` with `?` |
| Compound filter `.or` never matches | Using `allSatisfy` instead of `contains` | `.or` should use `filters.contains { matches(id:filter:) }` |

---

## Verification Checklist

After completing all three tasks:

```bash
# 1. All tests pass
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|PASS|FAIL|error:)'

# 2. Expected: 78+ tests (71 existing + 7 new), 0 failures

# 3. Verify filtered search
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/FilteredSearchTests 2>&1 | grep -E '(PASS|FAIL)'

# 4. Verify range search
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/RangeSearchTests 2>&1 | grep -E '(PASS|FAIL)'

# 5. Verify runtime metric
xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/RuntimeMetricTests 2>&1 | grep -E '(PASS|FAIL)'

# 6. Verify no regressions
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

**Expected end state**: 38 commits, 78+ tests, all passing.
