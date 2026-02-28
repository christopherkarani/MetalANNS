# Phase 22: In-Graph Filtering

### Mission

Replace the k×4 over-fetch + post-filter pattern in `ANNSIndex.search()` with a
predicate closure that gates result-heap insertion inside `BeamSearchCPU`. Filtered
searches become O(ef × filter_eval_cost) instead of O(k×4 × filter_eval_cost). This
is a pure internal optimization — the public API signature is unchanged.

---

### Verified Codebase Facts

Read each file before touching it. These facts were verified against the current source:

| Fact | Source |
|------|--------|
| `ANNSIndex.search()` inflates k×4 when filter present | `ANNSIndex.swift:469-473` |
| Post-filter applied after search via `softDeletion.filterResults()` + `filter { metadataStore.matches }` | `ANNSIndex.swift:529-531` |
| `rangeSearch()` does NOT inflate k — no k×4 needed there | `ANNSIndex.swift:564-566` |
| `BeamSearchCPU.search()` adds to both `candidates` and `results` at line 76-81 | `BeamSearchCPU.swift:75-81` |
| `HNSWSearchCPU.search()` calls `BeamSearchCPU.search()` for layer-0 | `HNSWSearchCPU.swift:44-57` |
| `QuantizedHNSWSearchCPU.search()` calls `BeamSearchCPU.search()` for layer-0 | `QuantizedHNSWSearchCPU.swift:44-58` |
| `MetadataStore.matches(id:filter:)` is a sync, pure function | `MetadataStore.swift:34` |
| `SoftDeletion.isDeleted(_:)` is a sync, pure function | `SoftDeletion.swift:16` |
| `batchSearch()` delegates to `search()` — gets predicate for free | `ANNSIndex.swift:636+` |
| Existing `FilteredSearchTests.swift` has 3 passing tests, thresholds are `recall >= 0.50` | `FilteredSearchTests.swift:62` |

---

### TDD Implementation Order

Work strictly test-first. Do not write implementation code before the test that drives it exists and fails.

**Round 1** — add predicate to `BeamSearchCPU` (unit tests)
Write tests that verify predicate gates results but not graph traversal. See Step 3.
Compile-fail → implement → pass.

**Round 2** — integration through `ANNSIndex`
Add recall + correctness tests. Then wire predicate into `ANNSIndex`.

Run after every step:
```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E "PASS|FAIL|error:"
```

---

### Step 1: Add Predicate to `BeamSearchCPU`

**File**: `Sources/MetalANNSCore/BeamSearchCPU.swift`

Add one optional parameter with a default of `nil` — backward-compatible, all existing call sites compile unchanged.

**New signature:**
```swift
public static func search(
    query: [Float],
    vectors: [[Float]],
    graph: [[(UInt32, Float)]],
    entryPoint: Int,
    k: Int,
    ef: Int,
    metric: Metric,
    predicate: ((UInt32) -> Bool)? = nil     // NEW — nil = no filtering
) async throws -> [SearchResult]
```

**How to apply the predicate — critical rule:**

Gate **result-heap insertion only**. Continue adding to `candidates` (the traversal
frontier) regardless of predicate, so the graph can still be navigated through
non-matching nodes as stepping stones.

Change this block (current lines 75-81):
```swift
if results.count < efLimit || candidateDistance < results[results.count - 1].distance {
    let candidate = Candidate(nodeID: neighborID, distance: candidateDistance)
    insertSorted(candidate, into: &candidates)
    insertSorted(candidate, into: &results)
    if results.count > efLimit {
        results.removeLast()
    }
}
```

To:
```swift
if results.count < efLimit || candidateDistance < results[results.count - 1].distance {
    let candidate = Candidate(nodeID: neighborID, distance: candidateDistance)
    insertSorted(candidate, into: &candidates)        // always — needed for traversal
    if predicate == nil || predicate!(neighborID) {   // gate result inclusion only
        insertSorted(candidate, into: &results)
        if results.count > efLimit {
            results.removeLast()
        }
    }
}
```

The termination condition (`results.count >= efLimit, current.distance > worst`) is
unchanged. With a selective predicate, fewer results accumulate, so the loop runs
longer — this is correct and expected behavior for filtered search.

---

### Step 2: Forward Predicate Through the Search Chain

**File**: `Sources/MetalANNSCore/HNSWSearchCPU.swift`

Add `predicate: ((UInt32) -> Bool)? = nil` to `search()` and forward it to the
`BeamSearchCPU.search()` call. Default nil keeps all existing call sites compiling.

```swift
public static func search(
    query: [Float],
    vectors: [[Float]],
    hnsw: HNSWLayers,
    baseGraph: [[(UInt32, Float)]],
    k: Int,
    ef: Int,
    metric: Metric,
    predicate: ((UInt32) -> Bool)? = nil    // NEW
) async throws(ANNSError) -> [SearchResult]
```

Forward to `BeamSearchCPU.search(... predicate: predicate)`.

The greedy descent (layers maxLayer → 1) is NOT predicate-gated — we always want
the best entry point regardless of whether the node passes the filter.

---

**File**: `Sources/MetalANNSCore/QuantizedHNSWSearchCPU.swift`

Same change — add `predicate` parameter to `search()`, forward to `BeamSearchCPU.search()`.
The `greedySearchLayer()` function is NOT changed — no predicate gating during descent.

---

### Step 3: Write `InGraphFilteringTests.swift`

**File**: `Tests/MetalANNSTests/InGraphFilteringTests.swift`

```swift
import Testing
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("In-Graph Filtering Tests")
struct InGraphFilteringTests {

    // TEST 1: predicateGatesResultsNotTraversal
    // Build a small graph (20 nodes, dim=8). Mark even-numbered nodes as "allowed".
    // Call BeamSearchCPU.search() with predicate that only allows even IDs.
    // Assert: all returned SearchResult.internalID values are even.
    // Assert: result count == min(k, count of even nodes).
    @Test func predicateGatesResultsNotTraversal() async throws { ... }

    // TEST 2: nilPredicateIsBackwardCompatible
    // Call BeamSearchCPU.search() without predicate (nil default).
    // Assert: results match the same call with predicate: nil explicitly.
    // Both must return identical SearchResult arrays.
    @Test func nilPredicateIsBackwardCompatible() async throws { ... }

    // TEST 3: filteredSearchCorrectness
    // Build ANNSIndex with 200 vectors (dim=32, L2, CPU-only).
    // Tag vectors 0-99 as category="A", 100-199 as category="B".
    // Search top-10 with filter: .equals(column: "category", value: "A").
    // Assert: all returned IDs are in 0-99.
    // Assert: result count == 10.
    @Test func filteredSearchCorrectness() async throws { ... }

    // TEST 4: filteredSearchRecall
    // Build ANNSIndex with 500 vectors (dim=32, L2, CPU-only).
    // Tag every other vector with category "even"/"odd".
    // For 20 queries from "even" vectors, compute recall@10 vs brute-force filtered ground truth.
    // Ground truth: sort all "even" vectors by L2 distance to query, take top 10.
    // #expect(recall >= 0.70)
    @Test func filteredSearchRecall() async throws { ... }

    // TEST 5: softDeletedExcludedFromResults
    // Build ANNSIndex with 100 vectors. Delete IDs "v0", "v1", "v2" via index.delete().
    // Run 10 searches. Assert: none of "v0", "v1", "v2" appear in any result.
    @Test func softDeletedExcludedFromResults() async throws { ... }

    // TEST 6: noKTimesFourInflation
    // Build ANNSIndex with 200 vectors, tag 100 as category "A".
    // This test instruments that effectiveK used during search is NOT > k * 2.
    // Proxy check: search with k=5, filter="A". If results.count == 5 and all are category A,
    // the predicate approach is working (not over-fetching and post-filtering).
    // Assert: results.count == 5, all match filter.
    @Test func filteredSearchReturnsPreciselyK() async throws { ... }
}
```

**Private helpers in the test file:**
```swift
private func makeVectors(count: Int, dim: Int) -> [[Float]] {
    // deterministic gaussian-ish via sin/cos seeding
}
private func bruteForceFilteredTopK(
    query: [Float], vectors: [[Float]], ids: [String],
    predicate: (String) -> Bool, k: Int
) -> Set<String> {
    // sort by L2, filter, take k
}
private func recall(results: [SearchResult], groundTruth: Set<String>) -> Double {
    Double(results.filter { groundTruth.contains($0.id) }.count) / Double(groundTruth.count)
}
```

---

### Step 4: Wire Predicate into `ANNSIndex.search()`

**File**: `Sources/MetalANNS/ANNSIndex.swift`

**4a. Remove k×4, build unified predicate**

Replace the current `effectiveK` block (lines 466-473):
```swift
// BEFORE:
let hasFilter = filter != nil
let deletedCount = softDeletion.deletedCount
let effectiveK: Int
if hasFilter {
    effectiveK = min(vectors.count, k * 4 + deletedCount)
} else {
    effectiveK = min(vectors.count, k + deletedCount)
}
```

With:
```swift
// AFTER:
let deletedCount = softDeletion.deletedCount
let effectiveK = min(vectors.count, k + max(deletedCount, 10))

// Build predicate — nil only when nothing to filter (fast path: no overhead)
let needsPredicate = filter != nil || deletedCount > 0
let searchPredicate: ((UInt32) -> Bool)? = needsPredicate
    ? { [softDeletion, metadataStore, filter] id in
          !softDeletion.isDeleted(id) &&
          (filter == nil || metadataStore.matches(id: id, filter: filter!))
      }
    : nil
```

**4b. Pass predicate to all three CPU search branches**

`BeamSearchCPU.search()` call — add `predicate: searchPredicate`
`HNSWSearchCPU.search()` call — add `predicate: searchPredicate`
`QuantizedHNSWSearchCPU.search()` call — add `predicate: searchPredicate`

The GPU path (`FullGPUSearch.search()`) does not change — GPU filtering is out of scope.

**4c. Remove the post-filter step**

The predicate already excludes deleted and non-matching nodes from results.
Replace lines 529-531:
```swift
// BEFORE:
var filtered = softDeletion.filterResults(rawResults)
if let filter {
    filtered = filtered.filter { metadataStore.matches(id: $0.internalID, filter: filter) }
}
```

With a safety net only (should never trigger, but guards against GPU path returning deleted nodes):
```swift
// AFTER:
let filtered = context != nil && supportsGPUSearch(for: vectors)
    ? softDeletion.filterResults(rawResults).filter { filter == nil || metadataStore.matches(id: $0.internalID, filter: filter!) }
    : rawResults   // predicate already handled it
```

---

### Step 5: Wire Predicate into `ANNSIndex.rangeSearch()`

`rangeSearch()` has no k×4 inflation. The change is simpler:

Build the same predicate closure (same code as 4a, minus the `effectiveK` change).
Pass it to all three CPU search branches.
Replace the post-filter (lines 621-624) with the same GPU-only safety net from 4c.
Keep the distance filter (`filter { $0.score <= maxDistance }`) — that is not predicate-based.

---

### Step 6: Verify No Regressions

Run the full test suite:
```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30
```

Pay particular attention to:
- `FilteredSearchTests.swift` — all 3 existing tests must still pass (thresholds are unchanged)
- `HNSWTests.swift`, `QuantizedHNSW*Tests.swift` — unfiltered paths must be unaffected
- Any `batchSearch` tests — delegates to `search()`, gets predicate for free

---

### Definition of Done

- [ ] `BeamSearchCPU.search()` accepts `predicate: ((UInt32) -> Bool)? = nil`; predicate gates `results` only, not `candidates`
- [ ] `HNSWSearchCPU.search()` and `QuantizedHNSWSearchCPU.search()` forward predicate to `BeamSearchCPU`
- [ ] `ANNSIndex.search()` builds predicate closure; removes k×4; removes post-filter (CPU path)
- [ ] `ANNSIndex.rangeSearch()` builds predicate closure; removes post-filter (CPU path)
- [ ] All 6 new tests in `InGraphFilteringTests` pass including `recall >= 0.70`
- [ ] All pre-existing tests pass — zero regressions
- [ ] No `// TODO`, no dead code, no commented-out blocks in committed files

---

### What Not To Do

- Do not apply the predicate to `candidates` — nodes must still be traversable even if they don't match the filter
- Do not change the GPU search path (`FullGPUSearch`) — out of scope for this phase
- Do not change `greedySearchLayer()` in either HNSW search type — greedy descent is unfiltered by design
- Do not raise `efSearch` or add extra multipliers to compensate — the predicate approach naturally explores more of the graph when filtering is selective
- Do not capture `self` in the predicate closure — capture `softDeletion`, `metadataStore`, and `filter` by value to avoid actor isolation issues inside an `async` context
