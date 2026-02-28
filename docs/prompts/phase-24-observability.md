# Phase 24: Index Observability

### Mission

Add opt-in structured metrics to `ANNSIndex` and `StreamingIndex`. Zero overhead when
disabled (a single `if let` nil check). When enabled, records operation counts and
nanosecond-resolution latency histograms, queryable via a `MetricsSnapshot`.

This is purely additive — no existing behaviour changes.

---

### Verified Codebase Facts

Read each file before touching it.

| Fact | Source |
|------|--------|
| `ANNSIndex` is a Swift actor; `search()`, `insert()`, `batchInsert()` are all `async throws` | `ANNSIndex.swift:5` |
| `StreamingIndex` is a Swift actor; `triggerMerge()` is `private func ... async throws` | `StreamingIndex.swift:10,548` |
| A real merge completes at `StreamingIndex.swift:578-591` — `base = newBase` then clears delta | `StreamingIndex.swift:578-591` |
| Early-return paths in `triggerMerge()` (empty/single-item) must NOT count as a merge | `StreamingIndex.swift:558-575` |
| `batchSearch()` in `ANNSIndex` uses a `withThrowingTaskGroup` calling `search()` per query | `ANNSIndex.swift:636+` |
| `batchSearch()` in `StreamingIndex` also calls `search()` per query | `StreamingIndex.swift:134-158` |
| Project targets iOS 17+ / macOS 14+ — `ContinuousClock` is fully available | `Package.swift` |
| `Sources/MetalANNS/` is the public-API target — `IndexMetrics` lives here | directory structure |

---

### TDD Implementation Order

Work strictly test-first.

**Round 1** — `IndexMetrics` in isolation
Write `IndexMetricsTests.swift`. Tests fail to compile. Implement `IndexMetrics` + `MetricsSnapshot`. Tests pass.

**Round 2** — integration into `ANNSIndex`
Tests that check counters via the public API. Wire into `ANNSIndex`. Tests pass.

**Round 3** — integration into `StreamingIndex`
`streamingMergeRecorded` test. Wire into `StreamingIndex`. Test passes.

Run after every step:
```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E "PASS|FAIL|error:"
```

---

### Step 1: Create `IndexMetrics.swift`

**File**: `Sources/MetalANNS/IndexMetrics.swift`

```swift
import Foundation

public actor IndexMetrics {

    // ---- Counters ----
    public private(set) var insertCount: Int = 0
    public private(set) var searchCount: Int = 0
    public private(set) var batchSearchCount: Int = 0
    public private(set) var mergeCount: Int = 0

    // ---- Latency histograms (10 buckets, indexed 0-9) ----
    // Bucket upper bounds in nanoseconds.
    // Bucket i holds samples where bucketBounds[i-1] <= ns < bucketBounds[i].
    // Bucket 9 is the overflow bucket (>= 500ms).
    private static let bucketBoundsNs: [UInt64] = [
        1_000_000,      // bucket 0: < 1ms
        2_000_000,      // bucket 1: < 2ms
        5_000_000,      // bucket 2: < 5ms
        10_000_000,     // bucket 3: < 10ms
        20_000_000,     // bucket 4: < 20ms
        50_000_000,     // bucket 5: < 50ms
        100_000_000,    // bucket 6: < 100ms
        200_000_000,    // bucket 7: < 200ms
        500_000_000,    // bucket 8: < 500ms
        1_000_000_000   // bucket 9: >= 500ms (capped at 1s for display)
    ]

    public private(set) var searchLatencyHistogram: [Int] = Array(repeating: 0, count: 10)
    public private(set) var insertLatencyHistogram: [Int] = Array(repeating: 0, count: 10)

    public init() {}

    // ---- Internal recording (called from ANNSIndex / StreamingIndex) ----

    func recordInsert(durationNs: UInt64) {
        insertCount += 1
        insertLatencyHistogram[bucketIndex(for: durationNs)] += 1
    }

    /// Increments insertCount by `count` with a single shared timing (for batch operations).
    func recordBatchInsert(count: Int, durationNs: UInt64) {
        insertCount += count
        insertLatencyHistogram[bucketIndex(for: durationNs)] += 1
    }

    func recordSearch(durationNs: UInt64) {
        searchCount += 1
        searchLatencyHistogram[bucketIndex(for: durationNs)] += 1
    }

    func recordBatchSearch() {
        batchSearchCount += 1
    }

    func recordMerge() {
        mergeCount += 1
    }

    // ---- Public snapshot ----

    public func snapshot() -> MetricsSnapshot {
        MetricsSnapshot(
            insertCount: insertCount,
            searchCount: searchCount,
            batchSearchCount: batchSearchCount,
            mergeCount: mergeCount,
            searchP50LatencyMs: percentile(0.50, from: searchLatencyHistogram),
            searchP99LatencyMs: percentile(0.99, from: searchLatencyHistogram),
            timestamp: Date()
        )
    }

    // ---- Helpers ----

    private func bucketIndex(for ns: UInt64) -> Int {
        for (i, bound) in Self.bucketBoundsNs.enumerated() {
            if ns < bound { return i }
        }
        return Self.bucketBoundsNs.count - 1
    }

    /// Computes an approximate percentile (linear interpolation within bucket).
    private func percentile(_ p: Double, from histogram: [Int]) -> Double {
        let total = histogram.reduce(0, +)
        guard total > 0 else { return 0.0 }
        let target = max(1, Int(ceil(Double(total) * p)))
        var cumulative = 0
        for (i, count) in histogram.enumerated() {
            cumulative += count
            if cumulative >= target {
                let lowerNs = i == 0 ? 0.0 : Double(Self.bucketBoundsNs[i - 1])
                let upperNs = Double(Self.bucketBoundsNs[i])
                return (lowerNs + upperNs) / 2.0 / 1_000_000.0   // convert to ms
            }
        }
        // Fell through (rounding): return last bucket midpoint
        let last = Self.bucketBoundsNs.count - 1
        return Double(Self.bucketBoundsNs[last]) / 1_000_000.0
    }
}
```

---

### Step 2: Create `MetricsSnapshot.swift`

**File**: `Sources/MetalANNS/MetricsSnapshot.swift`

```swift
import Foundation

public struct MetricsSnapshot: Sendable, Codable {
    public let insertCount: Int
    public let searchCount: Int
    public let batchSearchCount: Int
    public let mergeCount: Int
    public let searchP50LatencyMs: Double
    public let searchP99LatencyMs: Double
    public let timestamp: Date
}
```

---

### Step 3: Write `IndexMetricsTests.swift`

**File**: `Tests/MetalANNSTests/IndexMetricsTests.swift`

```swift
import Testing
import Foundation
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Index Metrics Tests")
struct IndexMetricsTests {

    // TEST 1: metricsDisabledByDefault
    // ANNSIndex() and StreamingIndex() both have metrics == nil.
    // Build a small index, run 5 searches and 1 insert — no crash, no side effects.
    @Test func metricsDisabledByDefault() async throws {
        let index = ANNSIndex(configuration: .default)
        #expect(await index.metrics == nil)

        let streaming = StreamingIndex()
        #expect(await streaming.metrics == nil)
    }

    // TEST 2: searchCountIncrements
    // Set metrics on ANNSIndex. Build index with 50 vectors.
    // Run 10 searches. Assert searchCount == 10.
    @Test func searchCountIncrements() async throws { ... }

    // TEST 3: insertCountIncrements
    // Set metrics. Build index with 50 vectors, insert 5 more one-by-one.
    // Assert insertCount == 5.
    @Test func insertCountIncrements() async throws { ... }

    // TEST 4: batchInsertCountsVectors
    // Set metrics. Build index with 50 vectors, batchInsert 20 more.
    // Assert insertCount == 20 (batch records count of vectors, not 1 call).
    @Test func batchInsertCountsVectors() async throws { ... }

    // TEST 5: batchSearchCountIncrements
    // Set metrics. Build index, run batchSearch 3 times (5 queries each).
    // Assert batchSearchCount == 3. Assert searchCount == 15 (individual queries).
    @Test func batchSearchCountIncrements() async throws { ... }

    // TEST 6: latencyHistogramPopulated
    // Set metrics. Run 50 searches.
    // Assert searchLatencyHistogram.reduce(0, +) == 50.
    // Assert at least one bucket has count > 0.
    @Test func latencyHistogramPopulated() async throws { ... }

    // TEST 7: snapshotSerializesToJSON
    // Set metrics. Run 5 searches and 2 inserts.
    // Call snapshot(). Encode to JSON via JSONEncoder. Decode via JSONDecoder.
    // Assert decoded.searchCount == 5, decoded.insertCount == 2.
    // Assert decoded.searchP50LatencyMs >= 0.
    @Test func snapshotSerializesToJSON() async throws { ... }

    // TEST 8: streamingMergeRecorded
    // Create StreamingIndex with small deltaCapacity (e.g. 5).
    // Set metrics. Insert enough vectors to trigger a merge. Call flush().
    // Assert mergeCount >= 1.
    @Test func streamingMergeRecorded() async throws { ... }

    // TEST 9: sharedMetricsAcrossIndexes
    // Create one IndexMetrics instance. Assign it to both an ANNSIndex and a StreamingIndex.
    // Run 3 searches on ANNSIndex, 2 searches on StreamingIndex.
    // Assert shared metrics.searchCount == 5.
    @Test func sharedMetricsAcrossIndexes() async throws { ... }
}
```

**Private helpers:**
```swift
private func makeIndex(count: Int = 50, dim: Int = 16) async throws -> ANNSIndex {
    let vectors = (0..<count).map { i in (0..<dim).map { _ in Float.random(in: -1...1) } }
    let ids = (0..<count).map { "v\($0)" }
    let index = ANNSIndex(configuration: IndexConfiguration(
        degree: 8, metric: .cosine, hnswConfiguration: .init(enabled: false)
    ))
    try await index.build(vectors: vectors, ids: ids)
    return index
}
```

---

### Step 4: Wire into `ANNSIndex`

**File**: `Sources/MetalANNS/ANNSIndex.swift`

**4a. Add the `metrics` property**

Inside the actor body (alongside other private stored properties):
```swift
public var metrics: IndexMetrics? = nil
```

**4b. Instrument `search()`**

Place the clock start after all `guard` checks (before the GPU/CPU dispatch). Place
the recording call immediately before `return`:

```swift
// After guards, before search dispatch:
let _searchStart = ContinuousClock.now

// ... existing search logic unchanged ...

// Immediately before `return Array(mapped.prefix(k))`:
if let m = metrics {
    let elapsed = ContinuousClock.now - _searchStart
    let ns = UInt64(elapsed.components.seconds) * 1_000_000_000
        + UInt64(elapsed.components.attoseconds) / 1_000_000_000
    await m.recordSearch(durationNs: ns)
}
return Array(mapped.prefix(k))
```

Apply the same pattern to `rangeSearch()` before its `return`.

**4c. Instrument `insert()`**

Start clock after all guards (just before `IncrementalBuilder.insert`):
```swift
let _insertStart = ContinuousClock.now

try IncrementalBuilder.insert(...)
// ... rest of insert ...

if let m = metrics {
    let elapsed = ContinuousClock.now - _insertStart
    let ns = UInt64(elapsed.components.seconds) * 1_000_000_000
        + UInt64(elapsed.components.attoseconds) / 1_000_000_000
    await m.recordInsert(durationNs: ns)
}
```

**4d. Instrument `batchInsert()`**

Start clock just before `BatchIncrementalBuilder.batchInsert(...)`. Record after:
```swift
let _batchStart = ContinuousClock.now
let insertedCount = vectors.count

try BatchIncrementalBuilder.batchInsert(...)

if let m = metrics {
    let elapsed = ContinuousClock.now - _batchStart
    let ns = UInt64(elapsed.components.seconds) * 1_000_000_000
        + UInt64(elapsed.components.attoseconds) / 1_000_000_000
    await m.recordBatchInsert(count: insertedCount, durationNs: ns)
}
```

**4e. Instrument `batchSearch()`**

Add one line at the entry of the function (after the `guard !queries.isEmpty` check):
```swift
if let m = metrics { await m.recordBatchSearch() }
```

---

### Step 5: Wire into `StreamingIndex`

**File**: `Sources/MetalANNS/StreamingIndex.swift`

**5a. Add the `metrics` property**

```swift
public var metrics: IndexMetrics? = nil
```

**5b. Instrument `triggerMerge()` — real-merge path only**

In `triggerMerge()`, the real merge completes at line ~578 where `base = newBase` is
assigned. Add recording ONLY after this assignment, not in the early-return paths:

```swift
base = newBase
idInBase = Set(merged.ids)
delta = nil
idInDelta.removeAll()
pendingVectors.removeAll(keepingCapacity: true)
pendingIDs.removeAll(keepingCapacity: true)

// Record merge only here — after the real merge completes
if let m = metrics { await m.recordMerge() }

// ... tail handling continues unchanged ...
```

**5c. Do not instrument search/insert on `StreamingIndex` directly**

`StreamingIndex.search()` delegates to `base.search()` and `delta.search()`, which
already increment their own `ANNSIndex.metrics`. To avoid double-counting, do not
add separate instrumentation to `StreamingIndex.search()`.

If a shared `IndexMetrics` instance is assigned to both the `StreamingIndex` and its
internal `base`/`delta` indexes, the user accepts that search counts will accumulate
from the sub-indexes. Document this in a comment on `StreamingIndex.metrics`.

---

### Step 6: Verify No Regressions

```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30
```

Critical checks:
- All `StreamingIndex*Tests.swift` must still pass — `triggerMerge()` call sites unchanged
- `FilteredSearchTests.swift` — no timing overhead with nil metrics
- Any persistence tests — `metrics` property is not persisted (in-memory only); assert it's nil after load

---

### Definition of Done

- [ ] `IndexMetrics` actor exists in `Sources/MetalANNS/` with all counters and histograms
- [ ] `MetricsSnapshot` is `Sendable + Codable`; `JSONEncoder().encode(snapshot())` succeeds
- [ ] `ANNSIndex.metrics: IndexMetrics? = nil` — nil by default
- [ ] `StreamingIndex.metrics: IndexMetrics? = nil` — nil by default
- [ ] `search()` and `rangeSearch()` in `ANNSIndex` record latency when metrics non-nil
- [ ] `insert()` records per-vector latency; `batchInsert()` records vector count + single timing
- [ ] `batchSearch()` increments `batchSearchCount` once per call
- [ ] `StreamingIndex.triggerMerge()` increments `mergeCount` only on real merges
- [ ] All 9 new tests pass
- [ ] All pre-existing tests pass — zero regressions
- [ ] `metrics` is NOT saved/loaded by `IndexSerializer` or `StreamingIndex.save/load`

---

### What Not To Do

- Do not make timing calls outside `async` functions — `ContinuousClock.now` is sync and safe everywhere, but `await m.record*(...)` requires the calling function to be `async`
- Do not add instrumentation to `rangeSearch()` in `StreamingIndex` — it delegates to sub-indexes
- Do not persist `metrics` to disk — it is an in-memory, session-scoped object
- Do not use `os_signpost` or `os.log` — the plan calls for structured in-process counters, not system tracing
- Do not add any mutex or lock — `IndexMetrics` is an actor; Swift handles the synchronization
- Do not start the clock before the `guard` checks — we time only successful operations
