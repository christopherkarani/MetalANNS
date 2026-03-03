# Phase 3: Fix StreamingIndex Unbounded Memory Growth

## Role

You are a senior Swift engineer working in MetalANNS — a GPU-native ANN search library for Apple Silicon. You have deep expertise in Swift actors, memory management, and graph-based index design.

## Context

**Project:** MetalANNS at `/Users/chriskarani/CodingProjects/MetalANNS`
**Branch:** `grdb4`
**Language:** Swift 6.0 with strict concurrency
**Build system:** SPM via xcodebuild
**Task tracker:** `tasks/wax-readiness-todo.md`
**Full plan:** `docs/plans/2026-02-28-metalanns-wax-readiness.md`

### Root-cause analysis — read this carefully before touching any code

**Files to read first:** `Sources/MetalANNS/StreamingIndex.swift` (1043 lines), `Sources/MetalANNS/ANNSIndex.swift`

`StreamingIndex` has two arrays that grow with every insert and are never trimmed:

```swift
// StreamingIndex.swift:88-89
private var allVectorData: [Float] = []   // flat vector payload
private var allIDsList: [String] = []     // parallel ID list
```

Every `insert()`/`batchInsert()` appends to both (lines 133-135, 166-171).

**The critical constraint — why you can't just clear these after a merge:**

`triggerMerge()` (line 763) does a **full graph rebuild** from allVectorData:

```swift
let snapshotCount = allIDsList.count
let merged = activeRecords(upperBound: snapshotCount)  // reads ALL of allVectorData
let newBase = try await buildIndex(vectors: merged.vectors, ids: merged.ids)
```

If allVectorData is cleared after a merge, the next merge rebuilds from an empty set — losing all base vectors. Simply clearing after merge **breaks correctness**.

**The correct fix: change what `triggerMerge` reads from.**

Instead of reading from `allVectorData` for the base+delta portion of the merge, extract live vectors directly from the existing `base` and `delta` `ANNSIndex` actors. Then `allVectorData` only needs to hold the **pre-delta pending log** (vectors inserted but not yet flushed into any index). That portion is bounded by `deltaCapacity * dim`.

**Scale context for Wax:** 100k vectors × 384 dims × 4 bytes = 153 MB of allVectorData in RAM, serialized to SQLite on every save. This is the OOM blocker.

---

## What you will implement

**Task 3.1:** Add `func exportLiveVectors() async -> (vectors: [[Float]], ids: [String])` to `ANNSIndex`. This lets `triggerMerge` extract the live (non-deleted) vectors from base/delta without access to allVectorData.

**Task 3.2:** Change `triggerMerge` to extract vectors from `base.exportLiveVectors()` + `delta.exportLiveVectors()` + `pendingVectors` instead of reading from `allVectorData`.

**Task 3.3:** After merge completes, compact `allVectorData` and `allIDsList` to only the post-merge tail (records added during the merge that didn't make it into the new base).

**Task 3.4:** In `delete(id:)`, immediately remove the vector from `allVectorData`/`allIDsList` if it's still in the pending log.

**Task 3.5:** Update `count`, `PersistedMeta`, `save()`, and `applyLoadedState()` to work correctly with the new trimmed arrays.

---

## Constraints (READ FIRST)

- Swift 6.0 strict concurrency — `ANNSIndex` is an actor; `exportLiveVectors()` must be `async`
- No change to the public API of `ANNSIndex` or `StreamingIndex`
- `exportLiveVectors()` can be `internal` (not `public`) — it is only called from `StreamingIndex`
- The save/load round-trip must remain correct: search must return the same results before and after save+load
- Tests must use Swift Testing (`import Testing`, `@Test`, `#expect`), NOT XCTest
- `SeededGenerator` already exists in `Tests/MetalANNSTests/SearchBufferPoolTests.swift` — reference that pattern
- Commit after each task. Do not batch tasks into one commit.

## Definition of Done

Phase 3 is complete when ALL of these are true:
1. `mergedVectorsAreEvictedFromHistory` test passes: after merging N vectors, `allIDsList.count` is bounded by `deltaCapacity` (not growing with N)
2. `deletedVectorsAreRemovedFromHistory` test passes: deleted pending-log vectors are removed immediately
3. Save+load round-trip test passes: search results are identical before and after save+reload
4. Full suite passes with zero regressions
5. Three clean commits on the branch
6. All Phase 3 checkboxes in `tasks/wax-readiness-todo.md` marked `[x]`

---

## Task 3.1: Add exportLiveVectors to ANNSIndex

**Modifies:** `Sources/MetalANNS/ANNSIndex.swift`

Read `ANNSIndex.swift` fully first. The key properties:
- `vectors: (any VectorStorage)?` — holds the GPU-side vector buffer
- `idMap: IDMap` — maps internalID (UInt32) ↔ externalID (String)
- `softDeletion: SoftDeletion` — tracks deleted internal IDs
- `isBuilt: Bool` — false until `build()` has been called

Add this method to `ANNSIndex`:

```swift
/// Extracts all live (non-deleted) vectors from the index.
/// Used by StreamingIndex.triggerMerge to build a new merged base
/// without needing the allVectorData append-log.
func exportLiveVectors() async -> (vectors: [[Float]], ids: [String]) {
    guard isBuilt, let vectors else {
        return ([], [])
    }

    var resultVectors: [[Float]] = []
    var resultIDs: [String] = []
    resultVectors.reserveCapacity(vectors.count)
    resultIDs.reserveCapacity(vectors.count)

    for i in 0..<vectors.count {
        let internalID = UInt32(i)
        guard !softDeletion.isDeleted(internalID),
              let externalID = idMap.externalID(for: internalID) else {
            continue
        }
        resultVectors.append(vectors.vector(at: i))
        resultIDs.append(externalID)
    }

    return (resultVectors, resultIDs)
}
```

**Check IDMap's API:** look for `externalID(for:)` or similar. It may be named `externalID(for internalID: UInt32) -> String?`. Adapt the call accordingly.

**Check SoftDeletion's API:** look for `isDeleted(_ id: UInt32) -> Bool` or similar. Adapt.

**Check VectorStorage's API:** `vector(at: Int) -> [Float]` should exist on `VectorStorage`. Confirm.

### Commit

```bash
git add Sources/MetalANNS/ANNSIndex.swift
git commit -m "feat: add exportLiveVectors to ANNSIndex for StreamingIndex merge optimization"
```

---

## Task 3.2 & 3.3: Change triggerMerge + compact allVectorData

**Creates:** `Tests/MetalANNSTests/StreamingIndexMemoryTests.swift`
**Modifies:** `Sources/MetalANNS/StreamingIndex.swift`

### Step 1 — Write the failing tests

Create `Tests/MetalANNSTests/StreamingIndexMemoryTests.swift`:

```swift
import Testing
import Foundation
@testable import MetalANNS
@testable import MetalANNSCore

private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state
    }
}

@Suite("StreamingIndex Memory Tests")
struct StreamingIndexMemoryTests {

    @Test func mergedVectorsAreEvictedFromHistory() async throws {
        let deltaCapacity = 5
        let config = StreamingConfiguration(
            deltaCapacity: deltaCapacity,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 4, metric: .cosine)
        )
        let index = StreamingIndex(config: config)

        let dim = 32
        var rng = SeededGenerator(state: 42)

        // Insert 3x deltaCapacity vectors — triggers multiple merges
        for i in 0..<(deltaCapacity * 3) {
            let vector = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
            try await index.insert(vector, id: "vec-\(i)")
        }
        try await index.flush()

        // After merging all vectors into base, allIDsList should only contain
        // the post-merge tail — NOT all 15 vectors ever inserted.
        // Bound: <= deltaCapacity (pending records that didn't trigger a full merge)
        let historyCount = await index.allIDsListCountForTesting
        #expect(
            historyCount <= deltaCapacity,
            "allIDsList has \(historyCount) items after merge — expected <= \(deltaCapacity). allVectorData is not being evicted."
        )
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

        // Delete 5 vectors while they're still in pending (no merge yet)
        for i in 0..<5 {
            try await index.delete(id: "vec-\(i)")
        }

        // allIDsList should not retain deleted pending entries
        let historyCount = await index.allIDsListCountForTesting
        #expect(historyCount == 5, "Expected 5 remaining in pending log, got \(historyCount)")
        #expect(await index.count == 5)
    }

    @Test func saveLoadRoundTripCorrectAfterEviction() async throws {
        let config = StreamingConfiguration(
            deltaCapacity: 5,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(degree: 4, metric: .cosine)
        )
        let index = StreamingIndex(config: config)

        let dim = 16
        var rng = SeededGenerator(state: 99)
        var vectors: [[Float]] = []
        for i in 0..<12 {
            let v = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
            vectors.append(v)
            try await index.insert(v, id: "vec-\(i)")
        }
        try await index.flush()

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("streaming-roundtrip-\(UUID().uuidString)")
        try await index.save(to: tempDir)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let loaded = try await StreamingIndex.load(from: tempDir)

        // Loaded index should return the same count
        #expect(await loaded.count == 12)

        // Search should return results from the base (pre-merge vectors)
        let query = vectors[0]
        let results = try await loaded.search(query: query, k: 3)
        #expect(results.count == 3)
        #expect(results[0].id == "vec-0", "Top result should be the query vector itself")
    }
}
```

### Step 2 — Add testing hook to StreamingIndex

The tests reference `index.allIDsListCountForTesting`. Add this to `StreamingIndex`:

```swift
// Add inside StreamingIndex (internal — not public):
var allIDsListCountForTesting: Int {
    allIDsList.count
}
```

### Step 3 — Run to verify tests fail

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/StreamingIndexMemoryTests 2>&1 | tail -30
```

**Expected:** `mergedVectorsAreEvictedFromHistory` FAILS — `allIDsList` has 15 items (all ever-inserted), not <= 5.

### Step 4 — Change triggerMerge to extract from base+delta

Read `triggerMerge()` in `StreamingIndex.swift` (starting at line 763). The current structure is:

```swift
private func triggerMerge() async throws {
    // ...
    let snapshotCount = allIDsList.count
    let merged = activeRecords(upperBound: snapshotCount)  // ← reads allVectorData
    let newBase = try await buildIndex(vectors: merged.vectors, ids: merged.ids)
    // ...
    let tail = activeRecords(in: snapshotCount..<allIDsList.count)
    // ...
}
```

Replace with:

```swift
private func triggerMerge() async throws {
    guard !_isMerging else { return }
    _isMerging = true
    defer { _isMerging = false }

    // Snapshot the boundary: records at indices >= snapshotCount were added
    // concurrently during this merge and form the "tail".
    let snapshotCount = allIDsList.count

    // Step 1: Collect all live vectors from existing base + delta + pending.
    // This replaces the old activeRecords(upperBound: snapshotCount) which
    // read the unbounded allVectorData append-log.
    var mergeVectors: [[Float]] = []
    var mergeIDs: [String] = []

    if let base {
        let (bv, bi) = await base.exportLiveVectors()
        mergeVectors.append(contentsOf: bv)
        mergeIDs.append(contentsOf: bi)
    }

    if let delta {
        let (dv, di) = await delta.exportLiveVectors()
        mergeVectors.append(contentsOf: dv)
        mergeIDs.append(contentsOf: di)
    }

    // Pending vectors are the pre-delta buffer — add non-deleted ones
    for (v, id) in zip(pendingVectors, pendingIDs) where !deletedIDs.contains(id) {
        mergeVectors.append(v)
        mergeIDs.append(id)
    }

    // Compute tail from the append-log BEFORE compacting it.
    // Tail = records added to allIDsList during this merge (indices >= snapshotCount).
    // These were inserted concurrently and are not yet in base/delta.
    let tailCount = allIDsList.count - snapshotCount
    var tailVectors: [[Float]] = []
    var tailIDs: [String] = []
    if tailCount > 0, let dim = vectorDimension {
        let tailIDsSlice = allIDsList.suffix(tailCount)
        let tailDataStart = allVectorData.count - tailCount * dim
        let tailDataSlice = allVectorData.suffix(tailCount * dim)
        tailIDs = Array(tailIDsSlice)
        for i in 0..<tailCount {
            let start = i * dim
            let v = Array(tailDataSlice[tailDataSlice.startIndex + start ..< tailDataSlice.startIndex + start + dim])
            tailVectors.append(v)
        }
    }

    guard !mergeIDs.isEmpty else {
        base = nil; delta = nil
        idInBase.removeAll(); idInDelta.removeAll()
        pendingVectors.removeAll(keepingCapacity: true)
        pendingIDs.removeAll(keepingCapacity: true)
        // Compact to just tail
        allIDsList = tailIDs
        allVectorData = tailVectors.flatMap { $0 }
        return
    }

    guard mergeIDs.count >= 2 else {
        base = nil; delta = nil
        idInBase.removeAll(); idInDelta.removeAll()
        pendingVectors = mergeVectors
        pendingIDs = mergeIDs
        // Compact: pending = mergeIDs + tail (both small)
        allIDsList = mergeIDs + tailIDs
        allVectorData = (mergeVectors + tailVectors).flatMap { $0 }
        return
    }

    let newBase = try await buildIndex(vectors: mergeVectors, ids: mergeIDs)

    base = newBase
    idInBase = Set(mergeIDs)
    delta = nil
    idInDelta.removeAll()
    pendingVectors.removeAll(keepingCapacity: true)
    pendingIDs.removeAll(keepingCapacity: true)
    if let metrics { await metrics.recordMerge() }

    // Compact: allVectorData and allIDsList now only track the tail
    // (records added during this merge). They'll grow again with future inserts,
    // but reset after each merge — bounded by rate of insertion, not total count.
    allIDsList = tailIDs
    allVectorData = tailVectors.flatMap { $0 }

    // Handle tail: route into delta (if 2+) or pending (if 1)
    let filteredTail = zip(tailVectors, tailIDs).filter { !deletedIDs.contains($0.1) }
    let fTailVectors = filteredTail.map(\.0)
    let fTailIDs = filteredTail.map(\.1)

    if fTailIDs.count >= 2 {
        let newDelta = try await buildIndex(vectors: fTailVectors, ids: fTailIDs)
        delta = newDelta
        idInDelta = Set(fTailIDs)
    } else if fTailIDs.count == 1 {
        pendingVectors = fTailVectors
        pendingIDs = fTailIDs
    }

    lastBackgroundMergeError = nil
}
```

**Note on the `count` property:** `count = allIDsList.count - deletedIDs.count` is now wrong since `allIDsList` is just the pending log. Change it to use `allIDs`:

```swift
public var count: Int {
    allIDs.count - deletedIDs.count
}
```

`allIDs: Set<String>` already contains every ID ever inserted (never compacted), so this gives the correct live count.

### Step 5 — Fix delete to remove from pending log

In `delete(id:)`, after `deletedIDs.insert(id)`, add:

```swift
// Remove from pending log if still there (before it was indexed)
if let idx = allIDsList.firstIndex(of: id) {
    allIDsList.remove(at: idx)
    if let dim = vectorDimension {
        allVectorData.removeSubrange((idx * dim)..<((idx + 1) * dim))
    }
}
```

### Step 6 — Run tests

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/StreamingIndexMemoryTests 2>&1 | tail -30
```

**Expected:** All 3 tests PASS.

### Step 7 — Run full suite

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|Passed|Failed|error:)'
```

**Expected:** All pass. Pay special attention to:
- `StreamingIndexMergeTests` — exercises merge paths
- `StreamingIndexPersistenceTests` — exercises save/load
- `StreamingIndexSearchTests` — exercises search across base+delta+pending

**If `StreamingIndexPersistenceTests` fails:** The save/load path may need updating (see Task 3.4 below).

### Step 8 — Commit

```bash
git add Sources/MetalANNS/StreamingIndex.swift Tests/MetalANNSTests/StreamingIndexMemoryTests.swift
git commit -m "fix: evict merged vectors from StreamingIndex pending log to prevent unbounded growth"
```

---

## Task 3.4: Fix Save/Load if needed

**Depends on:** Task 3.2/3.3 results — only needed if `StreamingIndexPersistenceTests` fail

**Modifies:** `Sources/MetalANNS/StreamingIndex.swift` (PersistedMeta, save, applyLoadedState)

After compaction, `allIDsList` only contains the pending log. But `applyLoadedState` (line 540) sets:

```swift
self.idInBase = Set(meta.allIDsList.filter { !self.deletedIDs.contains($0) })
```

If `meta.allIDsList` is now just the pending log, `idInBase` would be wrong (empty). `idInBase` is used for metadata propagation in `setMetadata`.

**The fix:** Add `baseIDs: [String]` to `PersistedMeta` and update save/load:

**In PersistedMeta struct**, add:
```swift
let baseIDs: [String]
```

Update `CodingKeys` to include `case baseIDs`.

Update `init(from decoder:)` with backward-compat decode:
```swift
baseIDs = (try? container.decode([String].self, forKey: .baseIDs)) ?? []
```

Update `encode(to encoder:)`:
```swift
try container.encode(baseIDs, forKey: .baseIDs)
```

**In `save()`**, before creating `PersistedMeta`:
```swift
let baseIDsSnapshot = Array(idInBase)
```

Pass `baseIDs: baseIDsSnapshot` to the `PersistedMeta` init.

**In `applyLoadedState`**, fix `idInBase`:
```swift
self.allIDs = Set(meta.baseIDs).union(Set(meta.allIDsList))
self.idInBase = Set(meta.baseIDs).subtracting(Set(meta.deletedIDs))
```

**Also fix `validateLoadedMeta`**: the check `allVectorData.count == allIDsList.count * dim` is still valid since both are the pending log.

After fixing, re-run persistence tests:
```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/StreamingIndexPersistenceTests 2>&1 | tail -30
```

### Commit if changes were needed

```bash
git add Sources/MetalANNS/StreamingIndex.swift
git commit -m "fix: update StreamingIndex save/load to persist baseIDs separately from pending log"
```

---

## Key Files Reference

| File | Role | Touch? |
|------|------|--------|
| `Sources/MetalANNS/ANNSIndex.swift` | Add `exportLiveVectors()` | YES — add one method |
| `Sources/MetalANNS/StreamingIndex.swift` | Rewrite `triggerMerge`, fix `delete`, fix `count`, fix save/load | YES — core changes |
| `Tests/MetalANNSTests/StreamingIndexMemoryTests.swift` | New test file | YES — create |
| `Sources/MetalANNSCore/IDMap.swift` | Check API for `externalID(for:)` | Read only |
| `Sources/MetalANNSCore/SoftDeletion.swift` | Check API for `isDeleted(_:)` | Read only |
| `Sources/MetalANNSCore/VectorStorage.swift` | Check `vector(at:)` signature | Read only |

## Anti-Patterns to Avoid

- **Do NOT call `activeRecords(upperBound:)` in the new triggerMerge for the base portion** — that reads the old unbounded allVectorData. Use `exportLiveVectors()` instead.
- **Do NOT clear `allIDs: Set<String>`** — it's the O(1) dedup guard for `insert()`. Only `allIDsList` and `allVectorData` are compacted.
- **Do NOT change the public API** of `ANNSIndex` or `StreamingIndex`.
- **Read IDMap, SoftDeletion, VectorStorage first** — verify method names before writing `exportLiveVectors()`. The actual API may differ slightly from what the prompt assumes.

## Verification Checklist

Before marking Phase 3 complete:

- [ ] `allIDsListCountForTesting` after N merges <= `deltaCapacity` (pending only)
- [ ] `delete(id:)` removes pending-log entry from allIDsList and allVectorData
- [ ] `count` uses `allIDs.count - deletedIDs.count` (not allIDsList)
- [ ] Save/load round-trip: search results match before and after
- [ ] `idInBase` correctly restored on load from `meta.baseIDs`
- [ ] Full suite green: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
- [ ] No changes to any `.metal` shader files
- [ ] Two or three commits (one per logical task)
