# Phase 4: Native UInt64 ID Support

## Role
You are a Swift 6.0 systems engineer implementing native `UInt64` ID support in MetalANNS so Wax can use its `frameId: UInt64` primary key without any String conversion overhead.

## Branch
`grdb4`

## Context

### Why This Phase Exists
Wax's `VectorSearchEngine` protocol uses `frameId: UInt64` as its primary key. MetalANNS currently only supports `String` external IDs in `IDMap`. Without native UInt64 support, a Wax adapter would need to call `String(frameId)` on every insert and `UInt64(result.id)!` on every search result — allocating a new String for every single operation on the hot path.

### What Needs to Change (Nothing Is Pre-Done)

**`Sources/MetalANNSCore/IDMap.swift`** (70 lines):
- `IDMap` only has `String↔UInt32` mappings. No `numericToInternal`/`internalToNumeric` dictionaries. No `assign(numericID:)`, `internalID(forNumeric:)`, or `numericID(for:)` methods.
- `IDMap` is `Codable` via synthesis — adding new stored properties WITHOUT a custom `init(from:)` would **break loading existing persisted indexes** because the synthesized decoder uses `decode` (throws if key missing), not `decodeIfPresent`.

**`Sources/MetalANNSCore/SearchResult.swift`** (11 lines):
- Only has `id: String`, `score: Float`, `internalID: UInt32`. No `numericID: UInt64?`.

**`Sources/MetalANNS/ANNSIndex.swift`**:
- `insert(_:id:)` and `batchInsert(_:ids:)` only take `String`. No UInt64 overloads.
- Search post-processing at lines 629–634 calls `idMap.externalID(for: result.internalID)` and does a `compactMap` that **drops** results where `externalID == nil`. This will silently discard every result for a UInt64-inserted vector.
- `rangeSearch` has the same compactMap pattern at lines 761–766.

**`Tests/MetalANNSTests/IDMapTests.swift`** — Does NOT exist.
**`Tests/MetalANNSTests/ANNSIndexTests.swift`** — EXISTS (418 lines). Add UInt64 test here.

### Key Architecture Rules
- `SeededGenerator` is defined `private` in each test file that uses it. Define it locally in new test files — do NOT import from another test file.
- Swift Testing framework: `import Testing`, `@Suite`, `@Test`, `#expect`.
- Run tests with: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/<SuiteName> 2>&1 | tail -30`

---

## Constraints (Read Before Writing Any Code)

1. **TDD: write the failing test, verify it fails, then implement, then verify it passes.** Never write implementation before its test.
2. **`IDMap.Codable` backward compatibility is mandatory.** You MUST write a custom `init(from decoder:)` using `decodeIfPresent` for the two new dictionary fields. Omitting this breaks production index load/save.
3. **Do not change the `assign(externalID:)` / `internalID(for:)` / `externalID(for:)` method signatures.** These are stable API. Only add new overloads.
4. **Do not change `FullGPUSearch.swift`.** It returns `SearchResult(id: "", score:, internalID:)` — the ID resolution happens in ANNSIndex, not in the GPU kernel.
5. **Both String-keyed and UInt64-keyed entries coexist in the same index.** An index built with String IDs then extended with UInt64 inserts must work correctly.
6. **Commit after each task** (4.1, 4.2).

---

## Task 4.1: UInt64 Key Support in IDMap

### Step 1: Write the failing test

Create `Tests/MetalANNSTests/IDMapTests.swift`:

```swift
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

    @Test func uint64KeyAndStringKeyAreIndependentNamespaces() {
        var map = IDMap()
        // String "42" and UInt64 42 occupy separate namespaces but share
        // the same internal ID counter — so they get consecutive internal IDs.
        let strInternalID = map.assign(externalID: "42")
        let numInternalID = map.assign(numericID: 42)
        #expect(strInternalID != nil)
        #expect(numInternalID != nil)
        #expect(strInternalID != numInternalID, "Share counter but different slots")
        // Lookup in the wrong namespace returns nil
        #expect(map.internalID(for: "42") == strInternalID)
        #expect(map.internalID(forNumeric: 42) == numInternalID)
        #expect(map.externalID(for: numInternalID!) == nil, "UInt64 slot has no String ID")
        #expect(map.numericID(for: strInternalID!) == nil, "String slot has no UInt64 ID")
    }

    @Test func uint64EdgeValues() {
        var map = IDMap()
        let edgeCases: [UInt64] = [0, 1, UInt64.max - 1, 12345678901234]
        for val in edgeCases {
            let internalID = map.assign(numericID: val)
            #expect(internalID != nil, "assign(\(val)) returned nil")
            #expect(map.numericID(for: internalID!) == val, "round-trip failed for \(val)")
        }
    }

    @Test func duplicateUInt64KeyReturnsNil() {
        var map = IDMap()
        let first = map.assign(numericID: 99)
        let second = map.assign(numericID: 99)
        #expect(first != nil)
        #expect(second == nil, "Duplicate assign must return nil")
    }

    @Test func countIncludesBothStringAndUInt64Keys() {
        var map = IDMap()
        _ = map.assign(externalID: "a")
        _ = map.assign(externalID: "b")
        _ = map.assign(numericID: 1)
        _ = map.assign(numericID: 2)
        #expect(map.count == 4)
    }

    @Test func canAllocateReflectsSharedCounter() {
        var map = IDMap()
        // Insert 3 via String, 3 via UInt64 = 6 total internal IDs consumed
        for i in 0..<3 { _ = map.assign(externalID: "s\(i)") }
        for i in 0..<3 { _ = map.assign(numericID: UInt64(i)) }
        #expect(map.count == 6)
        #expect(map.canAllocate(1))
    }
}
```

### Step 2: Run the test to verify it FAILS

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/IDMapTests 2>&1 | tail -30
```

Expected: **FAIL** — `assign(numericID:)`, `internalID(forNumeric:)`, `numericID(for:)` do not exist.

### Step 3: Implement UInt64 support in IDMap.swift

Add the two new stored properties alongside the existing ones (after `nextID`):

```swift
private var numericToInternal: [UInt64: UInt32] = [:]
private var internalToNumeric: [UInt32: UInt64] = [:]
```

Update `count` to reflect both namespaces:

```swift
public var count: Int {
    externalToInternal.count + numericToInternal.count
}
```

Add the new methods in a new extension section before the `// MARK: - Persistence Reconstruction` section:

```swift
// MARK: - UInt64 Key Support

/// Assigns a new internal ID for a numeric (UInt64) key.
/// Returns nil if the numeric key already exists or capacity is exhausted.
public mutating func assign(numericID: UInt64) -> UInt32? {
    guard numericToInternal[numericID] == nil else { return nil }
    guard nextID < UInt32.max else { return nil }
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

**Add backward-compatible `Codable` implementation.** The struct currently relies on synthesized `Codable`. You MUST replace this with explicit `CodingKeys` + `init(from:)` + `encode(to:)` to ensure old persisted indexes (which have no `numericToInternal`/`internalToNumeric` keys) still load successfully:

```swift
// MARK: - Codable (explicit, for backward-compat when loading old indexes)
private enum CodingKeys: CodingKey {
    case externalToInternal, internalToExternal, nextID
    case numericToInternal, internalToNumeric
}

public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    externalToInternal = try c.decode([String: UInt32].self, forKey: .externalToInternal)
    internalToExternal = try c.decode([UInt32: String].self, forKey: .internalToExternal)
    nextID = try c.decode(UInt32.self, forKey: .nextID)
    // decodeIfPresent provides empty defaults when loading pre-Phase-4 indexes
    numericToInternal = try c.decodeIfPresent([UInt64: UInt32].self, forKey: .numericToInternal) ?? [:]
    internalToNumeric = try c.decodeIfPresent([UInt32: UInt64].self, forKey: .internalToNumeric) ?? [:]
}

public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(externalToInternal, forKey: .externalToInternal)
    try c.encode(internalToExternal, forKey: .internalToExternal)
    try c.encode(nextID, forKey: .nextID)
    try c.encode(numericToInternal, forKey: .numericToInternal)
    try c.encode(internalToNumeric, forKey: .internalToNumeric)
}
```

### Step 4: Run tests to verify PASS

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/IDMapTests 2>&1 | tail -30
```

Expected: **PASS** — all 6 IDMap tests green.

### Step 5: Run the full test suite to catch regressions

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -40
```

Expected: All previously passing tests still green.

### Step 6: Commit

```bash
git add Sources/MetalANNSCore/IDMap.swift Tests/MetalANNSTests/IDMapTests.swift
git commit -m "feat: add native UInt64 key support to IDMap for Wax frameId compatibility"
```

---

## Task 4.2: SearchResult + ANNSIndex Integration

### Step 1: Write the failing test

Add this test to `Tests/MetalANNSTests/ANNSIndexTests.swift`. Define `SeededGenerator` locally at the top of the file **only if it does not already exist in the file**. Check first with a search before adding it:

```swift
@Test("Insert and search with UInt64 ID returns numericID in results")
func insertAndSearchWithUInt64IDs() async throws {
    let dim = 16
    let baseVectors = makeVectors(count: 20, dim: dim, seedOffset: 1_000)
    let baseIDs = (0..<20).map { "base-\($0)" }

    let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
    try await index.build(vectors: baseVectors, ids: baseIDs)

    // Insert with UInt64 key
    let newVector = makeVectors(count: 1, dim: dim, seedOffset: 9_999)[0]
    try await index.insert(newVector, numericID: 9_001)

    // Search using the inserted vector as query — first result should be itself
    let results = try await index.search(query: newVector, k: 1)
    #expect(results.count == 1)
    #expect(results[0].numericID == 9_001, "Expected numericID 9001, got \(String(describing: results[0].numericID))")
    // String ID for a UInt64-keyed result should be empty (no String representation assigned)
    #expect(results[0].id.isEmpty, "UInt64-inserted nodes have no String ID")
}

@Test("String-keyed results have nil numericID; UInt64-keyed results have nil String id")
func stringAndUInt64ResultsCoexist() async throws {
    let dim = 16
    let baseVectors = makeVectors(count: 10, dim: dim, seedOffset: 500)
    let baseIDs = (0..<10).map { "str-\($0)" }

    let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
    try await index.build(vectors: baseVectors, ids: baseIDs)
    try await index.insert(makeVectors(count: 1, dim: dim, seedOffset: 88_888)[0], numericID: 42)

    // Search with broad k to get both String and UInt64 results
    let results = try await index.search(query: makeVectors(count: 1, dim: dim, seedOffset: 88_888)[0], k: 5)

    let numericResults = results.filter { $0.numericID != nil }
    let stringResults = results.filter { !$0.id.isEmpty }

    #expect(!numericResults.isEmpty, "At least one UInt64-keyed result expected")
    // String-keyed results must not have a numericID
    for r in stringResults {
        #expect(r.numericID == nil, "String-keyed result '\(r.id)' should have nil numericID")
    }
}
```

### Step 2: Run the test to verify it FAILS

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing 'MetalANNSTests/ANNSIndexTests/insertAndSearchWithUInt64IDs' 2>&1 | tail -20
```

Expected: **FAIL** — `insert(_:numericID:)` does not exist, `numericID` property not on `SearchResult`.

### Step 3: Add `numericID` to SearchResult

`Sources/MetalANNSCore/SearchResult.swift` is only 11 lines. Add the new optional field and update the initializer:

```swift
public struct SearchResult: Sendable {
    public let id: String
    public let score: Float
    public let internalID: UInt32
    public let numericID: UInt64?

    public init(id: String, score: Float, internalID: UInt32, numericID: UInt64? = nil) {
        self.id = id
        self.score = score
        self.internalID = internalID
        self.numericID = numericID
    }
}
```

The `numericID: UInt64? = nil` default ensures all existing `SearchResult(id:score:internalID:)` call sites compile without changes.

### Step 4: Add `insert(_:numericID:)` to ANNSIndex

Add this method in `Sources/MetalANNS/ANNSIndex.swift` immediately after the existing `insert(_:id:)` method (which ends around line 263):

```swift
/// Inserts a vector with a numeric (UInt64) key. For use by Wax's UInt64 frameId-based API.
public func insert(_ vector: [Float], numericID: UInt64) async throws {
    guard isBuilt, let vectors, let graph else {
        throw ANNSError.indexEmpty
    }
    guard !isReadOnlyLoadedIndex else {
        throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
    }
    guard vector.count == vectors.dim else {
        throw ANNSError.dimensionMismatch(expected: vectors.dim, got: vector.count)
    }
    if idMap.internalID(forNumeric: numericID) != nil {
        throw ANNSError.idAlreadyExists(String(numericID))
    }
    guard idMap.canAllocate(1) else {
        throw ANNSError.constructionFailed("Internal ID space exhausted")
    }

    let nextInternalID = Int(idMap.nextInternalID)
    guard nextInternalID < vectors.capacity, nextInternalID < graph.capacity else {
        throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
    }

    let graphVector = vectors is BinaryVectorBuffer ? Self.quantizeForHamming(vector) : vector
    let slot = nextInternalID
    let previousVectorCount = vectors.count
    let previousGraphCount = graph.nodeCount

    do {
        try vectors.insert(vector: graphVector, at: slot)
        if vectors.count < slot + 1 {
            vectors.setCount(slot + 1)
        }

        try IncrementalBuilder.insert(
            vector: graphVector,
            at: slot,
            into: graph,
            vectors: vectors,
            entryPoint: entryPoint,
            metric: configuration.metric,
            degree: configuration.degree
        )
        if graph.nodeCount < slot + 1 {
            graph.setCount(slot + 1)
        }
        hnsw = nil

        let repairConfig = configuration.repairConfiguration
        if repairConfig.enabled && repairConfig.repairInterval > 0 {
            pendingRepairIDs.append(UInt32(slot))
            if pendingRepairIDs.count >= repairConfig.repairInterval {
                try triggerRepair(throwOnFailure: false)
            }
        }

        guard let assignedID = idMap.assign(numericID: numericID), Int(assignedID) == slot else {
            throw ANNSError.constructionFailed("Failed to commit internal ID for numeric \(numericID)")
        }
    } catch {
        vectors.setCount(previousVectorCount)
        graph.setCount(previousGraphCount)
        let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
        let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
        try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)
        if let annError = error as? ANNSError { throw annError }
        throw ANNSError.constructionFailed("Incremental insert (numeric) failed: \(error)")
    }
}
```

### Step 5: Fix search post-processing in ANNSIndex to include numericID

Find the `compactMap` in `search(query:k:filter:metric:)` at approximately line 629 and update it:

**Before:**
```swift
let mapped = filtered.compactMap { result -> SearchResult? in
    guard let externalID = idMap.externalID(for: result.internalID) else {
        return nil
    }
    return SearchResult(id: externalID, score: result.score, internalID: result.internalID)
}
```

**After:**
```swift
let mapped = filtered.compactMap { result -> SearchResult? in
    let externalID = idMap.externalID(for: result.internalID) ?? ""
    let numericID = idMap.numericID(for: result.internalID)
    // Include result if it was registered under either ID type
    guard !externalID.isEmpty || numericID != nil else { return nil }
    return SearchResult(
        id: externalID,
        score: result.score,
        internalID: result.internalID,
        numericID: numericID
    )
}
```

Apply the same fix to the `rangeSearch` compactMap at approximately line 761 (identical pattern, same substitution).

### Step 6: Run tests to verify PASS

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/ANNSIndexTests 2>&1 | tail -40
```

Expected: All ANNSIndex tests pass, including the two new UInt64 tests.

### Step 7: Run full suite to confirm no regressions

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -40
```

Expected: All previously passing tests still green.

### Step 8: Commit

```bash
git add Sources/MetalANNSCore/SearchResult.swift \
         Sources/MetalANNS/ANNSIndex.swift \
         Tests/MetalANNSTests/ANNSIndexTests.swift
git commit -m "feat: add UInt64-keyed insert and search to ANNSIndex for Wax frameId integration"
```

---

## Definition of Done

All of the following must be true before declaring Phase 4 complete:

- [ ] `IDMapTests.swift` exists with 6 tests, all passing
- [ ] `insert(_ vector: [Float], numericID: UInt64) async throws` exists on `ANNSIndex`
- [ ] `SearchResult.numericID: UInt64?` exists (nil for String-keyed results, non-nil for UInt64-keyed)
- [ ] Existing String-ID tests still pass (no regressions)
- [ ] An index built with String IDs then extended with UInt64 inserts searches correctly for both
- [ ] `IDMap` Codable backward-compatibility: custom `init(from:)` uses `decodeIfPresent` for the two new dict fields
- [ ] Full `xcodebuild test` suite green

---

## Anti-Patterns to Avoid

- **Do NOT** rely on synthesized `Codable` after adding new stored properties to `IDMap`. Synthesized `init(from:)` would throw `keyNotFound` when loading old persisted indexes.
- **Do NOT** redefine existing `IDMap` methods (`assign(externalID:)`, etc.) — only add new overloads.
- **Do NOT** make `numericID` non-optional on `SearchResult`. The vast majority of use sites are String-keyed and would need `numericID: 0` as a default, which is a valid UInt64 value and therefore ambiguous.
- **Do NOT** silently convert UInt64→String inside `insert(_:numericID:)`. The whole point is zero String allocation on the insert path.
- **Do NOT** put both String and UInt64 lookup logic in `FullGPUSearch.swift` — ID resolution belongs in `ANNSIndex.search()` post-processing only.

---

## Verification Checklist

After completing both tasks, run each check and confirm pass:

```bash
# 1. IDMap unit tests
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/IDMapTests 2>&1 | grep -E "passed|failed|error"

# 2. ANNSIndex UInt64 tests
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/ANNSIndexTests 2>&1 | grep -E "passed|failed|error"

# 3. Full suite (no regressions)
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E "passed|failed|error"

# 4. Confirm Codable round-trip compiles (no synthesized Codable ambiguity)
xcodebuild build -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E "error:|Build succeeded"
```

All commands must show **no failures**.
