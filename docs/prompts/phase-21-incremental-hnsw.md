# Phase 21: Incremental HNSW Insertion

### Mission

Eliminate the `hnsw = nil` calls in `ANNSIndex.insert()` and `batchInsert()`. Instead of discarding skip layers on every insert and falling back to O(N) flat scan, maintain them incrementally using the standard HNSW insertion algorithm. This is a pure performance fix — external API is unchanged.

---

### Verified Codebase Facts

Read each file before touching it. These facts were verified against the current source:

| Fact | Source |
|------|--------|
| `HNSWLayers.layers/maxLayer/mL/entryPoint` are `let` | `HNSWLayers.swift:28-37` |
| `SkipLayer.nodeToLayerIndex/layerIndexToNode/adjacency` are already `var` | `HNSWLayers.swift:6-12` |
| `SkipLayer.adjacency` stores **global node IDs** (not layer-local) | Confirmed by `HNSWSearchCPU.greedySearchLayer()` which does `vectors[Int(neighborID)]` directly on values returned from `hnsw.neighbors()` |
| `HNSWLayers.neighbors(of:at:)` returns global node IDs from `adjacency[layerIndex]` | `HNSWLayers.swift:52-61` |
| `HNSWBuilder.assignLevel(mL:maxLayers:)` is `static func` (internal access) | `HNSWBuilder.swift:75` |
| `insert()` sets `hnsw = nil` at line 206 | `ANNSIndex.swift:206-207` |
| `batchInsert()` sets `hnsw = nil` at line 285 | `ANNSIndex.swift:285-286` |
| `triggerRepair()` sets `hnsw = nil` at line 338 | `ANNSIndex.swift:338-339` |
| Search path: when `hnsw != nil, quantizedHNSW == nil` → falls through to `HNSWSearchCPU.search()` | `ANNSIndex.swift:465,472,482` |
| `rebuildHNSWFromCurrentState()` does a full O(N²) rebuild | `ANNSIndex.swift:889-933` |

---

### TDD Implementation Order

Work strictly test-first. Do not write implementation code before the test that drives it exists and fails.

**Round 1** — unit-level (no `ANNSIndex`)
Write `HNSWInserterTests.swift` with the five tests below. All must fail to compile (type doesn't exist yet). Then implement `HNSWInserter` until they pass.

**Round 2** — integration-level
Add tests that go through `ANNSIndex`. Then wire `HNSWInserter` into `ANNSIndex`.

Run after every step:
```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E "PASS|FAIL|error:"
```

---

### Step 1: Make `HNSWLayers` Mutable

**File**: `Sources/MetalANNSCore/HNSWLayers.swift`

Change these four `let` declarations to `var`. No other changes.

```swift
public var layers: [SkipLayer]
public var maxLayer: Int
public var mL: Double
public var entryPoint: UInt32
```

Verify: the existing `HNSWTests.swift` still passes after this change.

---

### Step 2: Create `HNSWInserter.swift`

**File**: `Sources/MetalANNSCore/HNSWInserter.swift`

**Public interface:**
```swift
public enum HNSWInserter {
    /// Insert a single node into existing HNSW skip layers (layers 1+).
    /// Layer 0 / base graph is handled separately by IncrementalBuilder — do not touch it here.
    /// Throws if nodeID is already present in any skip layer.
    public static func insert(
        vector: [Float],
        nodeID: UInt32,
        into layers: inout HNSWLayers,
        vectorStorage: any VectorStorage,
        config: HNSWConfiguration,
        metric: Metric
    ) throws(ANNSError)
}
```

**Algorithm (implement exactly this):**

```
PRECONDITIONS:
  - config.enabled == true (caller checks this)
  - nodeID == vectorStorage.count - 1  (just inserted into VectorStorage)
  - vector.count == vectorStorage.dim

nodeLevel = HNSWBuilder.assignLevel(mL: layers.mL, maxLayers: config.maxLayers)

// nodeLevel == 0 means: this node only lives in layer 0 (the base graph).
// Layer 0 is IncrementalBuilder's domain. Nothing to do here.
if nodeLevel == 0 { return }

// ----- Greedy descent: find best entry at level nodeLevel -----
currentNodeID: UInt32 = layers.entryPoint
currentDist: Float = dist(vector, vectorStorage[currentNodeID])

for l = layers.maxLayer downTo nodeLevel + 1:
    // Greedy single-step descent within skip layer l
    improved = true
    while improved:
        improved = false
        for neighborID in adjacencyOf(currentNodeID, at: l, in: layers):
            d = dist(vector, vectorStorage[neighborID])
            if d < currentDist:
                currentDist = d
                currentNodeID = neighborID
                improved = true

// ----- Widen new skip layers if needed -----
if nodeLevel > layers.maxLayer:
    // Extend layers array
    while layers.layers.count < nodeLevel:
        layers.layers.append(SkipLayer())
    // New node becomes entry point for all new layers
    for l = layers.maxLayer + 1 ... nodeLevel:
        skipLayer = &layers.layers[l - 1]
        skipLayer.nodeToLayerIndex[nodeID] = 0
        skipLayer.layerIndexToNode = [nodeID]
        skipLayer.adjacency = [[]]        // no neighbors yet at new top layers
    layers.maxLayer = nodeLevel
    layers.entryPoint = nodeID

// ----- Insert node into layers nodeLevel downTo 1 -----
for l = nodeLevel downTo 1:
    skipLayer = &layers.layers[l - 1]

    // Find M nearest existing neighbors at this layer
    var candidates: [(nodeID: UInt32, dist: Float)] = []
    for existingID in skipLayer.layerIndexToNode:
        candidates.append((existingID, dist(vector, vectorStorage[existingID])))
    candidates.sort by dist ascending
    let neighbors = Array(candidates.prefix(config.M)).map(\.nodeID)

    // Add new node to this skip layer
    let newLayerIndex = UInt32(skipLayer.layerIndexToNode.count)
    skipLayer.nodeToLayerIndex[nodeID] = newLayerIndex
    skipLayer.layerIndexToNode.append(nodeID)
    skipLayer.adjacency.append(neighbors)  // adjacency stores global nodeIDs

    // Add bidirectional edges (with M-pruning)
    for neighborID in neighbors:
        let nbrLayerIdx = Int(skipLayer.nodeToLayerIndex[neighborID]!)
        var nbrAdj = skipLayer.adjacency[nbrLayerIdx]
        nbrAdj.append(nodeID)
        if nbrAdj.count > config.M:
            // Keep M closest to the neighbor
            let nbrVec = vectorStorage.vector(at: Int(neighborID))
            nbrAdj = nbrAdj
                .map { (id: $0, d: dist(nbrVec, vectorStorage.vector(at: Int($0)))) }
                .sorted { $0.d < $1.d }
                .prefix(config.M)
                .map(\.id)
        skipLayer.adjacency[nbrLayerIdx] = nbrAdj

// Helper: adjacencyOf(nodeID, at layer, in layers)
func adjacencyOf(_ nodeID: UInt32, at layer: Int, in layers: HNSWLayers) -> [UInt32]:
    guard let layerIdx = layers.layers[layer-1].nodeToLayerIndex[nodeID] else { return [] }
    return layers.layers[layer-1].adjacency[Int(layerIdx)]
```

**Critical data layout notes:**
- `SkipLayer.adjacency[localIndex]` contains an array of **global node IDs** (not layer-local). This matches how `HNSWSearchCPU` reads them: `vectors[Int(neighborID)]` directly.
- `SIMDDistance.distance(_:_:metric:)` is the correct distance function to use throughout.
- Do not use `HNSWSearchCPU.greedySearchLayer()` — it requires a full `[[Float]]` extraction and is intended for search, not insertion.
- Accessing `vectorStorage.vector(at: Int(neighborID))` for any node that has already been inserted is safe because `nodeID` was just assigned the next slot.

---

### Step 3: Write `HNSWInserterTests.swift`

**File**: `Tests/MetalANNSTests/HNSWInserterTests.swift`

```swift
import Testing
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("HNSWInserter Unit Tests")
struct HNSWInserterTests {

    // Helpers
    private func randomVectors(count: Int, dim: Int, seed: UInt64 = 42) -> [[Float]] { ... }
    private func bruteForceTopK(query: [Float], vectors: [[Float]], k: Int, metric: Metric) -> [Int] { ... }
    private func recall(results: [SearchResult], groundTruth: [Int], ids: [String], k: Int) -> Double { ... }

    // TEST 1: levelAssignment
    // HNSWBuilder.assignLevel produces values in 0...maxLayers, and at least some > 0 over 1000 trials.
    @Test func levelAssignment() { ... }

    // TEST 2: insertSingleNodeIntoExistingLayers
    // Build a small HNSWLayers from scratch (5 nodes, 2 skip layers) using HNSWBuilder.buildLayers.
    // Call HNSWInserter.insert for node 5.
    // Assert: node 5 appears in layers.layers[l-1].nodeToLayerIndex for every l in 1...nodeLevel.
    // Assert: each neighbor's adjacency contains node 5 (bidirectional edges).
    @Test func insertSingleNodeIntoExistingLayers() throws { ... }

    // TEST 3: entryPointUpdated
    // Start with layers.maxLayer = 1. Force-loop HNSWInserter.insert until a node gets nodeLevel == 2
    // (max 2000 tries). Assert layers.maxLayer == 2 and layers.entryPoint == that nodeID.
    @Test func entryPointUpdated() throws { ... }

    // TEST 4: insertAndSearchRecall
    // Build ANNSIndex with 200 float32 vectors (dim=32, L2, CPU-only via MetalContext? = nil).
    // Insert 50 more one-by-one via index.insert().
    // After all inserts: await index.isHNSWBuilt == true.
    // Run 20 queries, compute recall@10 vs brute force.
    // #expect(recall > 0.75)
    @Test func insertAndSearchRecall() async throws { ... }

    // TEST 5: recallVsBatchBuild
    // Build index A: build() with 200 vectors.
    // Build index B: build() with 150 vectors, then insert() 50 more incrementally.
    // For 20 queries, measure recall@10 for both.
    // #expect(recallB >= recallA - 0.10)  // incremental recall within 10pp of batch
    @Test func recallVsBatchBuild() async throws { ... }
}
```

**Acceptance thresholds** (non-negotiable):
- `insertAndSearchRecall`: recall@10 > 0.75 (L2 metric, random gaussian vectors)
- `recallVsBatchBuild`: gap < 10 percentage points

Use `metric: .l2`, `config: HNSWConfiguration(enabled: true, M: 8, maxLayers: 6)` consistently across recall tests so results are comparable.

---

### Step 4: Wire into `ANNSIndex.swift`

**4a. Add `isHNSWBuilt` (for tests)**

```swift
// ANNSIndex.swift — inside the actor body
public var isHNSWBuilt: Bool { hnsw != nil }
```

**4b. `insert()` — replace lines 206-207**

```swift
// Before:
hnsw = nil
quantizedHNSW = nil

// After:
if var liveHNSW = hnsw, configuration.hnswConfiguration.enabled {
    try HNSWInserter.insert(
        vector: vector,
        nodeID: UInt32(slot),
        into: &liveHNSW,
        vectorStorage: vectors,
        config: configuration.hnswConfiguration,
        metric: configuration.metric
    )
    hnsw = liveHNSW
    quantizedHNSW = nil   // quantized falls back to regular HNSW until next explicit rebuild
} else {
    hnsw = nil
    quantizedHNSW = nil
}
```

**4c. `batchInsert()` — replace lines 285-286**

After `BatchIncrementalBuilder.batchInsert()` completes, the inserted `slots` are contiguous starting from `startSlot`. Call `HNSWInserter.insert()` per slot:

```swift
// After:
if var liveHNSW = hnsw, configuration.hnswConfiguration.enabled {
    for (offset, vector) in vectors.enumerated() {
        try HNSWInserter.insert(
            vector: vector,
            nodeID: UInt32(slots[offset]),
            into: &liveHNSW,
            vectorStorage: vectorStorage,
            config: configuration.hnswConfiguration,
            metric: configuration.metric
        )
    }
    hnsw = liveHNSW
    quantizedHNSW = nil
} else {
    hnsw = nil
    quantizedHNSW = nil
}
```

**4d. `triggerRepair()` — replace lines 338-339**

Repair mutates graph topology significantly. Rebuild from scratch rather than attempting incremental update:

```swift
// Replace:
hnsw = nil
quantizedHNSW = nil

// With:
try rebuildHNSWFromCurrentState()
```

---

### Step 5: Verify No Regressions

Run the full test suite. Every existing test must pass:

```
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -20
```

Pay particular attention to:
- `HNSWTests.swift` — existing layer/neighbor tests
- `QuantizedHNSWBuilderTests.swift` — quantized path must be unaffected
- `QuantizedHNSWIntegrationTests.swift` — end-to-end search with quantized path
- Any streaming index tests — `StreamingIndex` calls `insert()` internally

---

### Definition of Done

- [ ] `HNSWLayers.layers/maxLayer/mL/entryPoint` are `var`
- [ ] `HNSWInserter.swift` compiles with zero warnings under `swift build` (Swift logic) and `xcodebuild` (full)
- [ ] `ANNSIndex.insert()` never sets `hnsw = nil` when HNSW is live and enabled
- [ ] `ANNSIndex.batchInsert()` never sets `hnsw = nil` when HNSW is live and enabled
- [ ] `triggerRepair()` calls `rebuildHNSWFromCurrentState()` instead of nil
- [ ] All 5 new tests pass with thresholds met
- [ ] All pre-existing tests pass (zero regressions)
- [ ] No `// TODO`, no dead code, no commented-out blocks in committed files

---

### What Not To Do

- Do not call `rebuildHNSWFromCurrentState()` inside `insert()` or `batchInsert()` — that is O(N²) and defeats the purpose
- Do not use `HNSWSearchCPU.greedySearchLayer()` inside `HNSWInserter` — it requires a full `[[Float]]` extraction; use `vectorStorage.vector(at:)` directly
- Do not store layer-local indices in `adjacency` — always store global node IDs
- Do not make `HNSWInserter` `async` — the insertion logic is synchronous
