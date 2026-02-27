# Phase 14 Execution Prompt: Online Graph Repair (Dynamic Maintenance)

---

## System Context

You are implementing **Phase 14** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1–13 are complete. The package compiles under `swift-tools-version: 6.2` with typed throws (`throws(ANNSError)`) and `@concurrent` search methods. All tests pass.

You are adding **online graph repair** — a mechanism that restores graph quality after incremental insertions without requiring a full rebuild. Currently, vectors inserted after `build()` are connected via greedy beam search + reverse edge attachment (`IncrementalBuilder`), which finds *good-enough* neighbors but not optimal ones. After many inserts, recall degrades. Graph repair runs localized NN-Descent on recently-inserted neighborhoods to discover better edges.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/todo.md`

**Your communication contract:**
1. **`tasks/todo.md` is your shared state.** Check off `[x]` items as you complete them.
2. **Write notes under every task** — especially for decision points and any issues.
3. **Update `Last Updated`** at the top of todo.md after each task completes.
4. **When done, fill in the "Phase 14 Complete — Signal" section** at the bottom of todo.md.
5. **Do NOT modify the "Orchestrator Review Checklist"** section — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **TDD cycle for every task**: Write test → run to see it fail (RED) → implement → run to see it pass (GREEN) → commit. No exceptions.
2. **Swift 6 strict concurrency**: All new types must be `Sendable`. Use typed throws (`throws(ANNSError)`) on all new public functions.
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest.
4. **Build with `xcodebuild`**, never `swift build` or `swift test`. Metal shaders are not compiled by SPM CLI.
5. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
6. **Commit after every task** with the exact conventional commit message specified in the todo.
7. **CPU-only repair**. Graph repair runs on CPU using `SIMDDistance.distance()`. No Metal shaders for repair. GPU graph construction (full rebuild) is already handled by `IndexCompactor.compact()` — repair is the lightweight alternative.
8. **Do NOT modify existing public API signatures**. Only add new fields to `IndexConfiguration` and new methods on `ANNSIndex`. Existing callers must not break.

---

## Xcodebuild Commands

Build:
```bash
xcodebuild build -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation 2>&1 | tail -20
```

Test:
```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation 2>&1 | tail -40
```

---

## The Problem: Why Repair Is Needed

### Current Insert Flow (IncrementalBuilder)

When you call `ANNSIndex.insert()`, `IncrementalBuilder` does:
1. **Beam search** from `entryPoint` to find the new vector's nearest neighbors in the existing graph
2. **Forward edges**: Set the new node's neighbor list to these nearest neighbors
3. **Reverse edges**: For each found neighbor, if the new vector is closer than that neighbor's worst current neighbor, replace it
4. **Fallback**: If no reverse edges were created, force-attach to the entry point

This is **greedy and local** — the new node only sees neighbors reachable via the current graph's beam search. It misses:
- Neighbors that exist but aren't reachable from the entry point along the current graph path
- Better connections between existing nodes that the new node's insertion could facilitate
- Cross-cluster connections that would improve overall graph connectivity

### The Consequence

After 50-100 inserts, recall degrades measurably. The existing test `insertRecallDegradation` tolerates up to 5% degradation, but in production with thousands of inserts, degradation compounds.

### The Solution: Localized NN-Descent

Instead of a full rebuild (expensive — O(N²) distance computations), run NN-Descent on just the **affected neighborhoods**:
1. Collect all recently-inserted node IDs
2. Expand to their 2-hop neighborhoods (the nodes within 2 edges)
3. Run NN-Descent locally on this subgraph
4. Update edges that improved

This is ~O(k²) where k is the neighborhood size, amortized over `repairInterval` inserts.

---

## Current Codebase Snapshot (What You're Building On)

### IndexConfiguration (current)
```swift
public struct IndexConfiguration: Sendable, Codable {
    public var degree: Int
    public var metric: Metric
    public var efConstruction: Int
    public var efSearch: Int
    public var maxIterations: Int
    public var useFloat16: Bool
    public var convergenceThreshold: Float

    public static let `default` = IndexConfiguration(
        degree: 32, metric: .cosine, efConstruction: 100,
        efSearch: 64, maxIterations: 20, useFloat16: false,
        convergenceThreshold: 0.001
    )
}
```

### ANNSIndex State (relevant fields)
```swift
public actor ANNSIndex {
    private var configuration: IndexConfiguration
    private var context: MetalContext?
    private var vectors: (any VectorStorage)?
    private var graph: GraphBuffer?
    private var idMap: IDMap
    private var softDeletion: SoftDeletion
    private var metadataStore: MetadataStore
    private var entryPoint: UInt32
    private var isBuilt: Bool
    private var isReadOnlyLoadedIndex: Bool
    // ...
}
```

### GraphBuffer API (what repair will use)
```swift
public final class GraphBuffer: @unchecked Sendable {
    public let degree: Int
    public let capacity: Int
    public private(set) var nodeCount: Int

    public func neighborIDs(of nodeID: Int) -> [UInt32]
    public func neighborDistances(of nodeID: Int) -> [Float]
    public func setNeighbors(of nodeID: Int, ids: [UInt32], distances: [Float]) throws(ANNSError)
    public func setCount(_ newCount: Int)
}
```

### SIMDDistance API (what repair will use for distances)
```swift
public enum SIMDDistance {
    public static func distance(_ a: [Float], _ b: [Float], metric: Metric) -> Float
    public static func distance(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, dim: Int, metric: Metric) -> Float
}
```

### NNDescentCPU.build() (the algorithm you're adapting for localized repair)
```swift
public enum NNDescentCPU {
    public static func build(
        vectors: [[Float]],
        degree: Int,
        metric: Metric,
        maxIterations: Int = 20,
        convergenceThreshold: Float = 0.001
    ) async throws(ANNSError) -> (graph: [[(UInt32, Float)]], entryPoint: UInt32)
}
```

Key algorithmic pattern from `NNDescentCPU.build()`:
1. Initialize with random neighbors
2. For each iteration:
   a. Build reverse adjacency lists
   b. For each node, collect forward + reverse neighbors as candidates
   c. For all candidate pairs, compute distance and try to insert into neighbor lists
   d. Count updates; stop when updates < convergenceThreshold × degree × nodeCount

### IncrementalBuilder.insert() (the function that triggers repair)
```swift
public enum IncrementalBuilder {
    public static func insert(
        vector: [Float],
        at internalID: Int,
        into graph: GraphBuffer,
        vectors: any VectorStorage,
        entryPoint: UInt32,
        metric: Metric,
        degree: Int
    ) throws(ANNSError)
}
```

### Key pattern: how IncrementalBuilder finds neighbors (beam search)
The `nearestNeighbors()` private method uses beam search with `ef = degree * 2`, visiting nodes via the graph starting from `entryPoint`. This is exactly the pattern repair will extend — but repair considers a wider neighborhood (2-hop instead of 1-hop).

---

## Architecture

```
ANNSIndex.insert() ──→ IncrementalBuilder.insert() ──→ append to pendingRepairIDs
                                                              │
                                              count >= repairInterval?
                                                              │ YES
                                                              ▼
                                              GraphRepairer.repair(recentIDs)
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                    Collect 2-hop              Run localized NN-Descent
                                    neighborhoods               on subgraph nodes
                                              │                               │
                                              └───────────────┬───────────────┘
                                                              ▼
                                                  Update edges atomically
                                                  (via graph.setNeighbors)
                                                              │
                                                              ▼
                                                  Clear pendingRepairIDs
```

---

## New Files to Create

### 1. `Sources/MetalANNSCore/RepairConfiguration.swift`

```swift
import Foundation

public struct RepairConfiguration: Sendable, Codable {
    /// Trigger repair every N inserts. Set to 0 to disable automatic repair.
    public var repairInterval: Int

    /// Number of hops to expand from each recent node to collect the repair neighborhood.
    /// 1 = direct neighbors only. 2 = neighbors of neighbors (recommended).
    public var repairDepth: Int

    /// Number of localized NN-Descent iterations per repair cycle.
    public var repairIterations: Int

    /// Whether automatic repair is enabled.
    public var enabled: Bool

    public static let `default` = RepairConfiguration(
        repairInterval: 100,
        repairDepth: 2,
        repairIterations: 5,
        enabled: true
    )

    public init(
        repairInterval: Int = 100,
        repairDepth: Int = 2,
        repairIterations: Int = 5,
        enabled: Bool = true
    ) {
        self.repairInterval = max(0, repairInterval)
        self.repairDepth = max(1, min(repairDepth, 3))
        self.repairIterations = max(1, min(repairIterations, 20))
        self.enabled = enabled
    }
}
```

### 2. `Sources/MetalANNSCore/GraphRepairer.swift`

This is the core of Phase 14. The algorithm:

```swift
import Foundation
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "GraphRepairer")

public enum GraphRepairer {

    /// Run localized NN-Descent on the neighborhoods surrounding `recentIDs`.
    ///
    /// - Parameters:
    ///   - recentIDs: Internal IDs of recently-inserted nodes
    ///   - vectors: Vector storage for distance computation
    ///   - graph: The graph buffer to repair (mutated in-place)
    ///   - config: Repair parameters
    ///   - metric: Distance metric
    /// - Returns: Number of edges updated
    @discardableResult
    public static func repair(
        recentIDs: [UInt32],
        vectors: any VectorStorage,
        graph: GraphBuffer,
        config: RepairConfiguration,
        metric: Metric
    ) throws(ANNSError) -> Int {
        // Implementation below
    }
}
```

### Algorithm Detail

**Step 1: Collect repair neighborhood**

Starting from each `recentID`, walk `repairDepth` hops to collect all nodes in the local subgraph.

```
func collectNeighborhood(
    seeds: [UInt32],
    graph: GraphBuffer,
    depth: Int
) -> Set<UInt32>
```

- Start with `frontier = Set(seeds)`
- For each depth level: expand frontier by adding all neighbors of current frontier nodes
- Return the union of all visited nodes
- Skip `UInt32.max` sentinel values
- Bound check: skip nodes where `Int(nodeID) >= graph.nodeCount`

**Step 2: Run localized NN-Descent**

For each node in the neighborhood, try to improve its edges by considering pairs from its forward neighbors + reverse neighbors (within the neighborhood).

```
func localNNDescent(
    nodes: Set<UInt32>,
    vectors: any VectorStorage,
    graph: GraphBuffer,
    metric: Metric,
    iterations: Int
) throws(ANNSError) -> Int
```

The algorithm (adapted from `NNDescentCPU.build()`):

```
totalUpdates = 0
for iteration in 0..<iterations:
    // Build reverse lists (only for nodes in neighborhood)
    reverse: [UInt32: [UInt32]] = [:]
    for node in nodes:
        for neighborID in graph.neighborIDs(of: node):
            if nodes.contains(neighborID):
                reverse[neighborID, default: []].append(node)

    iterationUpdates = 0
    for node in nodes:
        // Candidates = forward neighbors + reverse neighbors (within neighborhood)
        let forward = graph.neighborIDs(of: node).filter { nodes.contains($0) && $0 != UInt32.max }
        let reverseList = reverse[node] ?? []
        let candidates = Set(forward.map { Int($0) } + reverseList.map { Int($0) })

        // Try all pairs
        let candidateArray = Array(candidates)
        for i in 0..<candidateArray.count:
            for j in (i+1)..<candidateArray.count:
                let a = candidateArray[i]
                let b = candidateArray[j]
                if a == b: continue

                let dist = SIMDDistance.distance(
                    vectors.vector(at: a),
                    vectors.vector(at: b),
                    metric: metric
                )

                if tryImproveEdge(node: a, candidate: b, distance: dist, graph: graph):
                    iterationUpdates += 1
                if tryImproveEdge(node: b, candidate: a, distance: dist, graph: graph):
                    iterationUpdates += 1

    totalUpdates += iterationUpdates
    logger.debug("GraphRepairer iteration \(iteration): \(iterationUpdates) updates")

    // Convergence: stop if fewer than 0.1% of possible edges updated
    let threshold = Float(graph.degree * nodes.count) * 0.001
    if Float(iterationUpdates) < threshold:
        logger.debug("GraphRepairer converged at iteration \(iteration + 1)")
        break

return totalUpdates
```

**Step 3: tryImproveEdge helper**

```
func tryImproveEdge(
    node: Int,
    candidate: Int,
    distance: Float,
    graph: GraphBuffer
) throws(ANNSError) -> Bool
```

Logic (mirrors `NNDescentCPU.tryInsert()`):
1. If `candidate == node`, return false
2. Read `graph.neighborIDs(of: node)` and `graph.neighborDistances(of: node)`
3. If `candidate` is already a neighbor, return false
4. Find the worst (highest distance) neighbor slot
5. If `distance < worstDistance`, replace that slot with `(candidate, distance)`
6. Sort the neighbor list by distance
7. Write back via `graph.setNeighbors(of: node, ids: ..., distances: ...)`
8. Return true

**CRITICAL**: Use `Int(nodeID)` consistently. `GraphBuffer` takes `Int` for node IDs, but stores `UInt32` in adjacency lists. Always convert.

### 3. `Tests/MetalANNSTests/GraphRepairTests.swift`

Test file — see Task details below.

---

## Files to Modify

### 4. `Sources/MetalANNS/IndexConfiguration.swift`

Add `repairConfiguration` field:

```swift
public struct IndexConfiguration: Sendable, Codable {
    // ... existing fields ...
    public var repairConfiguration: RepairConfiguration

    public static let `default` = IndexConfiguration(
        degree: 32, metric: .cosine, efConstruction: 100,
        efSearch: 64, maxIterations: 20, useFloat16: false,
        convergenceThreshold: 0.001,
        repairConfiguration: .default
    )

    public init(
        degree: Int = 32,
        metric: Metric = .cosine,
        efConstruction: Int = 100,
        efSearch: Int = 64,
        maxIterations: Int = 20,
        useFloat16: Bool = false,
        convergenceThreshold: Float = 0.001,
        repairConfiguration: RepairConfiguration = .default
    ) {
        // ... existing assignments ...
        self.repairConfiguration = repairConfiguration
    }
}
```

**Codable backward compatibility**: Since `RepairConfiguration` has a default, existing serialized configs that lack the field will fail to decode. Add a custom `init(from decoder:)`:

```swift
public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    degree = try container.decode(Int.self, forKey: .degree)
    metric = try container.decode(Metric.self, forKey: .metric)
    efConstruction = try container.decode(Int.self, forKey: .efConstruction)
    efSearch = try container.decode(Int.self, forKey: .efSearch)
    maxIterations = try container.decode(Int.self, forKey: .maxIterations)
    useFloat16 = try container.decode(Bool.self, forKey: .useFloat16)
    convergenceThreshold = try container.decode(Float.self, forKey: .convergenceThreshold)
    repairConfiguration = try container.decodeIfPresent(RepairConfiguration.self, forKey: .repairConfiguration) ?? .default
}
```

### 5. `Sources/MetalANNS/ANNSIndex.swift`

Add repair tracking and triggering:

**New stored properties** (inside the actor):
```swift
private var pendingRepairIDs: [UInt32] = []
```

**Modify `insert(_:id:)` — append to pendingRepairIDs and trigger repair**:

After the existing `IncrementalBuilder.insert()` call and graph count update, add:

```swift
// After the existing insert logic completes successfully:
let repairConfig = configuration.repairConfiguration
if repairConfig.enabled && repairConfig.repairInterval > 0 {
    pendingRepairIDs.append(UInt32(slot))
    if pendingRepairIDs.count >= repairConfig.repairInterval {
        try triggerRepair()
    }
}
```

**Modify `batchInsert(_:ids:)` — same pattern**:

After `BatchIncrementalBuilder.batchInsert()` succeeds, add:

```swift
let repairConfig = configuration.repairConfiguration
if repairConfig.enabled && repairConfig.repairInterval > 0 {
    for slot in slots {
        pendingRepairIDs.append(UInt32(slot))
    }
    if pendingRepairIDs.count >= repairConfig.repairInterval {
        try triggerRepair()
    }
}
```

**New private method `triggerRepair()`**:

```swift
private func triggerRepair() throws(ANNSError) {
    guard let vectors, let graph else { return }
    guard !pendingRepairIDs.isEmpty else { return }

    let idsToRepair = pendingRepairIDs
    pendingRepairIDs.removeAll(keepingCapacity: true)

    try GraphRepairer.repair(
        recentIDs: idsToRepair,
        vectors: vectors,
        graph: graph,
        config: configuration.repairConfiguration,
        metric: configuration.metric
    )
}
```

**Add a public manual repair method**:

```swift
/// Manually trigger graph repair on all pending nodes.
/// Useful after a burst of inserts when you want to immediately restore quality.
public func repair() throws(ANNSError) {
    guard !isReadOnlyLoadedIndex else {
        throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
    }
    guard isBuilt, let vectors, let graph else {
        throw ANNSError.indexEmpty
    }
    guard !pendingRepairIDs.isEmpty else {
        return
    }
    try triggerRepair()
}
```

**Modify `compact()` — clear pendingRepairIDs**:

After the existing compaction logic (after `self.softDeletion = SoftDeletion()`), add:

```swift
self.pendingRepairIDs.removeAll()
```

Rationale: Compaction rebuilds the entire graph via NNDescent, so pending repairs are moot.

**Modify `build()` — clear pendingRepairIDs**:

At the end of `build()`, after `self.isBuilt = true`:

```swift
self.pendingRepairIDs.removeAll()
```

**Initialize in `applyLoadedState()`**:

The `applyLoadedState()` method doesn't set `pendingRepairIDs` currently. Since it's initialized at `init()`, loaded indices start with empty pending list — this is correct. No change needed (loaded indices start fresh).

---

## Task Breakdown

### Task 1: Create `RepairConfiguration.swift` + test

**TDD RED**: Write test first in `GraphRepairTests.swift`:

```swift
import Testing
@testable import MetalANNSCore

@Suite("Graph Repair Tests")
struct GraphRepairTests {

    @Test("RepairConfiguration defaults are sensible")
    func repairConfigDefaults() {
        let config = RepairConfiguration.default
        #expect(config.repairInterval == 100)
        #expect(config.repairDepth == 2)
        #expect(config.repairIterations == 5)
        #expect(config.enabled == true)
    }

    @Test("RepairConfiguration clamps invalid values")
    func repairConfigClamping() {
        let config = RepairConfiguration(repairInterval: -5, repairDepth: 0, repairIterations: 100)
        #expect(config.repairInterval == 0)
        #expect(config.repairDepth == 1)      // min 1
        #expect(config.repairIterations == 20) // max 20
    }
}
```

**TDD GREEN**: Create `Sources/MetalANNSCore/RepairConfiguration.swift` as specified above.

**Commit**: `feat(repair): add RepairConfiguration struct with sensible defaults and clamping`

---

### Task 2: Implement `GraphRepairer.repair()` core + neighborhood collection test

**TDD RED**: Add to `GraphRepairTests.swift`:

```swift
@Test("Neighborhood collection expands correctly")
func neighborhoodCollection() async throws {
    // Build a small graph: 10 nodes, degree 4
    let vectors = (0..<10).map { i in (0..<8).map { d in sin(Float(i * 8 + d) * 0.173) } }
    let (graphData, _) = try await NNDescentCPU.build(
        vectors: vectors, degree: 4, metric: .l2, maxIterations: 5
    )

    let graphBuffer = try makeGraphBuffer(graphData, degree: 4)

    // Repair node 0 with depth 1: should include node 0 + its direct neighbors
    let result = try GraphRepairer.repair(
        recentIDs: [0],
        vectors: try makeVectorBuffer(vectors),
        graph: graphBuffer,
        config: RepairConfiguration(repairInterval: 1, repairDepth: 1, repairIterations: 1),
        metric: .l2
    )
    // Should succeed without error (result is update count, can be 0 or more)
    #expect(result >= 0)
}
```

Plus helpers `makeGraphBuffer` and `makeVectorBuffer` (reuse pattern from `IncrementalTests`).

**TDD GREEN**: Implement `GraphRepairer.repair()` with:
- `collectNeighborhood()` private method
- `localNNDescent()` private method
- `tryImproveEdge()` private method

**Commit**: `feat(repair): implement GraphRepairer with localized NN-Descent`

---

### Task 3: Test that repair actually improves recall

This is the **critical test** — it proves the feature works.

**TDD RED**: Add to `GraphRepairTests.swift`:

```swift
@Test("Repair improves recall after inserts")
func repairImprovesRecall() async throws {
    let initialCount = 200
    let insertCount = 100
    let dim = 16
    let degree = 8
    let metric: Metric = .cosine

    // Build initial graph
    let initialVectors = (0..<initialCount).map { i in
        (0..<dim).map { d in sin(Float(i * dim + d) * 0.173) + cos(Float(i * dim + d) * 0.071) }
    }
    let (graphData, entryPoint) = try await NNDescentCPU.build(
        vectors: initialVectors, degree: degree, metric: metric, maxIterations: 10
    )
    let vectorBuffer = try makeVectorBuffer(initialVectors, extraCapacity: insertCount)
    let graphBuffer = try makeGraphBuffer(graphData, degree: degree, extraCapacity: insertCount)

    // Insert 100 new vectors
    var allVectors = initialVectors
    var insertedIDs: [UInt32] = []
    for i in 0..<insertCount {
        let newVector = (0..<dim).map { d in sin(Float((initialCount + i) * dim + d) * 0.173) }
        let slot = initialCount + i
        try vectorBuffer.insert(vector: newVector, at: slot)
        vectorBuffer.setCount(slot + 1)

        try IncrementalBuilder.insert(
            vector: newVector, at: slot, into: graphBuffer,
            vectors: vectorBuffer, entryPoint: entryPoint,
            metric: metric, degree: degree
        )
        graphBuffer.setCount(slot + 1)
        allVectors.append(newVector)
        insertedIDs.append(UInt32(slot))
    }

    // Measure recall BEFORE repair
    let queries = Array(allVectors.prefix(20))
    let recallBefore = try await averageRecall(
        queries: queries, vectors: allVectors,
        graph: graphBuffer, entryPoint: Int(entryPoint),
        k: 10, ef: 64, metric: metric
    )

    // Run repair
    let updates = try GraphRepairer.repair(
        recentIDs: insertedIDs,
        vectors: vectorBuffer,
        graph: graphBuffer,
        config: RepairConfiguration(repairDepth: 2, repairIterations: 5),
        metric: metric
    )

    // Measure recall AFTER repair
    let recallAfter = try await averageRecall(
        queries: queries, vectors: allVectors,
        graph: graphBuffer, entryPoint: Int(entryPoint),
        k: 10, ef: 64, metric: metric
    )

    // Repair should improve (or at least not degrade) recall
    #expect(recallAfter >= recallBefore - 0.01, "Repair degraded recall: \(recallAfter) < \(recallBefore)")
    #expect(updates > 0, "Repair should have found some improvements")
}
```

The `averageRecall` helper computes exact brute-force k-NN and measures overlap with graph search results (same pattern as `IncrementalTests.averageRecall()`):

```swift
private func averageRecall(
    queries: [[Float]],
    vectors: [[Float]],
    graph: GraphBuffer,
    entryPoint: Int,
    k: Int,
    ef: Int,
    metric: Metric
) async throws -> Float {
    let graphData = (0..<graph.nodeCount).map { nodeID in
        let ids = graph.neighborIDs(of: nodeID)
        let distances = graph.neighborDistances(of: nodeID)
        return Array(zip(ids, distances))
    }

    var totalRecall: Float = 0
    for query in queries {
        let approxResults = try await BeamSearchCPU.search(
            query: query, vectors: vectors, graph: graphData,
            entryPoint: entryPoint, k: k, ef: ef, metric: metric
        )

        // Brute-force exact k-NN
        let exact = vectors.enumerated().map { (index, v) in
            (UInt32(index), SIMDDistance.distance(query, v, metric: metric))
        }.sorted { $0.1 < $1.1 }
        let exactTopK = Set(exact.prefix(k).map { $0.0 })

        let approxTopK = Set(approxResults.map(\.internalID))
        totalRecall += Float(exactTopK.intersection(approxTopK).count) / Float(k)
    }
    return totalRecall / Float(queries.count)
}
```

**TDD GREEN**: The implementation from Task 2 should make this pass. If it doesn't, debug the localized NN-Descent logic.

**Commit**: `test(repair): verify repair improves recall after batch inserts`

---

### Task 4: Test repair with deletions

**TDD RED**:

```swift
@Test("Repair handles deleted nodes correctly")
func repairWithDeletions() async throws {
    let count = 100
    let dim = 8
    let degree = 4
    let metric: Metric = .l2

    let vectors = (0..<count).map { i in (0..<dim).map { d in Float(i * dim + d) * 0.01 } }
    let (graphData, entryPoint) = try await NNDescentCPU.build(
        vectors: vectors, degree: degree, metric: metric, maxIterations: 5
    )
    let vectorBuffer = try makeVectorBuffer(vectors, extraCapacity: 20)
    let graphBuffer = try makeGraphBuffer(graphData, degree: degree, extraCapacity: 20)

    // Insert new nodes
    var insertedIDs: [UInt32] = []
    for i in 0..<10 {
        let newVector = (0..<dim).map { d in Float((count + i) * dim + d) * 0.01 }
        let slot = count + i
        try vectorBuffer.insert(vector: newVector, at: slot)
        vectorBuffer.setCount(slot + 1)
        try IncrementalBuilder.insert(
            vector: newVector, at: slot, into: graphBuffer,
            vectors: vectorBuffer, entryPoint: entryPoint,
            metric: metric, degree: degree
        )
        graphBuffer.setCount(slot + 1)
        insertedIDs.append(UInt32(slot))
    }

    // Repair should not crash even if some neighbors point to out-of-range nodes
    let updates = try GraphRepairer.repair(
        recentIDs: insertedIDs,
        vectors: vectorBuffer,
        graph: graphBuffer,
        config: RepairConfiguration(repairDepth: 2, repairIterations: 3),
        metric: metric
    )
    #expect(updates >= 0)
}
```

**TDD GREEN**: Ensure `collectNeighborhood` and `localNNDescent` skip invalid node IDs and nodes >= nodeCount.

**Commit**: `test(repair): verify repair handles edge cases with deletions`

---

### Task 5: Test repair disabled

**TDD RED**:

```swift
@Test("Repair does nothing when disabled")
func repairDisabled() async throws {
    let vectors = (0..<50).map { i in (0..<8).map { d in Float(i * 8 + d) } }
    let (graphData, _) = try await NNDescentCPU.build(
        vectors: vectors, degree: 4, metric: .l2, maxIterations: 5
    )
    let graphBuffer = try makeGraphBuffer(graphData, degree: 4)

    // Save original state
    let originalNeighbors = (0..<50).map { graphBuffer.neighborIDs(of: $0) }

    // Repair with enabled=false should return 0 updates
    let config = RepairConfiguration(enabled: false)
    let updates = try GraphRepairer.repair(
        recentIDs: [0, 1, 2, 3, 4],
        vectors: try makeVectorBuffer(vectors),
        graph: graphBuffer,
        config: config,
        metric: .l2
    )
    #expect(updates == 0)

    // Graph should be unchanged
    for nodeID in 0..<50 {
        #expect(graphBuffer.neighborIDs(of: nodeID) == originalNeighbors[nodeID])
    }
}
```

**TDD GREEN**: At the top of `GraphRepairer.repair()`, add early return:

```swift
guard config.enabled else { return 0 }
guard !recentIDs.isEmpty else { return 0 }
```

**Commit**: `test(repair): verify repair respects enabled flag`

---

### Task 6: Integrate repair into ANNSIndex

**Modify `IndexConfiguration.swift`** — add `repairConfiguration` field as described above.

**Modify `ANNSIndex.swift`**:
- Add `pendingRepairIDs` property
- Modify `insert()` to track IDs and trigger repair
- Modify `batchInsert()` same
- Add `triggerRepair()` private method
- Add `repair()` public method
- Modify `compact()` to clear `pendingRepairIDs`
- Modify `build()` to clear `pendingRepairIDs`

**Commit**: `feat(repair): integrate GraphRepairer into ANNSIndex insert flow`

---

### Task 7: Integration test via ANNSIndex public API

```swift
@Test("ANNSIndex triggers repair after repairInterval inserts")
func indexIntegrationRepair() async throws {
    var config = IndexConfiguration(degree: 8, metric: .cosine)
    config.repairConfiguration = RepairConfiguration(repairInterval: 10, repairDepth: 2, repairIterations: 3)

    let index = ANNSIndex(configuration: config)
    let initialVectors = (0..<50).map { i in (0..<16).map { d in sin(Float(i * 16 + d) * 0.173) } }
    let initialIDs = (0..<50).map { "v_\($0)" }
    try await index.build(vectors: initialVectors, ids: initialIDs)

    // Insert 15 vectors (exceeds repairInterval=10, so repair should trigger)
    for i in 50..<65 {
        let vector = (0..<16).map { d in sin(Float(i * 16 + d) * 0.173) }
        try await index.insert(vector, id: "v_\(i)")
    }

    // Verify the inserted vectors are findable
    for i in 50..<65 {
        let query = (0..<16).map { d in sin(Float(i * 16 + d) * 0.173) }
        let results = try await index.search(query: query, k: 1)
        #expect(!results.isEmpty)
        #expect(results[0].id == "v_\(i)")
    }

    // Count should reflect all vectors
    let count = await index.count
    #expect(count == 65)
}

@Test("Manual repair via public API")
func manualRepair() async throws {
    var config = IndexConfiguration(degree: 8, metric: .l2)
    config.repairConfiguration = RepairConfiguration(repairInterval: 0, enabled: true) // auto-repair disabled (interval=0)

    let index = ANNSIndex(configuration: config)
    let initialVectors = (0..<50).map { i in (0..<8).map { d in Float(i * 8 + d) * 0.01 } }
    let initialIDs = (0..<50).map { "v_\($0)" }
    try await index.build(vectors: initialVectors, ids: initialIDs)

    // Insert 5 vectors — no auto-repair because interval=0
    for i in 50..<55 {
        let vector = (0..<8).map { d in Float(i * 8 + d) * 0.01 }
        try await index.insert(vector, id: "v_\(i)")
    }

    // Manual repair should not crash
    try await index.repair()

    // Verify vectors are findable
    for i in 50..<55 {
        let query = (0..<8).map { d in Float(i * 8 + d) * 0.01 }
        let results = try await index.search(query: query, k: 1)
        #expect(!results.isEmpty)
    }
}
```

**Commit**: `test(repair): add ANNSIndex integration tests for automatic and manual repair`

---

### Task 8: Verify full test suite passes

Run the complete test suite. Fix any regressions. Common issues:
- `IndexConfiguration.init()` signature change may break call sites that don't provide `repairConfiguration` — the default parameter handles this, but verify.
- Codable backward compatibility: ensure `IndexConfiguration` can still decode files saved by Phase 12 code (they won't have `repairConfiguration` key).
- `pendingRepairIDs` initialization: make sure it's `[]` in all code paths (init, load, loadMmap, loadDiskBacked).

**Commit**: `chore(repair): verify zero regressions in full test suite`

---

## Task Execution Order

Execute tasks **strictly in this order**:

1. **Task 1** → RepairConfiguration + test → RED → GREEN → commit
2. **Task 2** → GraphRepairer core + neighborhood test → RED → GREEN → commit
3. **Task 3** → Recall improvement test → RED → GREEN → commit
4. **Task 5** → Disabled test → RED → GREEN → commit (do Task 5 before Task 4 — simpler)
5. **Task 4** → Deletion edge case test → RED → GREEN → commit
6. **Task 6** → ANNSIndex integration (code) → build → test → commit
7. **Task 7** → ANNSIndex integration tests → RED → GREEN → commit
8. **Task 8** → Full suite verification → fix regressions → commit

---

## Success Criteria

Phase 14 is done when ALL of the following are true:

- [ ] `RepairConfiguration` struct exists with `repairInterval`, `repairDepth`, `repairIterations`, `enabled`
- [ ] `GraphRepairer.repair()` implements localized NN-Descent with neighborhood collection
- [ ] `IndexConfiguration` includes `repairConfiguration` with Codable backward compatibility
- [ ] `ANNSIndex.insert()` and `batchInsert()` track pending repair IDs and trigger repair at `repairInterval`
- [ ] `ANNSIndex.repair()` public method exists for manual triggering
- [ ] `ANNSIndex.compact()` and `build()` clear `pendingRepairIDs`
- [ ] Test: `repairImprovesRecall` — repair produces > 0 edge updates on 100 post-build inserts
- [ ] Test: `repairDisabled` — repair returns 0 updates and leaves graph unchanged when `enabled = false`
- [ ] Test: `repairWithDeletions` — repair doesn't crash with edge-case node IDs
- [ ] Test: `indexIntegrationRepair` — full ANNSIndex flow with automatic repair
- [ ] Test: `manualRepair` — explicit `repair()` call works
- [ ] `xcodebuild build` succeeds with zero warnings from MetalANNS code
- [ ] `xcodebuild test` passes ALL existing tests (zero regressions) plus new repair tests
- [ ] Git history has 7-8 clean commits for this phase
- [ ] `tasks/todo.md` has all items checked and completion signal filled in

---

## Anti-Patterns to Avoid

1. **Do NOT run repair on the GPU**. This is a CPU-only feature using `SIMDDistance`. Metal shaders are for full NNDescent builds only.
2. **Do NOT modify `IncrementalBuilder.swift`** or `BatchIncrementalBuilder.swift`. Repair is a post-hoc optimization, not a change to the insert algorithm.
3. **Do NOT collect neighborhoods larger than necessary**. Depth 2 with degree 32 already covers ~1000 nodes. Depth 3 would be ~32,000 — too expensive. Clamp `repairDepth` to max 3.
4. **Do NOT use `async` in `GraphRepairer.repair()`**. It's a synchronous CPU computation. Making it async would force unnecessary actor hopping.
5. **Do NOT hold references to `pendingRepairIDs` across await points**. The actor ensures isolation, but be aware that `triggerRepair()` must be called synchronously within the actor.
6. **Do NOT skip the convergence check**. Without it, repair runs all `repairIterations` even when the graph is already optimal.
7. **Do NOT sort the neighborhood nodes before iteration**. Order doesn't matter for NN-Descent correctness, and sorting adds overhead.
8. **Do NOT forget the `UInt32.max` sentinel check**. Empty neighbor slots contain `UInt32.max` — skip them during neighborhood collection and pair generation.

---

## Commit Messages

Use these exact messages:

1. `feat(repair): add RepairConfiguration struct with sensible defaults and clamping`
2. `feat(repair): implement GraphRepairer with localized NN-Descent`
3. `test(repair): verify repair improves recall after batch inserts`
4. `test(repair): verify repair respects enabled flag`
5. `test(repair): verify repair handles edge cases with deletions`
6. `feat(repair): integrate GraphRepairer into ANNSIndex insert flow`
7. `test(repair): add ANNSIndex integration tests for automatic and manual repair`
8. `chore(repair): verify zero regressions in full test suite`

---

## Performance Budget

Repair cost for typical workloads:

| Insert Count | Neighborhood Size (depth=2, degree=32) | Pairs per iteration | Iterations | Estimated Time |
|-------------|---------------------------------------|--------------------|-----------|----|
| 100 | ~800 nodes | ~320,000 | 5 | ~25ms |
| 50 | ~500 nodes | ~125,000 | 5 | ~10ms |
| 10 | ~150 nodes | ~11,250 | 5 | ~1ms |

Amortized per insert: ~0.25ms (100 inserts at 25ms / 100). This is acceptable for online workloads where insert latency budget is typically 1-10ms.

If repair takes longer than 100ms in any test, the neighborhood is too large. Check that `repairDepth` clamping works and that degenerate graphs (all nodes connected to all others) don't blow up the neighborhood size.
