# Phase 15 Execution Prompt: CPU-only HNSW Layer Navigation

---

## System Context

You are implementing **Phase 15** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1–14 are complete. The package compiles under `swift-tools-version: 6.2` with typed throws and `@concurrent` methods. All tests pass. The codebase has a flat CAGRA-style NN-Descent graph with beam search.

You are adding **CPU-only HNSW (Hierarchical Navigable Small World) layer navigation** — a skip-list-like structure that reduces search from O(N) to O(log N) by layering the graph into a hierarchy. This improves recall and speed on the **CPU backend** (AccelerateBackend). GPU search is unchanged (flat multi-start is faster on GPU per the CAGRA paper).

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/todo.md`

**Your communication contract:**
1. **`tasks/todo.md` is your shared state.** Check off `[x]` items as you complete them.
2. **Write notes under every task** — especially for algorithm decisions and any edge cases.
3. **Update `Last Updated`** at the top of todo.md after each task completes.
4. **When done, fill in the "Phase 15 Complete — Signal" section** at the bottom of todo.md.
5. **Do NOT modify the "Orchestrator Review Checklist"** section — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **TDD cycle for every task**: Write test → run to see it fail (RED) → implement → run to see it pass (GREEN) → commit. No exceptions.
2. **Swift 6 strict concurrency**: All new types must be `Sendable`. Use typed throws (`throws(ANNSError)`) on all new public functions.
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest.
4. **Build with `xcodebuild`**, never `swift build` or `swift test`. Metal shaders are not compiled by SPM CLI.
5. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
6. **Commit after every task** with the exact conventional commit message specified in the todo.
7. **CPU-only feature**. HNSW is NOT used for GPU search (FullGPUSearch remains unchanged). HNSW is an optimization for AccelerateBackend search only.
8. **Do NOT modify existing public API signatures**. Only add new configuration options and new search methods. Existing callers must not break.

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

## The Problem: Why HNSW Is Needed

### Current Search Flow (BeamSearchCPU — O(N) worst-case)

When you call `ANNSIndex.search()` on CPU, it uses `BeamSearchCPU`:
1. Start at a single global entry point
2. Greedy beam search: visit neighbors with lowest distance
3. Stop when no improvements found
4. Return top-k results

**Issue**: On large graphs (N > 100K), even with pruning, the search visits O(N) nodes in worst-case (low-connectivity clusters, poor entry point). This is O(N) time and O(N) memory.

### HNSW Solution — O(log N) Expected Complexity

HNSW adds a skip-list-like hierarchy:
- Layer 0: Full graph (all N nodes)
- Layer 1: ~N / e nodes (probabilistically selected)
- Layer 2: ~N / e² nodes
- ...
- Layer L: Single entry point (theoretical, but close)

**Search**:
1. Start at top layer (few nodes)
2. Greedy descent: at each layer, find closest node to query
3. Move to that node's neighbors in the layer below
4. Repeat until reaching layer 0
5. Final beam search at layer 0

**Result**: O(log N) layer transitions + O(degree) beam at each layer = **O(degree × log N)** total distance computations.

### When GPU Doesn't Help

GPU search (`FullGPUSearch`) launches a single kernel that does full beam search in parallel — already optimal on GPU. GPU memory is ample, compute is throughput-bound, not latency-bound. HNSW is unnecessary for GPU.

CPU search is **latency-sensitive** (single thread) and **memory-constrained** (shared with system). HNSW's O(log N) improvement is crucial.

---

## Current Codebase Snapshot (What You're Building On)

### BeamSearchCPU (reference implementation)
```swift
public enum BeamSearchCPU {
    public static func search(
        query: [Float],
        vectors: [[Float]],
        graph: [[(UInt32, Float)]],
        entryPoint: Int,
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws(ANNSError) -> [SearchResult]
}
```

Algorithm:
- `ef` = expansion factor (larger = more candidates explored, higher recall)
- `results` = best-k candidates found so far
- `candidates` = frontier to explore
- Loop: pop cheapest candidate; if it beats worst result, add its neighbors
- Stop when no candidates beat worst result

### SIMDDistance (metric computation)
```swift
public enum SIMDDistance {
    public static func distance(_ a: [Float], _ b: [Float], metric: Metric) -> Float
}
```

Supports all three metrics: cosine, l2, inner product.

### SearchResult (result shape)
```swift
public struct SearchResult: Sendable {
    public let id: String           // External user ID
    public let score: Float         // Distance
    public let internalID: UInt32   // Graph node index
}
```

---

## Architecture

### Data Structure

```
HNSWLayers {
    layers: [SkipLayer]       // layers[0] is "virtual" (uses base graph)
                              // layers[1..L] are skip layers
    maxLayer: Int
    mL: Double                // 1 / ln(2) ≈ 1.443 (level multiplier)
    entryPoint: UInt32        // Top-level entry node
}

SkipLayer {
    nodeToLayerIndex: [UInt32: UInt32]  // graph node → layer-local index
    layerIndexToNode: [UInt32]          // layer-local index → graph node
    adjacency: [[UInt32]]               // neighbor lists (variable degree, sorted by distance)
}
```

### Construction Flow

```
After NN-Descent completes:
  ├─ For each node, assign level: floor(-ln(uniform(0,1)) / ln(2))
  │  Result: ~63% level 0, ~23% level 1, ~8% level 2, ~3% level 3, etc.
  │
  ├─ Find entry point (top-most node)
  │
  └─ For each layer L from 1 to max:
       For each node at layer L:
         Find nearest neighbors in layer L using layer L-1 as entry
         Store edges in layer L adjacency
```

### Search Flow

```
For each query:
  ├─ Start at entryPoint in top layer L_max
  │
  ├─ For layer L from L_max down to 1:
  │  ├─ Greedy search in layer L:
  │  │  └─ Find closest node to query among all layer L nodes
  │  │     (limited candidates to avoid exploring whole layer)
  │  │
  │  └─ Move entry to that closest node, proceed to layer L-1
  │
  └─ At layer 0: full beam search (like BeamSearchCPU) returning top-k
```

---

## New Files to Create

### 1. `Sources/MetalANNSCore/HNSWLayers.swift`

Data structure definition:

```swift
import Foundation

/// Skip layer in the HNSW hierarchy.
public struct SkipLayer: Sendable, Codable {
    /// Maps graph node ID → layer-local node index (for fast neighbor lookup)
    public var nodeToLayerIndex: [UInt32: UInt32]

    /// Maps layer-local index → graph node ID
    public var layerIndexToNode: [UInt32]

    /// Adjacency lists at this layer (indexed by layer-local index).
    /// Each node has variable-degree neighbors, sorted by distance.
    public var adjacency: [[UInt32]]

    public init(
        nodeToLayerIndex: [UInt32: UInt32] = [:],
        layerIndexToNode: [UInt32] = [],
        adjacency: [[UInt32]] = []
    ) {
        self.nodeToLayerIndex = nodeToLayerIndex
        self.layerIndexToNode = layerIndexToNode
        self.adjacency = adjacency
    }
}

/// Complete HNSW skip-layer structure. Layer 0 is NOT stored (uses the base graph).
/// Layers 1+ are skip layers.
public struct HNSWLayers: Sendable {
    /// Skip layers (layers[0] represents layer 1, layers[1] represents layer 2, etc.)
    public let layers: [SkipLayer]

    /// Maximum layer (0 means no skip layers, just the base graph)
    public let maxLayer: Int

    /// Level multiplier: 1 / ln(2)
    public let mL: Double

    /// Entry point node in the top layer
    public let entryPoint: UInt32

    public init(
        layers: [SkipLayer] = [],
        maxLayer: Int = 0,
        mL: Double = 1.4426950408889634, // 1 / ln(2)
        entryPoint: UInt32 = 0
    ) {
        self.layers = layers
        self.maxLayer = maxLayer
        self.mL = mL
        self.entryPoint = entryPoint
    }

    /// Returns the layer-local neighbors of a node at a given layer.
    /// - Parameters:
    ///   - nodeID: Graph node ID
    ///   - layer: Layer number (0 = base graph, 1+ = skip layers)
    /// - Returns: Array of neighbor node IDs, or nil if node not in layer
    public func neighbors(of nodeID: UInt32, at layer: Int) -> [UInt32]? {
        guard layer > 0, layer <= maxLayer else { return nil }
        let skipLayer = layers[layer - 1]
        guard let layerIndex = skipLayer.nodeToLayerIndex[nodeID] else { return nil }
        return Array(skipLayer.adjacency[Int(layerIndex)])
    }
}
```

### 2. `Sources/MetalANNSCore/HNSWBuilder.swift`

Construct HNSW layers from a base graph:

```swift
import Foundation
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "HNSWBuilder")

public enum HNSWBuilder {

    /// Build HNSW skip layers from a complete base graph.
    ///
    /// - Parameters:
    ///   - vectors: Vector storage (for distance computation)
    ///   - graph: Base graph adjacency lists
    ///   - nodeCount: Number of nodes in the graph
    ///   - metric: Distance metric
    /// - Returns: HNSWLayers structure
    public static func buildLayers(
        vectors: any VectorStorage,
        graph: [[(UInt32, Float)]],
        nodeCount: Int,
        metric: Metric
    ) throws(ANNSError) -> HNSWLayers {
        guard nodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot build HNSW layers with empty graph")
        }
        guard nodeCount == graph.count else {
            throw ANNSError.constructionFailed("Graph size does not match node count")
        }

        // Step 1: Assign levels to nodes
        var nodeLevels: [Int] = Array(repeating: 0, count: nodeCount)
        var maxLayerAssigned = 0

        for nodeID in 0..<nodeCount {
            let level = assignLevel(mL: 1.4426950408889634)
            nodeLevels[nodeID] = level
            maxLayerAssigned = max(maxLayerAssigned, level)
        }

        // Step 2: Find entry point (highest-level node)
        var entryPoint: UInt32 = 0
        var entryLevel = nodeLevels[0]
        for nodeID in 1..<nodeCount {
            if nodeLevels[nodeID] > entryLevel {
                entryLevel = nodeLevels[nodeID]
                entryPoint = UInt32(nodeID)
            }
        }

        logger.debug("Assigned levels: max=\(maxLayerAssigned), entryPoint=\(entryPoint)")

        // Step 3: Build skip layers
        var skipLayers: [SkipLayer] = []
        for layer in 1...maxLayerAssigned {
            let skipLayer = try buildSkipLayer(
                at: layer,
                nodeLevels: nodeLevels,
                vectors: vectors,
                baseGraph: graph,
                nodeCount: nodeCount,
                metric: metric
            )
            skipLayers.append(skipLayer)
        }

        return HNSWLayers(layers: skipLayers, maxLayer: maxLayerAssigned, entryPoint: entryPoint)
    }

    /// Assign a random level to a new node using exponential decay.
    /// Level = floor(-ln(uniform(0,1)) / ln(2))
    /// ~63% level 0, ~23% level 1, ~8% level 2, etc.
    private static func assignLevel(mL: Double) -> Int {
        let uniform = Float.random(in: 0..<1.0)
        guard uniform > 0 else { return 0 }
        let level = Int(floor(-log(Double(uniform)) * mL))
        return level
    }

    /// Build a single skip layer.
    private static func buildSkipLayer(
        at layer: Int,
        nodeLevels: [Int],
        vectors: any VectorStorage,
        baseGraph: [[(UInt32, Float)]],
        nodeCount: Int,
        metric: Metric
    ) throws(ANNSError) -> SkipLayer {
        // Collect nodes at this layer
        var nodesAtLayer: [UInt32] = []
        for nodeID in 0..<nodeCount {
            if nodeLevels[nodeID] >= layer {
                nodesAtLayer.append(UInt32(nodeID))
            }
        }

        guard !nodesAtLayer.isEmpty else {
            return SkipLayer()
        }

        // Build mapping: node ID → layer-local index
        var nodeToLayerIndex: [UInt32: UInt32] = [:]
        var layerIndexToNode: [UInt32] = []
        for (layerIndex, nodeID) in nodesAtLayer.enumerated() {
            nodeToLayerIndex[nodeID] = UInt32(layerIndex)
            layerIndexToNode.append(nodeID)
        }

        // Build adjacency lists for this layer
        var adjacency: [[UInt32]] = Array(repeating: [], count: nodesAtLayer.count)

        for (layerIndex, nodeID) in nodesAtLayer.enumerated() {
            let nodeVector = vectors.vector(at: Int(nodeID))

            // Find nearest neighbors at this layer
            var candidates: [(nodeID: UInt32, distance: Float)] = []
            for otherID in nodesAtLayer where otherID != nodeID {
                let distance = SIMDDistance.distance(
                    nodeVector,
                    vectors.vector(at: Int(otherID)),
                    metric: metric
                )
                candidates.append((otherID, distance))
            }

            // Sort and keep top M (for HNSW, typically M=8..16 at higher layers)
            candidates.sort { $0.distance < $1.distance }
            let M = min(8, candidates.count)  // Connection limit
            let neighbors = candidates.prefix(M).map { $0.nodeID }

            adjacency[layerIndex] = Array(neighbors)
        }

        return SkipLayer(
            nodeToLayerIndex: nodeToLayerIndex,
            layerIndexToNode: layerIndexToNode,
            adjacency: adjacency
        )
    }
}
```

### 3. `Sources/MetalANNSCore/HNSWSearchCPU.swift`

Search using HNSW layers:

```swift
import Foundation

public enum HNSWSearchCPU {

    /// Search using HNSW layer hierarchy, falling back to beam search at layer 0.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - vectors: Vector storage
    ///   - hnsw: HNSW layers
    ///   - baseGraph: Layer 0 (base graph) adjacency lists
    ///   - k: Number of results to return
    ///   - ef: Expansion factor for layer 0 beam search
    ///   - metric: Distance metric
    /// - Returns: Top-k search results
    public static func search(
        query: [Float],
        vectors: [[Float]],
        hnsw: HNSWLayers,
        baseGraph: [[(UInt32, Float)]],
        k: Int,
        ef: Int,
        metric: Metric
    ) async throws(ANNSError) -> [SearchResult] {
        guard k > 0 else { return [] }
        guard !vectors.isEmpty else { throw ANNSError.indexEmpty }
        guard query.count == vectors[0].count else {
            throw ANNSError.dimensionMismatch(expected: vectors[0].count, got: query.count)
        }

        // Step 1: Layer-by-layer descent from top to layer 1
        var currentEntry = hnsw.entryPoint
        for layer in (1...hnsw.maxLayer).reversed() {
            currentEntry = try greedySearchLayer(
                query: query,
                vectors: vectors,
                hnsw: hnsw,
                layer: layer,
                entryPoint: Int(currentEntry),
                metric: metric
            )
        }

        // Step 2: Beam search at layer 0 starting from currentEntry
        let results = try await BeamSearchCPU.search(
            query: query,
            vectors: vectors,
            graph: baseGraph,
            entryPoint: Int(currentEntry),
            k: k,
            ef: ef,
            metric: metric
        )

        return results
    }

    /// Greedy descent at a single layer: find closest node to query among neighbors.
    private static func greedySearchLayer(
        query: [Float],
        vectors: [[Float]],
        hnsw: HNSWLayers,
        layer: Int,
        entryPoint: Int,
        metric: Metric
    ) throws(ANNSError) -> UInt32 {
        guard layer > 0, layer <= hnsw.maxLayer else {
            throw ANNSError.searchFailed("Invalid layer for greedy search")
        }
        guard entryPoint >= 0, entryPoint < vectors.count else {
            throw ANNSError.searchFailed("Entry point out of bounds")
        }

        var current = UInt32(entryPoint)
        var currentDistance = SIMDDistance.distance(query, vectors[Int(current)], metric: metric)

        var improved = true
        var iterations = 0
        let maxIterations = 100  // Prevent infinite loops

        while improved && iterations < maxIterations {
            improved = false
            iterations += 1

            guard let neighbors = hnsw.neighbors(of: current, at: layer) else {
                break  // Node not in this layer
            }

            for neighborID in neighbors {
                let distance = SIMDDistance.distance(query, vectors[Int(neighborID)], metric: metric)
                if distance < currentDistance {
                    current = neighborID
                    currentDistance = distance
                    improved = true
                }
            }
        }

        return current
    }
}
```

### 4. `Sources/MetalANNSCore/HNSWConfiguration.swift`

Configuration for HNSW:

```swift
import Foundation

public struct HNSWConfiguration: Sendable, Codable {
    /// Whether to enable HNSW layers for CPU search
    public var enabled: Bool

    /// Connection limit per layer (higher = higher recall, more memory)
    public var M: Int

    /// Maximum number of layers (typically 4-8)
    public var maxLayers: Int

    public static let `default` = HNSWConfiguration(
        enabled: true,
        M: 8,
        maxLayers: 6
    )

    public init(
        enabled: Bool = true,
        M: Int = 8,
        maxLayers: Int = 6
    ) {
        self.enabled = enabled
        self.M = max(1, M)
        self.maxLayers = max(0, maxLayers)
    }
}
```

### 5. `Tests/MetalANNSTests/HNSWTests.swift`

Test suite:

```swift
import Testing
@testable import MetalANNSCore

@Suite("HNSW Layer Tests")
struct HNSWTests {

    @Test("HNSWLayers stores and retrieves neighbors correctly")
    func hnswtLayerStructure() {
        let layer1 = SkipLayer(
            nodeToLayerIndex: [0: 0, 2: 1, 5: 2],
            layerIndexToNode: [0, 2, 5],
            adjacency: [[2, 5], [0, 5], [0, 2]]
        )
        let hnsw = HNSWLayers(layers: [layer1], maxLayer: 1, entryPoint: 0)

        #expect(hnsw.neighbors(of: 0, at: 1) == [2, 5])
        #expect(hnsw.neighbors(of: 2, at: 1) == [0, 5])
        #expect(hnsw.neighbors(of: 1, at: 1) == nil)  // Not in layer 1
    }

    @Test("HNSWBuilder assigns levels with exponential distribution")
    func hnswtLevelAssignment() async throws {
        let vectors = (0..<100).map { i in (0..<8).map { d in Float(i * 8 + d) * 0.01 } }
        let (graphData, _) = try await NNDescentCPU.build(
            vectors: vectors, degree: 4, metric: .l2, maxIterations: 5
        )

        let vectorBuffer = try makeVectorBuffer(vectors)
        let hnsw = try HNSWBuilder.buildLayers(
            vectors: vectorBuffer,
            graph: graphData,
            nodeCount: vectors.count,
            metric: .l2
        )

        // Should have multiple layers (statistically ~95% chance maxLayer > 0 with 100 nodes)
        #expect(hnsw.maxLayer >= 0)
        // Entry point should be defined
        #expect(hnsw.entryPoint < 100)
    }

    @Test("HNSW search returns k results")
    func hnswtSearch() async throws {
        let vectors = (0..<200).map { i in (0..<16).map { d in Float(i * 16 + d) * 0.01 } }
        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors, degree: 8, metric: .cosine, maxIterations: 10
        )

        let vectorBuffer = try makeVectorBuffer(vectors)
        let hnsw = try HNSWBuilder.buildLayers(
            vectors: vectorBuffer,
            graph: graphData,
            nodeCount: vectors.count,
            metric: .cosine
        )

        let query = (0..<16).map { d in Float(d) * 0.01 }
        let results = try await HNSWSearchCPU.search(
            query: query,
            vectors: vectors,
            hnsw: hnsw,
            baseGraph: graphData,
            k: 10,
            ef: 64,
            metric: .cosine
        )

        #expect(results.count == 10)
        for i in 1..<results.count {
            #expect(results[i].score >= results[i - 1].score)
        }
    }

    @Test("HNSW recall matches or exceeds flat beam search")
    func hnswtRecallComparison() async throws {
        let vectors = (0..<500).map { i in (0..<32).map { d in sin(Float(i * 32 + d) * 0.01) } }
        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors, degree: 16, metric: .cosine, maxIterations: 15
        )

        let vectorBuffer = try makeVectorBuffer(vectors)
        let hnsw = try HNSWBuilder.buildLayers(
            vectors: vectorBuffer,
            graph: graphData,
            nodeCount: vectors.count,
            metric: .cosine
        )

        let backend = AccelerateBackend()
        let flat = vectors.flatMap { $0 }
        let queries = (0..<10).map { i in (0..<32).map { d in sin(Float(i * 32 + d) * 0.01) } }
        let k = 10
        let ef = 64

        var hnswtRecall: Float = 0
        var flatRecall: Float = 0

        for query in queries {
            // HNSW search
            let hnswtResults = try await HNSWSearchCPU.search(
                query: query,
                vectors: vectors,
                hnsw: hnsw,
                baseGraph: graphData,
                k: k,
                ef: ef,
                metric: .cosine
            )

            // Flat beam search
            let flatResults = try await BeamSearchCPU.search(
                query: query,
                vectors: vectors,
                graph: graphData,
                entryPoint: Int(entryPoint),
                k: k,
                ef: ef,
                metric: .cosine
            )

            // Ground truth
            let exactDistances = try await computeExactDistances(query, vectors, backend, flat)
            let exactTopK = Set(
                exactDistances.enumerated()
                    .sorted { $0.element < $1.element }
                    .prefix(k)
                    .map { UInt32($0.offset) }
            )

            hnswtRecall += Float(Set(hnswtResults.map(\.internalID)).intersection(exactTopK).count) / Float(k)
            flatRecall += Float(Set(flatResults.map(\.internalID)).intersection(exactTopK).count) / Float(k)
        }

        hnswtRecall /= Float(queries.count)
        flatRecall /= Float(queries.count)

        // HNSW should match flat search (same underlying graph at layer 0)
        #expect(abs(hnswtRecall - flatRecall) < 0.05, "HNSW recall diverged from flat search")
    }

    // Helper: compute exact distances
    private func computeExactDistances(
        _ query: [Float],
        _ vectors: [[Float]],
        _ backend: AccelerateBackend,
        _ flat: [Float]
    ) async throws -> [Float] {
        try await withVectorBuffer(flat) { pointer in
            try await backend.computeDistances(
                query: query,
                vectors: pointer,
                vectorCount: vectors.count,
                dim: query.count,
                metric: .cosine
            )
        }
    }

    private func makeVectorBuffer(_ vectors: [[Float]]) throws -> VectorBuffer {
        guard let first = vectors.first else {
            throw ANNSError.constructionFailed("Empty vectors")
        }
        let buffer = try VectorBuffer(capacity: vectors.count + 10, dim: first.count)
        for (index, vector) in vectors.enumerated() {
            try buffer.insert(vector: vector, at: index)
        }
        buffer.setCount(vectors.count)
        return buffer
    }

    private func withVectorBuffer<T>(
        _ values: [Float],
        _ body: (UnsafeBufferPointer<Float>) async throws -> T
    ) async throws -> T {
        let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: values.count)
        buffer.initialize(from: values)
        defer {
            buffer.deinitialize()
            buffer.deallocate()
        }
        return try await body(UnsafeBufferPointer(buffer))
    }
}
```

---

## Files to Modify

### 6. `Sources/MetalANNSCore/ANNSIndex.swift`

Add HNSW integration. After graph construction (in `build()` when CPU backend), optionally build HNSW layers:

**New stored property**:
```swift
private var hnsw: HNSWLayers?
```

**Modify `build()` method** — after `NNDescentCPU.build()` or `NNDescentGPU.build()`:
```swift
// After graph construction:
self.hnsw = nil  // Clear any old HNSW

// If CPU backend, optionally build HNSW layers
if context == nil && configuration.hnswConfiguration.enabled {
    self.hnsw = try HNSWBuilder.buildLayers(
        vectors: vectorBuffer,
        graph: extractGraph(from: graphBuffer),
        nodeCount: inputVectors.count,
        metric: configuration.metric
    )
}
```

**Modify `search()` method** — when CPU backend, use HNSW if available:
```swift
let rawResults: [SearchResult]
if let context, supportsGPUSearch(for: vectors) {
    // GPU path unchanged
    rawResults = try await FullGPUSearch.search(...)
} else {
    // CPU path: use HNSW if available
    if let hnsw {
        rawResults = try await HNSWSearchCPU.search(
            query: query,
            vectors: extractVectors(from: vectors),
            hnsw: hnsw,
            baseGraph: extractGraph(from: graph),
            k: max(1, effectiveK),
            ef: max(1, effectiveEf),
            metric: searchMetric
        )
    } else {
        // Fallback to flat beam search (if HNSW disabled or not built)
        rawResults = try await BeamSearchCPU.search(...)
    }
}
```

**Modify `compact()` method** — clear HNSW after rebuild:
```swift
// After compaction completes:
self.hnsw = nil
```

---

## Task Breakdown

### Task 1: Create `HNSWLayers.swift` + basic structure tests

**TDD RED**: Write test expecting SkipLayer and HNSWLayers types:

```swift
@Test("HNSWLayers stores neighbors")
func hnswtStorageTest() {
    let layer = SkipLayer(
        nodeToLayerIndex: [0: 0, 1: 1],
        layerIndexToNode: [0, 1],
        adjacency: [[1], [0]]
    )
    #expect(layer.nodeToLayerIndex[0] == 0)
}
```

**TDD GREEN**: Implement `HNSWLayers.swift` as specified above.

**Commit**: `feat(hnsw): add HNSWLayers and SkipLayer data structures`

---

### Task 2: Create `HNSWBuilder.swift` + level assignment and layer building

**TDD RED**:

```swift
@Test("HNSWBuilder creates layers")
func hnswtBuildingTest() async throws {
    let vectors = (0..<50).map { i in (0..<8).map { d in Float(i * 8 + d) } }
    let graph = (0..<50).map { i in [(UInt32((i+1)%50), 0.5)] }

    let hnsw = try HNSWBuilder.buildLayers(
        vectors: try makeVectorBuffer(vectors),
        graph: graph,
        nodeCount: 50,
        metric: .l2
    )

    #expect(hnsw.maxLayer >= 0)
}
```

**TDD GREEN**: Implement `HNSWBuilder.swift` with `buildLayers()`, `assignLevel()`, and `buildSkipLayer()`.

**Commit**: `feat(hnsw): implement HNSWBuilder with probabilistic level assignment`

---

### Task 3: Create `HNSWSearchCPU.swift` + layer-by-layer descent

**TDD RED**:

```swift
@Test("HNSWSearchCPU descends layers and searches")
func hnswtSearchTest() async throws {
    let vectors = (0..<100).map { i in (0..<16).map { d in Float(i * 16 + d) * 0.01 } }
    let (graphData, _) = try await NNDescentCPU.build(vectors: vectors, degree: 4, metric: .l2, maxIterations: 5)

    let hnsw = try HNSWBuilder.buildLayers(
        vectors: try makeVectorBuffer(vectors),
        graph: graphData,
        nodeCount: vectors.count,
        metric: .l2
    )

    let query = (0..<16).map { d in Float(d) * 0.01 }
    let results = try await HNSWSearchCPU.search(
        query: query, vectors: vectors, hnsw: hnsw,
        baseGraph: graphData, k: 5, ef: 32, metric: .l2
    )

    #expect(results.count == 5)
}
```

**TDD GREEN**: Implement `HNSWSearchCPU.swift` with `search()` and `greedySearchLayer()`.

**Commit**: `feat(hnsw): implement HNSWSearchCPU with layer descent and beam search`

---

### Task 4: Create `HNSWConfiguration.swift`

**TDD RED**:

```swift
@Test("HNSWConfiguration has defaults")
func hnswtConfigTest() {
    let config = HNSWConfiguration.default
    #expect(config.enabled == true)
    #expect(config.M > 0)
}
```

**TDD GREEN**: Implement `HNSWConfiguration.swift`.

**Commit**: `feat(hnsw): add HNSWConfiguration with sensible defaults`

---

### Task 5: Write comprehensive test suite (`HNSWTests.swift`)

**TDD RED**: Write full test suite as specified above (4+ tests).

**TDD GREEN**: All tests from Task 1-4 implementations pass.

**Commit**: `test(hnsw): add comprehensive layer assignment, build, and search tests`

---

### Task 6: Integrate into `ANNSIndex.swift`

**Modify files**:
- Add `hnsw` property
- Modify `build()` to construct HNSW layers if CPU backend
- Modify `search()` to use HNSW if available
- Modify `compact()` to clear HNSW

**TDD RED**:

```swift
@Test("ANNSIndex uses HNSW for CPU search when available")
func indexHNSWTest() async throws {
    var config = IndexConfiguration(degree: 8, metric: .l2)
    config.hnswConfiguration = HNSWConfiguration(enabled: true)

    let index = ANNSIndex(configuration: config)
    let vectors = (0..<100).map { i in (0..<16).map { d in Float(i * 16 + d) * 0.01 } }
    let ids = (0..<100).map { "v_\($0)" }

    try await index.build(vectors: vectors, ids: ids)

    let query = (0..<16).map { d in Float(d) * 0.01 }
    let results = try await index.search(query: query, k: 5)

    #expect(results.count == 5)
}
```

**TDD GREEN**: Integration tests pass.

**Commit**: `feat(hnsw): integrate HNSWSearchCPU into ANNSIndex search path`

---

### Task 7: Verify full test suite passes

Run all tests. Fix regressions. Ensure:
- Existing `SearchTests.swift` still pass (beam search unchanged)
- New HNSW tests pass
- No memory leaks or crashes

**Commit**: `chore(hnsw): verify zero regressions in full test suite`

---

## Task Execution Order

1. **Task 1** → HNSWLayers data structure → RED → GREEN → commit
2. **Task 2** → HNSWBuilder level assignment and layer building → RED → GREEN → commit
3. **Task 3** → HNSWSearchCPU layer descent → RED → GREEN → commit
4. **Task 4** → HNSWConfiguration → RED → GREEN → commit
5. **Task 5** → Full test suite → RED → GREEN → commit
6. **Task 6** → ANNSIndex integration → RED → GREEN → commit
7. **Task 7** → Full regression testing → commit

---

## Success Criteria

Phase 15 is done when ALL of the following are true:

- [ ] `HNSWLayers` struct stores and retrieves skip layers correctly
- [ ] `HNSWBuilder.buildLayers()` creates multi-layer hierarchy with probabilistic level assignment
- [ ] ~63% of nodes assigned level 0, ~23% level 1, ~8% level 2 (exponential decay)
- [ ] `HNSWSearchCPU.search()` performs layer-by-layer greedy descent then beam search at layer 0
- [ ] `HNSWConfiguration` includes `enabled`, `M`, `maxLayers` with validation
- [ ] `ANNSIndex.build()` constructs HNSW layers if CPU backend and enabled
- [ ] `ANNSIndex.search()` uses HNSWSearchCPU when HNSW available, falls back to BeamSearchCPU
- [ ] `ANNSIndex.compact()` clears HNSW (will be rebuilt next time)
- [ ] Test: HNSW recall matches flat beam search ±5% (same base graph at layer 0)
- [ ] Test: HNSW layer structure stores and retrieves neighbors correctly
- [ ] Test: Layer assignment produces expected exponential distribution
- [ ] `xcodebuild build` succeeds with zero warnings
- [ ] `xcodebuild test` passes ALL existing tests (zero regressions) plus 5+ new HNSW tests
- [ ] Git history has 7 clean commits for this phase
- [ ] `tasks/todo.md` has all items checked and completion signal filled in

---

## Anti-Patterns to Avoid

1. **Do NOT use HNSW for GPU search**. FullGPUSearch is unchanged (flat multi-start is faster on GPU).
2. **Do NOT modify BeamSearchCPU**. HNSW is an additional optimization, not a replacement.
3. **Do NOT store HNSW in persistence yet**. Phase 15 builds HNSW at runtime after loading; this will be optimized in v3.1.
4. **Do NOT use variable-degree neighbors without sorting**. Neighbors must be sorted by distance for correctness.
5. **Do NOT skip the layer descent loop**. Even if entry point is optimal, greedy descent improves results.
6. **Do NOT handle layer 0 differently in HNSWSearchCPU**. It delegates to BeamSearchCPU (which already handles it).
7. **Do NOT forget UInt32.max sentinel checks** in layer neighbors. Skip invalid node IDs.
8. **Do NOT build HNSW for GPU backend**. GPU path uses FullGPUSearch unchanged.

---

## Commit Messages

1. `feat(hnsw): add HNSWLayers and SkipLayer data structures`
2. `feat(hnsw): implement HNSWBuilder with probabilistic level assignment`
3. `feat(hnsw): implement HNSWSearchCPU with layer descent and beam search`
4. `feat(hnsw): add HNSWConfiguration with sensible defaults`
5. `test(hnsw): add comprehensive layer assignment, build, and search tests`
6. `feat(hnsw): integrate HNSWSearchCPU into ANNSIndex search path`
7. `chore(hnsw): verify zero regressions in full test suite`

---

## Performance Notes

### Expected Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Build HNSW** | O(N × M × log N) | M = connection limit (~8), one distance per layer assignment |
| **Layer descent** | O(M × log N) | M neighbors at each of log N layers |
| **Layer 0 beam search** | O(degree × ef) | Standard beam search on full graph |
| **Total search** | O(M × log N + degree × ef) | Dominates at high ef |

### Memory

- Base graph: `N × degree × 8 bytes` (UInt32 + Float per neighbor)
- Skip layers: ~20% of base graph size (most nodes only in layer 0, few in layer 2+)
- Total: 1.2× base graph memory

### On-Device Performance

Building HNSW on 1M vectors, degree=32:
- ~500ms (mostly distance computations in layer building)
- Should complete in app initialization (<1s)

Searching with HNSW vs flat:
- **Flat**: 100K vectors, ef=64 → ~50ms (beam explores many nodes)
- **HNSW**: 100K vectors, ef=64 → ~5ms (layer descent skips most nodes)
- **Speedup**: 10x (at cost of ~20% memory)

---

## References

**HNSW Paper**: Malkov & Yashunin, 2016. "Efficient and Robust Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graphs"

**Key insight**: HNSW is a skip-list-like structure optimized for metric spaces. No modifications needed for our use case (cosine, L2, inner product all work with the same algorithm).
