# Phase 20: Quantized HNSW

> **For Claude:** This is an **implementation prompt** for Phase 20 of MetalANNS v4.
> Execute via TDD (RED→GREEN→commit). Dispatch to subagent; orchestrator reviews using
> the R1-R13 checklist in `tasks/todo.md`.

**Goal:** Replace full-precision distance calls in HNSW skip-layer greedy search with
Asymmetric Distance Computation (ADC) using Product Quantization. Skip layers navigate
to the right neighbourhood in O(log N) hops — each hop currently costs a full 128-float
dot product. With ADC, each hop costs M table lookups (M = pqSubspaces = 4). For N > 50K
vectors this delivers measurable latency reduction with recall degradation < 3%.

---

## Current Bottleneck (Verified from Codebase)

### HNSWSearchCPU.swift — lines 76 and 93-97

```swift
// Initial entry-point distance — full precision:
var currentDistance = SIMDDistance.distance(query, vectors[entryPoint], metric: metric)

// Per-hop distance in skip layer — full precision for every neighbor:
let neighborDistance = SIMDDistance.distance(
    query,
    vectors[Int(neighborID)],
    metric: metric
)                                                // ← replace this with ADC table lookup
```

Skip layer L1 has ≈23% of total nodes, L2 ≈8.6%. Every greedy hop in these layers pays
the full O(dim) distance. For `dim=128` with `Float` that is 128 FP32 multiplications.

### ProductQuantizer.swift — available but unused in search (lines 115-128)

```swift
// Already exists in the codebase — Phase 16:
public func approximateDistance(query: [Float], codes: [UInt8], metric: Metric) -> Float
// And:
func distanceTable(query: [Float], metric: Metric) -> [[Float]]?  // internal to MetalANNSCore
```

The `distanceTable` precomputes M×256 floats once per query. Each subsequent ADC lookup
is just M array reads — O(M) instead of O(dim).

### HNSWLayers.swift — adjacency uses global node IDs

```swift
// SkipLayer.adjacency[layerLocalIndex] = [UInt32]  ← graph-global node IDs (NOT layer-local)
public var adjacency: [[UInt32]]       // line 12

// layerIndexToNode maps layer-local → global:
public var layerIndexToNode: [UInt32]  // line 9
// nodeToLayerIndex maps global → layer-local:
public var nodeToLayerIndex: [UInt32: UInt32]  // line 7
```

A `QuantizedSkipLayer` wraps a `SkipLayer` and adds PQ codes indexed by **layer-local index**:
`codes[layerLocalIndex]` = PQ code for `layerIndexToNode[layerLocalIndex]`'s full vector.

---

## Architecture: Per-Layer PQ Codebook

```
HNSWLayers (existing)
  layers[0] = SkipLayer for layer 1   ← ≈23% of nodes
  layers[1] = SkipLayer for layer 2   ← ≈8.6% of nodes
  ...

QuantizedHNSWLayers (new)
  quantizedLayers[0] = QuantizedSkipLayer
    .base        = SkipLayer (existing data, unchanged)
    .pq          = ProductQuantizer trained on this layer's node vectors
    .codes       = [[UInt8]] indexed by layer-local index

Search path (QuantizedHNSWSearchCPU):
  entry → layer L (highest)
    distanceTable = pq_L.distanceTable(query, metric)    ← once per layer
    greedy hop: approximateDistance via table lookup     ← O(M) per hop
  → layer 0 (base graph)
    BeamSearchCPU.search() unchanged — full precision
```

---

## System Context

### Key types (verified in codebase)

```swift
// SkipLayer (Sources/MetalANNSCore/HNSWLayers.swift) — do NOT modify
public struct SkipLayer: Sendable, Codable {
    public var nodeToLayerIndex: [UInt32: UInt32]
    public var layerIndexToNode: [UInt32]
    public var adjacency: [[UInt32]]          // indexed by layer-local index
}

// HNSWLayers (Sources/MetalANNSCore/HNSWLayers.swift) — do NOT modify
public struct HNSWLayers: Sendable {
    public let layers: [SkipLayer]            // layers[0] = layer 1
    public let maxLayer: Int
    public let mL: Double
    public let entryPoint: UInt32
    public func neighbors(of nodeID: UInt32, at layer: Int) -> [UInt32]?
}

// ProductQuantizer (Sources/MetalANNSCore/ProductQuantizer.swift) — do NOT modify
public struct ProductQuantizer: Sendable, Codable {
    public let numSubspaces: Int
    public let centroidsPerSubspace: Int      // always 256
    public let subspaceDimension: Int
    public let codebooks: [[[Float]]]

    public static func train(vectors: [[Float]], numSubspaces: Int = 8,
                             centroidsPerSubspace: Int = 256,
                             maxIterations: Int = 20) throws -> ProductQuantizer

    public func encode(vector: [Float]) throws -> [UInt8]
    public func approximateDistance(query: [Float], codes: [UInt8], metric: Metric) -> Float
    // internal: func distanceTable(query:metric:) -> [[Float]]?
}

// HNSWSearchCPU (Sources/MetalANNSCore/HNSWSearchCPU.swift) — do NOT modify
public enum HNSWSearchCPU {
    public static func search(...) async throws(ANNSError) -> [SearchResult]
    // internal: static func greedySearchLayer(...) throws(ANNSError) -> UInt32
}

// IndexConfiguration (Sources/MetalANNS/IndexConfiguration.swift) — MODIFY (add field)
public struct IndexConfiguration: Sendable, Codable {
    // ... existing fields ...
    public var hnswConfiguration: HNSWConfiguration
    public var repairConfiguration: RepairConfiguration
    // ADD: quantizedHNSWConfiguration: QuantizedHNSWConfiguration
}

// ANNSIndex (Sources/MetalANNS/ANNSIndex.swift) — MODIFY (build+search dispatch)
public actor ANNSIndex {
    private var hnsw: HNSWLayers?
    // ADD: private var quantizedHNSW: QuantizedHNSWLayers?
    // MODIFY: rebuildHNSWFromCurrentState() — dispatch to QuantizedHNSWBuilder when enabled
    // MODIFY: search() + rangeSearch() — use QuantizedHNSWSearchCPU when quantizedHNSW != nil
}
```

### Critical constraints (read before writing code)

**Constraint 1 — PQ training requires ≥ 256 vectors:**
```swift
// ProductQuantizer.train() line 29:
guard vectors.count >= centroidsPerSubspace else {
    throw ANNSError.constructionFailed(...)
}
```
Skip layers with fewer than 256 nodes cannot have a PQ trained on them. The builder
must fall back gracefully: if `nodesAtLayer.count < 256`, store `pq = nil` for that
layer and fall back to exact distance in `greedySearchLayer`.

**Constraint 2 — dimension divisibility:**
```swift
// ProductQuantizer.train() line 39:
guard dimension.isMultiple(of: numSubspaces) else { throw ... }
```
The builder must validate `dim.isMultiple(of: config.pqSubspaces)`. If not, it must
auto-reduce `pqSubspaces` to the largest divisor ≤ requested value, or throw if no
valid divisor exists.

**Constraint 3 — `distanceTable` is internal:**
`ProductQuantizer.distanceTable(query:metric:)` has no `public` modifier — it is
accessible within `MetalANNSCore` but not from `MetalANNS`. Since all new quantized
types go into `MetalANNSCore`, this is fine with no changes needed. **Do not make
`distanceTable` public** — it is an internal optimization detail.

**Constraint 4 — layer 0 uses the base graph, not skip layers:**
`QuantizedHNSWLayers` only replaces skip layers 1…maxLayer. Layer 0 (base graph beam
search via `BeamSearchCPU`) remains full precision. `QuantizedHNSWSearchCPU` calls
`BeamSearchCPU.search()` for the final layer-0 phase, identical to `HNSWSearchCPU`.

---

## Tasks

### Task 1: QuantizedHNSWConfiguration + Tests

**Acceptance**: `QuantizedHNSWConfigurationTests` passes (3 tests). First git commit.

**Checklist:**

- [ ] 1.1 — Create `Tests/MetalANNSTests/QuantizedHNSWConfigurationTests.swift` with tests:
  - `defaultValues` — `QuantizedHNSWConfiguration()` has `useQuantizedEdges == true`,
    `pqSubspaces == 4`, `base == HNSWConfiguration.default`
  - `disabledByFlag` — `QuantizedHNSWConfiguration(useQuantizedEdges: false)` stored correctly
  - `codableRoundTrip` — encode to JSON, decode back, verify equality

- [ ] 1.2 — **RED**: Tests fail (`QuantizedHNSWConfiguration` not defined)

- [ ] 1.3 — Create `Sources/MetalANNSCore/QuantizedHNSWConfiguration.swift`:
  ```swift
  import Foundation

  /// Controls whether and how HNSW skip-layer greedy search uses ADC instead of
  /// exact distances. When enabled, each skip-layer node's vector is replaced by
  /// a PQ code; distances are computed via M table lookups (O(M)) instead of full
  /// vector dot products (O(dim)).
  public struct QuantizedHNSWConfiguration: Sendable, Codable, Equatable {
      /// Base HNSW settings (layer count, M). Forwarded to HNSWBuilder.
      public var base: HNSWConfiguration

      /// When true, skip layers use ADC instead of exact distance.
      /// Set to false to disable quantization and use HNSWLayers/HNSWSearchCPU unchanged.
      public var useQuantizedEdges: Bool

      /// Number of PQ subspaces for skip-layer codebooks.
      /// Must divide evenly into the vector dimension. Smaller values = faster but less accurate.
      /// Default 4 is safe for dim=128 (subspace dim = 32).
      public var pqSubspaces: Int

      public static let `default` = QuantizedHNSWConfiguration(
          base: .default,
          useQuantizedEdges: true,
          pqSubspaces: 4
      )

      public init(
          base: HNSWConfiguration = .default,
          useQuantizedEdges: Bool = true,
          pqSubspaces: Int = 4
      ) {
          self.base = base
          self.useQuantizedEdges = useQuantizedEdges
          self.pqSubspaces = max(1, pqSubspaces)
      }
  }
  ```

- [ ] 1.4 — **GREEN**: All 3 tests pass

- [ ] 1.5 — **GIT**: `git commit -m "feat: add QuantizedHNSWConfiguration"`

---

### Task 2: QuantizedSkipLayer + QuantizedHNSWLayers + Tests

**Acceptance**: `QuantizedHNSWLayersTests` passes (4 tests). Second git commit.

**Checklist:**

- [ ] 2.1 — Create `Tests/MetalANNSTests/QuantizedHNSWLayersTests.swift` with tests:
  - `constructSkipLayer` — create `QuantizedSkipLayer` from a `SkipLayer` + `ProductQuantizer`
    + codes array, verify `base.adjacency.count` and `codes.count` match
  - `nilPQSkipLayer` — create `QuantizedSkipLayer` with `pq = nil`, verify no crash
    (fallback codepath for layers with < 256 nodes)
  - `constructQuantizedLayers` — wrap two `QuantizedSkipLayer`s in `QuantizedHNSWLayers`,
    verify `maxLayer == 2` and `entryPoint` accessible
  - `codableRoundTrip` — encode `QuantizedHNSWLayers` to JSON, decode back, verify `maxLayer`
    and `entryPoint` equal (PQ codebooks are Codable — they will round-trip)

- [ ] 2.2 — **RED**: Tests fail (types not defined)

- [ ] 2.3 — Create `Sources/MetalANNSCore/QuantizedSkipLayer.swift`:
  ```swift
  import Foundation

  /// A skip layer where each node's vector is replaced by a PQ code for fast ADC lookup.
  /// When `pq` is nil (layer has < 256 nodes), exact distance is used as fallback.
  public struct QuantizedSkipLayer: Sendable, Codable {
      /// The base skip layer — adjacency lists and node mappings unchanged.
      public var base: SkipLayer

      /// PQ codebook trained on this layer's node vectors.
      /// Nil when the layer has fewer than 256 nodes (cannot train PQ).
      public var pq: ProductQuantizer?

      /// PQ codes indexed by layer-local index. `codes[i]` encodes the vector
      /// for `base.layerIndexToNode[i]`.
      /// Empty when `pq == nil`.
      public var codes: [[UInt8]]

      public init(base: SkipLayer, pq: ProductQuantizer?, codes: [[UInt8]]) {
          self.base = base
          self.pq = pq
          self.codes = codes
      }
  }

  /// Complete quantized HNSW skip-layer structure. Replaces HNSWLayers when
  /// QuantizedHNSWConfiguration.useQuantizedEdges is true.
  /// Layer 0 still uses the base graph + BeamSearchCPU at full precision.
  public final class QuantizedHNSWLayers: Sendable, Codable {
      /// Quantized skip layers where `quantizedLayers[0]` corresponds to layer 1.
      public let quantizedLayers: [QuantizedSkipLayer]

      /// Maximum assigned layer in the hierarchy.
      public let maxLayer: Int

      /// Level multiplier (1 / ln(2)).
      public let mL: Double

      /// Entry point node ID in the highest populated layer.
      public let entryPoint: UInt32

      public init(
          quantizedLayers: [QuantizedSkipLayer] = [],
          maxLayer: Int = 0,
          mL: Double = 1.4426950408889634,
          entryPoint: UInt32 = 0
      ) {
          self.quantizedLayers = quantizedLayers
          self.maxLayer = maxLayer
          self.mL = mL
          self.entryPoint = entryPoint
      }

      /// Returns the quantized skip layer for `layer` (1-indexed, matching HNSWLayers).
      public func quantizedLayer(at layer: Int) -> QuantizedSkipLayer? {
          guard layer > 0, layer <= maxLayer else { return nil }
          return quantizedLayers[layer - 1]
      }
  }
  ```

- [ ] 2.4 — **GREEN**: All 4 tests pass

- [ ] 2.5 — **GIT**: `git commit -m "feat: add QuantizedSkipLayer and QuantizedHNSWLayers data structures"`

---

### Task 3: QuantizedHNSWBuilder

**Acceptance**: `QuantizedHNSWBuilderTests` passes (5 tests). Third git commit.

**Checklist:**

- [ ] 3.1 — Create `Tests/MetalANNSTests/QuantizedHNSWBuilderTests.swift` with tests:
  - `buildsLayersFromGraph` — build a 500-vector index (CPU path, `dim=128`), call
    `QuantizedHNSWBuilder.build(...)`, verify `maxLayer >= 1`
  - `pqTrainedPerLayer` — for a layer with ≥ 256 nodes, verify `pq != nil` in that
    `QuantizedSkipLayer`
  - `fallbackForSmallLayer` — build a 300-vector index and inject a mock layer with
    < 256 nodes; verify `pq == nil` for that layer (no crash)
  - `codesCountMatchesLayerNodes` — for each quantized layer, verify
    `codes.count == base.layerIndexToNode.count`
  - `dimensionMismatchAdjusted` — build with `pqSubspaces` that doesn't divide `dim`;
    verify builder either auto-adjusts to largest valid divisor or throws a clear error
    (document choice in Task Notes 3)

  > Use `dim=128`, 500 vectors, metric `.cosine` for all builder tests.
  > Seed random vectors deterministically: `(0..<500).map { i in (0..<128).map { Float(i % 64) / 64 } }`

- [ ] 3.2 — **RED**: Tests fail (`QuantizedHNSWBuilder` not defined)

- [ ] 3.3 — Create `Sources/MetalANNSCore/QuantizedHNSWBuilder.swift`:
  ```swift
  import Foundation

  public enum QuantizedHNSWBuilder {
      /// Build quantized HNSW skip layers from a complete base HNSW structure.
      ///
      /// For each skip layer L (1…maxLayer):
      ///   1. Collect the full vectors for all nodes assigned to that layer.
      ///   2. Resolve effective pqSubspaces: largest divisor of dim that is ≤ requested.
      ///   3. If node count ≥ 256: train ProductQuantizer, encode all layer node vectors.
      ///   4. Otherwise: store pq = nil, codes = [] (exact distance fallback).
      ///
      /// - Parameters:
      ///   - hnsw: An existing HNSWLayers structure (built by HNSWBuilder).
      ///   - vectors: Full-precision vectors for all graph nodes.
      ///   - config: Quantization settings (pqSubspaces, useQuantizedEdges).
      ///   - metric: Distance metric (forwarded to PQ training).
      public static func build(
          from hnsw: HNSWLayers,
          vectors: [[Float]],
          config: QuantizedHNSWConfiguration,
          metric: Metric
      ) throws(ANNSError) -> QuantizedHNSWLayers {
          guard hnsw.maxLayer > 0 else {
              return QuantizedHNSWLayers(
                  quantizedLayers: [],
                  maxLayer: 0,
                  mL: hnsw.mL,
                  entryPoint: hnsw.entryPoint
              )
          }
          guard !vectors.isEmpty else {
              throw ANNSError.constructionFailed("Cannot build QuantizedHNSWLayers with empty vectors")
          }

          let dim = vectors[0].count
          let effectiveSubspaces = largestDivisorOf(dim, atMost: config.pqSubspaces)
          guard effectiveSubspaces > 0 else {
              throw ANNSError.constructionFailed(
                  "No valid pqSubspaces ≤ \(config.pqSubspaces) divides dimension \(dim)"
              )
          }

          var quantizedLayers: [QuantizedSkipLayer] = []

          for layerIndex in 0..<hnsw.layers.count {
              let skipLayer = hnsw.layers[layerIndex]
              let nodeCount = skipLayer.layerIndexToNode.count

              if nodeCount >= 256 {
                  // Collect vectors for nodes in this layer
                  let layerVectors = skipLayer.layerIndexToNode.map { nodeID in
                      vectors[Int(nodeID)]
                  }

                  do {
                      let pq = try ProductQuantizer.train(
                          vectors: layerVectors,
                          numSubspaces: effectiveSubspaces,
                          centroidsPerSubspace: 256,
                          maxIterations: 20
                      )
                      let codes = try layerVectors.map { try pq.encode(vector: $0) }
                      quantizedLayers.append(QuantizedSkipLayer(base: skipLayer, pq: pq, codes: codes))
                  } catch let error as ANNSError {
                      // PQ training failed (e.g. dimension mismatch after adjustment) — fall back
                      quantizedLayers.append(QuantizedSkipLayer(base: skipLayer, pq: nil, codes: []))
                      _ = error  // suppress unused warning
                  } catch {
                      quantizedLayers.append(QuantizedSkipLayer(base: skipLayer, pq: nil, codes: []))
                  }
              } else {
                  // Too few nodes to train PQ — exact distance fallback
                  quantizedLayers.append(QuantizedSkipLayer(base: skipLayer, pq: nil, codes: []))
              }
          }

          return QuantizedHNSWLayers(
              quantizedLayers: quantizedLayers,
              maxLayer: hnsw.maxLayer,
              mL: hnsw.mL,
              entryPoint: hnsw.entryPoint
          )
      }

      /// Returns the largest integer ≤ `maxValue` that divides `n` evenly.
      /// Returns 0 if no such divisor exists > 0.
      static func largestDivisorOf(_ n: Int, atMost maxValue: Int) -> Int {
          let cap = min(maxValue, n)
          for candidate in stride(from: cap, through: 1, by: -1) {
              if n.isMultiple(of: candidate) { return candidate }
          }
          return 0
      }
  }
  ```

- [ ] 3.4 — **GREEN**: All 5 builder tests pass

- [ ] 3.5 — **GIT**: `git commit -m "feat: QuantizedHNSWBuilder — per-layer PQ training and encoding"`

---

### Task 4: QuantizedHNSWSearchCPU

**Acceptance**: `QuantizedHNSWSearchCPUTests` passes (4 tests). Fourth git commit.

**Checklist:**

- [ ] 4.1 — Create `Tests/MetalANNSTests/QuantizedHNSWSearchCPUTests.swift` with tests:
  - `searchReturnsResults` — build 500-vector index, build `QuantizedHNSWLayers`, run
    `QuantizedHNSWSearchCPU.search(...)`, verify returns exactly k=10 results
  - `recallVsExact` — run 20 queries on same 500-vector index, compare quantized vs
    exact (`HNSWSearchCPU`) recall@10, verify quantized recall > 0.80
    (lower bar than production — small test index has few skip-layer nodes)
  - `fallsBackForNilPQ` — build a `QuantizedHNSWLayers` where all layers have `pq = nil`,
    run search, verify it completes correctly (falls back to exact per-hop distance)
  - `emptyQueryThrows` — pass empty `vectors` array, verify `ANNSError.indexEmpty` thrown

- [ ] 4.2 — **RED**: Tests fail (`QuantizedHNSWSearchCPU` not defined)

- [ ] 4.3 — Create `Sources/MetalANNSCore/QuantizedHNSWSearchCPU.swift`:
  ```swift
  import Foundation

  public enum QuantizedHNSWSearchCPU {
      /// Search using quantized HNSW skip layers with ADC in greedy phase.
      /// Layer 0 uses full-precision BeamSearchCPU (unchanged).
      public static func search(
          query: [Float],
          vectors: [[Float]],
          hnsw: QuantizedHNSWLayers,
          baseGraph: [[(UInt32, Float)]],
          k: Int,
          ef: Int,
          metric: Metric
      ) async throws(ANNSError) -> [SearchResult] {
          guard k > 0 else { return [] }
          guard !vectors.isEmpty else { throw ANNSError.indexEmpty }
          guard vectors.count == baseGraph.count else {
              throw ANNSError.searchFailed("Graph size does not match vector count")
          }
          guard query.count == vectors[0].count else {
              throw ANNSError.dimensionMismatch(expected: vectors[0].count, got: query.count)
          }

          var currentEntry = Int(hnsw.entryPoint)

          if hnsw.maxLayer > 0 {
              for layer in stride(from: hnsw.maxLayer, through: 1, by: -1) {
                  currentEntry = Int(
                      try greedySearchLayer(
                          query: query,
                          vectors: vectors,
                          hnsw: hnsw,
                          layer: layer,
                          entryPoint: currentEntry,
                          metric: metric
                      )
                  )
              }
          }

          // Layer 0: full-precision beam search (identical to HNSWSearchCPU)
          do {
              return try await BeamSearchCPU.search(
                  query: query,
                  vectors: vectors,
                  graph: baseGraph,
                  entryPoint: currentEntry,
                  k: k,
                  ef: ef,
                  metric: metric
              )
          } catch let error as ANNSError {
              throw error
          } catch {
              throw ANNSError.searchFailed("Quantized HNSW layer-0 beam search failed: \(error)")
          }
      }

      /// Greedy descent on a single skip layer using ADC when PQ is available.
      static func greedySearchLayer(
          query: [Float],
          vectors: [[Float]],
          hnsw: QuantizedHNSWLayers,
          layer: Int,
          entryPoint: Int,
          metric: Metric
      ) throws(ANNSError) -> UInt32 {
          guard layer > 0, layer <= hnsw.maxLayer else {
              throw ANNSError.searchFailed("Invalid layer for quantized greedy search")
          }
          guard entryPoint >= 0, entryPoint < vectors.count else {
              throw ANNSError.searchFailed("Entry point out of bounds")
          }
          guard let qLayer = hnsw.quantizedLayer(at: layer) else {
              throw ANNSError.searchFailed("No quantized layer at \(layer)")
          }

          let skipLayer = qLayer.base
          let pq = qLayer.pq

          // Precompute ADC table once per layer (O(M×256) = one-time cost)
          // distanceTable is internal to MetalANNSCore — accessible here
          let adcTable: [[Float]]? = pq?.distanceTable(query: query, metric: metric)

          var current = UInt32(entryPoint)
          var currentDistance: Float

          // Compute initial distance for entry point
          if let adcTable,
             let layerIdx = skipLayer.nodeToLayerIndex[current],
             Int(layerIdx) < qLayer.codes.count {
              // ADC path
              currentDistance = approximateWithTable(
                  adcTable, codes: qLayer.codes[Int(layerIdx)], numSubspaces: pq!.numSubspaces
              )
          } else {
              // Exact fallback
              currentDistance = SIMDDistance.distance(query, vectors[entryPoint], metric: metric)
          }

          var improved = true
          var iterations = 0
          let maxIterations = 128

          while improved && iterations < maxIterations {
              improved = false
              iterations += 1

              guard let neighbors = skipLayer.adjacency[
                  skipLayer.nodeToLayerIndex[current].map(Int.init) ?? -1
              ] as? [UInt32] else {
                  break
              }

              for neighborID in neighbors {
                  if neighborID == UInt32.max || Int(neighborID) >= vectors.count { continue }

                  let neighborDist: Float
                  if let adcTable,
                     let layerIdx = skipLayer.nodeToLayerIndex[neighborID],
                     Int(layerIdx) < qLayer.codes.count {
                      // ADC path: O(M) table lookups
                      neighborDist = approximateWithTable(
                          adcTable, codes: qLayer.codes[Int(layerIdx)],
                          numSubspaces: pq!.numSubspaces
                      )
                  } else {
                      // Exact fallback: O(dim)
                      neighborDist = SIMDDistance.distance(query, vectors[Int(neighborID)], metric: metric)
                  }

                  if neighborDist < currentDistance {
                      current = neighborID
                      currentDistance = neighborDist
                      improved = true
                  }
              }
          }

          return current
      }

      /// Inline ADC lookup — avoids calling `pq.approximateDistance()` to skip
      /// the `distanceTable` recomputation overhead (table already built above).
      @inline(__always)
      private static func approximateWithTable(
          _ table: [[Float]],
          codes: [UInt8],
          numSubspaces: Int
      ) -> Float {
          var dist: Float = 0
          for m in 0..<numSubspaces {
              dist += table[m][Int(codes[m])]
          }
          return dist
      }
  }
  ```

  > **⚠️ Adjacency access pattern:** The neighbor loop above uses a force-cast and
  > an awkward index path. Simplify using the pattern from `HNSWSearchCPU`:
  > ```swift
  > guard let neighbors = skipLayer.adjacency[
  >     guard let layerIdx = skipLayer.nodeToLayerIndex[current] else { break }
  >     Int(layerIdx)
  > ]
  > ```
  > Use a `guard let layerIdx = skipLayer.nodeToLayerIndex[current] else { break }`
  > before the neighbor loop and `skipLayer.adjacency[Int(layerIdx)]` directly.
  > Mirror the exact pattern from `HNSWSearchCPU.greedySearchLayer()` line 85-88.

- [ ] 4.4 — Fix the adjacency access pattern — mirror `HNSWSearchCPU.greedySearchLayer()` exactly:
  ```swift
  // Use guard let layerIdx pattern, not force-cast
  guard let layerIdxRaw = skipLayer.nodeToLayerIndex[current] else { break }
  let neighbors = skipLayer.adjacency[Int(layerIdxRaw)]
  ```

- [ ] 4.5 — **GREEN**: All 4 search tests pass

- [ ] 4.6 — **GIT**: `git commit -m "feat: QuantizedHNSWSearchCPU — ADC table lookup in greedy skip-layer descent"`

---

### Task 5: Integration into IndexConfiguration + ANNSIndex

**Acceptance**: `QuantizedHNSWIntegrationTests` passes (4 tests). Fifth git commit.

**Checklist:**

- [ ] 5.1 — Create `Tests/MetalANNSTests/QuantizedHNSWIntegrationTests.swift` with tests:
  - `defaultConfigUsesQuantized` — build `ANNSIndex` with default config (CPU path, no Metal),
    verify `indexConfiguration.quantizedHNSWConfiguration.useQuantizedEdges == true`
  - `searchWithQuantizedEnabled` — build 500-vector CPU index, search 10 queries, verify
    results.count == k (quantized path used, no crash)
  - `searchWithQuantizedDisabled` — build same index with
    `quantizedHNSWConfiguration.useQuantizedEdges = false`, verify results identical to
    default (no quantization, falls back to HNSWSearchCPU)
  - `quantizedRecall` — build 500-vector index, run 20 queries against the original vectors
    as ground truth (linear scan), verify recall@10 > 0.80

- [ ] 5.2 — **RED**: Tests fail (integration not wired up)

- [ ] 5.3 — Modify `Sources/MetalANNS/IndexConfiguration.swift`:
  - Add field: `public var quantizedHNSWConfiguration: QuantizedHNSWConfiguration`
  - Update `default` static:
    ```swift
    public static let `default` = IndexConfiguration(
        degree: 32,
        metric: .cosine,
        efConstruction: 100,
        efSearch: 64,
        maxIterations: 20,
        useFloat16: false,
        convergenceThreshold: 0.001,
        hnswConfiguration: .default,
        repairConfiguration: .default,
        quantizedHNSWConfiguration: .default    // ← new
    )
    ```
  - Update `init(...)` with new parameter (default `.default`)
  - Update `init(from decoder:)` to add:
    ```swift
    quantizedHNSWConfiguration = try container.decodeIfPresent(
        QuantizedHNSWConfiguration.self,
        forKey: .quantizedHNSWConfiguration
    ) ?? .default
    ```
  - Add `.quantizedHNSWConfiguration` to `CodingKeys` enum

- [ ] 5.4 — Modify `Sources/MetalANNS/ANNSIndex.swift`:
  - Add private property: `private var quantizedHNSW: QuantizedHNSWLayers?`
  - Set `quantizedHNSW = nil` in `init()`, `applyLoadedState()`, and `compact()` alongside
    the existing `hnsw = nil` reset
  - Modify `rebuildHNSWFromCurrentState()`:
    ```swift
    private func rebuildHNSWFromCurrentState() throws(ANNSError) {
        guard configuration.hnswConfiguration.enabled else {
            hnsw = nil
            quantizedHNSW = nil
            return
        }
        // ... existing early-return guards for context/empty ...

        let builtHNSW = try HNSWBuilder.buildLayers(
            vectors: vectors,
            graph: extractGraph(from: graph),
            nodeCount: vectors.count,
            metric: configuration.metric,
            config: configuration.hnswConfiguration
        )
        hnsw = builtHNSW

        // Build quantized layers if enabled (CPU path only)
        if configuration.quantizedHNSWConfiguration.useQuantizedEdges,
           builtHNSW.maxLayer > 0 {
            let extractedVectors = extractVectors(from: vectors)
            let qHNSW = try? QuantizedHNSWBuilder.build(
                from: builtHNSW,
                vectors: extractedVectors,
                config: configuration.quantizedHNSWConfiguration,
                metric: configuration.metric
            )
            quantizedHNSW = qHNSW   // nil if builder threw (e.g. dim constraint)
        } else {
            quantizedHNSW = nil
        }
    }
    ```
  - Modify `search()` and `rangeSearch()` CPU path to dispatch to `QuantizedHNSWSearchCPU`
    when `quantizedHNSW != nil`:
    ```swift
    if let qHNSW = quantizedHNSW {
        rawResults = try await QuantizedHNSWSearchCPU.search(
            query: query,
            vectors: extractedVectors,
            hnsw: qHNSW,
            baseGraph: extractedGraph,
            k: max(1, effectiveK),
            ef: max(1, effectiveEf),
            metric: searchMetric
        )
    } else if let hnsw {
        rawResults = try await HNSWSearchCPU.search(...)
    } else {
        rawResults = try await BeamSearchCPU.search(...)
    }
    ```

- [ ] 5.5 — **GREEN**: All 4 integration tests pass

- [ ] 5.6 — **REGRESSION**: All Phase 1-18 tests still pass. HNSW recall unchanged when
  `useQuantizedEdges = false`.

- [ ] 5.7 — **GIT**: `git commit -m "feat: integrate QuantizedHNSW into ANNSIndex build and search dispatch"`

---

### Task 6: Recall, Memory, and Performance Validation

**Acceptance**: `QuantizedHNSWBenchmarkTests` passes (4 tests). Sixth git commit.

**Checklist:**

- [ ] 6.1 — Create `Tests/MetalANNSTests/QuantizedHNSWBenchmarkTests.swift` with tests:
  - `quantizedVsExactRecall` — build 1000-vector index (dim=128, cosine), run 50 queries:
    - Exact: `IndexConfiguration` with `quantizedHNSWConfiguration.useQuantizedEdges = false`
    - Quantized: default config
    - Compute recall@10 for both. Assert: `quantizedRecall >= exactRecall - 0.05`
      (degradation < 5 percentage points; target < 3%)
  - `memoryReduction` — build 1000-vector index, measure:
    - `MemoryLayout<HNSWLayers>` vs encode `QuantizedHNSWLayers` to JSON and check byte count
    - Verify quantized representation JSON is smaller per skip-layer node than full Float32 vectors
    - Log the ratio; no hard assert (varies by dim/subspaces)
  - `adcFasterThanExact` — time 200 queries exact vs quantized on 1000-vector index:
    - Log elapsed times. Assert: quantized time ≤ exact time × 1.5
      (ADC should be ≤ exact; the 1.5x allows for JIT warmup variance in test context)
  - `pqTrainsInReasonableTime` — on a 5000-vector layer subset (simulate L1 of 20K index),
    time `ProductQuantizer.train(...)` with `numSubspaces=4`. Assert: completes in < 10s
    on any supported hardware

  > All tests skip on simulator with `#if targetEnvironment(simulator) return #endif` if GPU
  > is required; these tests are CPU-only and run everywhere.

- [ ] 6.2 — **GREEN**: All 4 benchmark tests pass (timing assertions are soft — log results)

- [ ] 6.3 — **GIT**: `git commit -m "test: add QuantizedHNSW recall, memory and performance validation"`

---

### Task 7: Persistence — save/load with QuantizedHNSWLayers

**Acceptance**: `QuantizedHNSWPersistenceTests` passes (3 tests). Seventh commit.

**Checklist:**

- [ ] 7.1 — Create `Tests/MetalANNSTests/QuantizedHNSWPersistenceTests.swift` with tests:
  - `saveAndLoadQuantizedIndex` — build 500-vector index with default config (quantized
    enabled), save, load, run 10 searches, verify results non-empty
  - `quantizedLayersSurviveRoundTrip` — save + load, verify `quantizedHNSW != nil` on
    loaded index (expose via a test-internal accessor or by checking search path)
  - `backwardCompatible` — load an index saved WITHOUT `quantizedHNSWConfiguration`
    (simulate by saving with old JSON lacking that key), verify load succeeds and
    `useQuantizedEdges == true` (default applied by `decodeIfPresent`)

  > For the persistence format, `QuantizedHNSWLayers` is `Codable`. Store it as a
  > separate sidecar alongside the existing `.meta.json`:
  > - `index.anns` — binary vectors + graph (unchanged, IndexSerializer)
  > - `index.anns.meta.json` — IndexConfiguration + SoftDeletion + MetadataStore (existing)
  > - `index.anns.qhnsw.json` — `QuantizedHNSWLayers` encoded via JSONEncoder (NEW)
  >
  > On `save(to:)`: after writing `.meta.json`, if `quantizedHNSW != nil` write `.qhnsw.json`.
  > On `load(from:)`: after `rebuildHNSWFromCurrentState()`, attempt to load `.qhnsw.json`;
  > if present, set `quantizedHNSW` from it and skip re-building from scratch.
  > This avoids re-running PQ training on every load (which could take seconds for large indexes).

- [ ] 7.2 — **RED**: Tests fail (save/load not wired for quantized layers)

- [ ] 7.3 — Modify `Sources/MetalANNS/ANNSIndex.swift`:

  Add `qhnswURL(for:)` helper:
  ```swift
  private nonisolated static func qhnswURL(for fileURL: URL) -> URL {
      URL(fileURLWithPath: fileURL.path + ".qhnsw.json")
  }
  ```

  In `save(to:)` (after the existing `.meta.json` write):
  ```swift
  if let qHNSW = quantizedHNSW {
      let qData = try JSONEncoder().encode(qHNSW)
      try qData.write(to: Self.qhnswURL(for: url), options: .atomic)
  }
  ```

  In `load(from:)` (after `rebuildHNSWFromCurrentState()`):
  ```swift
  // Load pre-built quantized layers if available (avoids re-training PQ)
  let qURL = qhnswURL(for: url)
  if FileManager.default.fileExists(atPath: qURL.path),
     let qData = try? Data(contentsOf: qURL),
     let qHNSW = try? JSONDecoder().decode(QuantizedHNSWLayers.self, from: qData) {
      await index.setQuantizedHNSW(qHNSW)
  }
  ```

  Add internal actor method:
  ```swift
  func setQuantizedHNSW(_ q: QuantizedHNSWLayers?) {
      self.quantizedHNSW = q
  }
  ```

- [ ] 7.4 — **GREEN**: All 3 persistence tests pass

- [ ] 7.5 — **REGRESSION**: All Phase 1-18 save/load tests still pass

- [ ] 7.6 — **GIT**: `git commit -m "feat: persist QuantizedHNSWLayers as sidecar .qhnsw.json on save/load"`

---

### Task 8 (was 7 in plan): Full Suite + Completion Signal

**Acceptance**: Full suite passes. Final commit.

**Checklist:**

- [ ] 8.1 — Build:
  ```
  xcodebuild build -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation
  ```
  → **BUILD SUCCEEDED**

- [ ] 8.2 — Test:
  ```
  xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation
  ```
  Expected new passing suites:
  - `QuantizedHNSWConfigurationTests` (3 tests)
  - `QuantizedHNSWLayersTests` (4 tests)
  - `QuantizedHNSWBuilderTests` (5 tests)
  - `QuantizedHNSWSearchCPUTests` (4 tests)
  - `QuantizedHNSWIntegrationTests` (4 tests)
  - `QuantizedHNSWBenchmarkTests` (4 tests)
  - `QuantizedHNSWPersistenceTests` (3 tests)

  All Phase 1-19 tests unchanged. Known `MmapTests` baseline failure allowed.

- [ ] 8.3 — Verify git log shows exactly 8 commits for this phase

- [ ] 8.4 — Manually verify the recall degradation claim:
  Run 100 queries on a 5000-vector index (dim=128, cosine). Log:
  - Exact HNSW recall@10
  - Quantized HNSW recall@10
  - Degradation (should be < 3% for dim=128, pqSubspaces=4)

- [ ] 8.5 — **GIT**: `git commit -m "chore: phase 20 complete - quantized HNSW skip layers"`

---

## Task Notes

Use this section to document decisions and issues as you work:

### Task 3 Notes
_(Document your decision on dimension mismatch handling: auto-adjust vs throw)_
_(Document actual skip-layer node counts for your test vectors)_

### Task 4 Notes
_(Document the `greedySearchLayer` adjacency access pattern used)_

### Task 6 Notes
_(Paste actual timing results: quantized ms vs exact ms for 200 queries on 1000 vectors)_
_(Paste actual recall: quantized vs exact recall@10)_

### Task 7 Notes
_(Document whether you store QuantizedHNSWLayers in full JSON or compressed)_
_(Document what happens when .qhnsw.json is present but stale/mismatched with base index)_

---

## Success Criteria

✅ `QuantizedHNSWConfiguration` — `useQuantizedEdges`, `pqSubspaces`, `base: HNSWConfiguration`, Codable
✅ `QuantizedSkipLayer` — wraps `SkipLayer` + optional `ProductQuantizer` + codes
✅ `QuantizedHNSWLayers` — Codable, mirrors `HNSWLayers` API
✅ `QuantizedHNSWBuilder` — per-layer PQ training, graceful fallback for < 256 nodes
✅ `QuantizedHNSWSearchCPU` — ADC table precomputed once per layer, exact fallback when pq==nil
✅ `IndexConfiguration` — `quantizedHNSWConfiguration` field with `decodeIfPresent` backward compat
✅ `ANNSIndex` — builds quantized layers in `rebuildHNSWFromCurrentState()`, dispatches in search
✅ Recall degradation < 5% vs exact HNSW (target < 3% for dim=128)
✅ Persistence — `.qhnsw.json` sidecar written on save, loaded before rebuild on load
✅ No regressions across all Phase 1-19 tests
✅ GPU path unchanged — quantized layers only used in CPU (no Metal context) path

---

## Anti-Patterns

❌ **Don't** make `ProductQuantizer.distanceTable()` public — it's internal to `MetalANNSCore`
❌ **Don't** call `pq.approximateDistance()` in the hot loop — it calls `distanceTable()` internally
   on every invocation. Precompute the table once per layer with `distanceTable()` and use
   `approximateWithTable()` inline
❌ **Don't** quantize the base graph (layer 0) — `BeamSearchCPU` handles it with full precision
❌ **Don't** modify `HNSWLayers`, `SkipLayer`, `HNSWBuilder`, or `HNSWSearchCPU` — new types
   are additive; old types remain for non-quantized fallback
❌ **Don't** crash when a skip layer has < 256 nodes — set `pq = nil` and fall back to exact
❌ **Don't** train PQ with `maxIterations > 20` in the builder — PQ training is O(N×k×iter);
   for large layers 20 iterations is sufficient and keeps build time under 100ms for L1
❌ **Don't** re-run PQ training on every `load()` — write `.qhnsw.json` on save, load it on read
❌ **Don't** set hard timing assertions for ADC vs exact comparison in tests — JIT warmup and
   hardware variation make < 20% differences unreliable; log and verify > 0 speedup
❌ **Don't** forget to add `quantizedHNSW = nil` in `insert()` and `batchInsert()` alongside
   the existing `hnsw = nil` — the quantized index is invalidated by inserts just as the
   plain HNSW is

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `Sources/MetalANNSCore/QuantizedHNSWConfiguration.swift` | **New** | Config struct |
| `Sources/MetalANNSCore/QuantizedSkipLayer.swift` | **New** | Data structures (QuantizedSkipLayer + QuantizedHNSWLayers) |
| `Sources/MetalANNSCore/QuantizedHNSWBuilder.swift` | **New** | Per-layer PQ training + encoding |
| `Sources/MetalANNSCore/QuantizedHNSWSearchCPU.swift` | **New** | ADC greedy search |
| `Sources/MetalANNS/IndexConfiguration.swift` | **Modified** | Add `quantizedHNSWConfiguration` |
| `Sources/MetalANNS/ANNSIndex.swift` | **Modified** | `quantizedHNSW` property, build/search dispatch, save/load |
| `Tests/MetalANNSTests/QuantizedHNSWConfigurationTests.swift` | **New** | Config tests |
| `Tests/MetalANNSTests/QuantizedHNSWLayersTests.swift` | **New** | Data structure tests |
| `Tests/MetalANNSTests/QuantizedHNSWBuilderTests.swift` | **New** | Builder tests |
| `Tests/MetalANNSTests/QuantizedHNSWSearchCPUTests.swift` | **New** | Search correctness tests |
| `Tests/MetalANNSTests/QuantizedHNSWIntegrationTests.swift` | **New** | End-to-end integration tests |
| `Tests/MetalANNSTests/QuantizedHNSWBenchmarkTests.swift` | **New** | Recall + performance tests |
| `Tests/MetalANNSTests/QuantizedHNSWPersistenceTests.swift` | **New** | Save/load tests |

**Total new code: ~700 lines (including tests). Two existing files modified.**

---

## Commits Expected

1. `feat: add QuantizedHNSWConfiguration`
2. `feat: add QuantizedSkipLayer and QuantizedHNSWLayers data structures`
3. `feat: QuantizedHNSWBuilder — per-layer PQ training and encoding`
4. `feat: QuantizedHNSWSearchCPU — ADC table lookup in greedy skip-layer descent`
5. `feat: integrate QuantizedHNSW into ANNSIndex build and search dispatch`
6. `test: add QuantizedHNSW recall, memory and performance validation`
7. `feat: persist QuantizedHNSWLayers as sidecar .qhnsw.json on save/load`
8. `chore: phase 20 complete - quantized HNSW skip layers`
