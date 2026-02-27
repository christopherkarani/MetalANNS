# Phase 16: Product Quantization + IVFPQ Implementation

> **For Claude:** This is an **implementation prompt** for Phase 16 of MetalANNS v3. Execute via TDD (RED→GREEN→commit). Dispatch to subagent; orchestrator reviews using the R1-R13 checklist in `tasks/todo.md`.

**Goal:** Implement full IVFPQ (coarse IVF + fine PQ) to compress vectors 32-64x while maintaining >0.8 recall@10. This enables 1M+ vector indices on-device.

**Architecture:**
```
Training:
  Vectors → Split into M subspaces (e.g., M=8)
          → K-means per subspace → 256 centroids per subspace (Ks=256, UInt8 codes)
          → K-means coarse → 256 IVF centroids

Encoding:
  Vector → Find nearest IVF centroid
         → Compute residual = vector - centroid
         → PQ encode residual into M bytes (one per subspace)
         → Result: (centroid_id, M-byte PQ code)

Search (ADC — Asymmetric Distance Computation):
  Query → Find nprobe nearest IVF centroids
       → For each cluster:
            1. Compute distance table: M × 256 floats (subspace × centroid distances)
            2. Scan PQ codes using table lookups (fast)
       → Merge results → Top-k
```

**Tech Stack:**
- Swift 6.2 with typed throws (`throws(ANNSError)`)
- Existing `SIMDDistance` (CPU fallback)
- Existing `KMeans` from Phase 12
- New Metal kernels: `pq_compute_distance_table`, `pq_adc_scan`
- No external ML libraries

**Phase Dependencies:**
- Phase 13: Typed throws required for error handling
- Phase 14: Graph repair (IVFPQ doesn't need it, but integrates cleanly)
- Phase 15: HNSW (IVFPQ is separate actor, doesn't affect HNSW)
- Phase 12: KMeans reused directly

**Key Decisions (Locked In):**
- IVFPQ is **standalone actor** (`IVFPQIndex`) — does NOT modify `ANNSIndex`
- GPU kernels: ADC computation only; training stays CPU-side
- Training: k-means++ on CPU (K=256 for UInt8 codes always)
- Encoding: CPU-side (simple residual + PQ assignment)
- Search: GPU ADC scan + CPU fallback

---

## System Context

### Existing Infrastructure (Reuse)

**KMeans** (`Sources/MetalANNSCore/KMeans.swift` from Phase 12):
- `static func fit(vectors: [[Float]], k: Int, maxIterations: Int) throws(ANNSError) -> (centroids: [[Float]], assignments: [UInt32])`
- Used for both coarse and fine-grained centroid training

**SIMDDistance** (existing):
- `static func distance(_ a: [Float], _ b: [Float], metric: Metric) -> Float`
- CPU fallback for distance computations

**MetalContext** (existing):
- `device`, `commandQueue`, `library`, `pipelineCache`
- GPU pipeline management

**VectorStorage** protocol (existing):
- `func vector(at: Int) -> [Float]`
- `func insert(vector: [Float], at: Int) throws`
- `var dim: Int`, `var count: Int`

### New Concepts

**Product Quantizer:**
- Split D-dimensional vector into M subspaces of size D/M
- Train Ks=256 centroids per subspace (UInt8 codes)
- Each subspace assigns its residual to nearest centroid → 1 byte
- Result: D-dimensional vector → M bytes

**Asymmetric Distance Computation (ADC):**
- Pre-compute distance table: for each subspace, distances from query residual to all 256 centroids
- Scan PQ codes: for each vector's M-byte code, sum table lookups (M operations per vector)
- ~100x faster than computing full-dimensional distances

**IVF (Inverted File):**
- Coarse quantization: partition vectors into K clusters
- Search: find nprobe clusters, then scan those clusters only
- Reduces search space from N vectors to N/K vectors

---

## Tasks

### Task 1: Quantization Storage Protocol and Tests

**Acceptance**: `QuantizedStorageTests` passes. First git commit.

**Checklist:**

- [ ] 1.1 — Create `Tests/MetalANNSTests/QuantizedStorageTests.swift` with tests:
  - `protocolExists` — verify `QuantizedStorage` protocol can be instantiated (via dummy impl)
  - `reconstructionError` — mock quantizer, verify reconstruction error < 5% of original norm
  - `codableRoundTrip` — verify quantized storage implements Codable (via mock)

- [ ] 1.2 — **RED**: Tests fail (protocol not defined)

- [ ] 1.3 — Create `Sources/MetalANNSCore/QuantizedStorage.swift`:
  ```swift
  public protocol QuantizedStorage: Sendable {
      /// Total number of vectors
      var count: Int { get }

      /// Original (unquantized) dimension
      var originalDimension: Int { get }

      /// Approximate distance from query to vector at index using ADC
      func approximateDistance(query: [Float], to index: UInt32, metric: Metric) -> Float

      /// Lossy reconstruction of vector (for inspection/debugging)
      func reconstruct(at index: UInt32) -> [Float]
  }
  ```

- [ ] 1.4 — **GREEN**: All 3 tests pass

- [ ] 1.5 — **GIT**: `git commit -m "feat: add QuantizedStorage protocol for ADC-based distance computation"`

---

### Task 2: Product Quantizer Training and Encoding

**Acceptance**: `ProductQuantizerTests` passes with training and encoding verification. Second git commit.

**Checklist:**

- [ ] 2.1 — Create `Tests/MetalANNSTests/ProductQuantizerTests.swift` with tests:
  - `trainPQCodebook` — train codebook on 10,000 random 128-dim vectors, verify no errors
  - `encodeVectors` — encode 100 vectors using trained codebook, verify output is M bytes per vector
  - `reconstructionAccuracy` — encode then reconstruct 100 vectors, verify L2 reconstruction error < 2% of original norm
  - `distanceApproximationAccuracy` — compare PQ approximate distances to exact distances, verify correlation > 0.95

- [ ] 2.2 — **RED**: Tests fail (ProductQuantizer not defined)

- [ ] 2.3 — Create `Sources/MetalANNSCore/ProductQuantizer.swift`:
  ```swift
  public struct ProductQuantizer: Sendable, Codable {
      public let numSubspaces: Int         // M: split dimension into M parts
      public let centroidsPerSubspace: Int // Ks: centroids per subspace (always 256 for UInt8)
      public let subspaceDimension: Int    // D/M

      /// Codebooks: [subspace_idx][centroid_idx] = [floats of size D/M]
      public let codebooks: [[[Float]]]

      /// Train PQ codebook from vectors using k-means per subspace
      public static func train(
          vectors: [[Float]],
          numSubspaces: Int = 8,
          centroidsPerSubspace: Int = 256,
          maxIterations: Int = 20
      ) throws(ANNSError) -> ProductQuantizer

      /// Encode a vector into M UInt8 codes (one per subspace)
      public func encode(vector: [Float]) throws(ANNSError) -> [UInt8]

      /// Reconstruct (lossy) from M codes
      public func reconstruct(codes: [UInt8]) throws(ANNSError) -> [Float]

      /// Compute approximate distance using codes (for testing)
      public func approximateDistance(query: [Float], codes: [UInt8], metric: Metric) -> Float
  }
  ```
  - `train()`: Split each vector into M subspaces, run KMeans.fit() per subspace
  - `encode()`: For each subspace, find nearest centroid index → store as UInt8
  - `reconstruct()`: For each subspace, fetch centroid from codebook
  - `approximateDistance()`: Pre-compute distance table per subspace, sum lookups for each code

- [ ] 2.4 — **EDGE CASES**:
  - Verify M divides D evenly (throw if not)
  - Handle Ks != 256 gracefully (clamp to [1, 256])
  - Verify vectors.isEmpty guard (return error)

- [ ] 2.5 — **GREEN**: All 4 tests pass, reconstruction error < 2%, distance correlation > 0.95

- [ ] 2.6 — **GIT**: `git commit -m "feat: implement ProductQuantizer with training, encoding, and reconstruction"`

---

### Task 3: PQ Vector Buffer Storage

**Acceptance**: `PQVectorBufferTests` passes. Third git commit.

**Checklist:**

- [ ] 3.1 — Create `Tests/MetalANNSTests/PQVectorBufferTests.swift` with tests:
  - `initAndInsert` — create PQVectorBuffer, insert 100 PQ-encoded vectors, verify count
  - `approximateDistance` — insert vectors, compute approximate distances, verify consistency
  - `memoryReduction` — compare PQVectorBuffer size vs uncompressed VectorBuffer, expect 30-60x reduction

- [ ] 3.2 — **RED**: Tests fail (PQVectorBuffer not defined)

- [ ] 3.3 — Create `Sources/MetalANNSCore/PQVectorBuffer.swift`:
  ```swift
  public final class PQVectorBuffer: VectorStorage, Sendable {
      public let dim: Int  // original dimension
      public private(set) var count: Int
      public let capacity: Int

      private let pq: ProductQuantizer
      private var codes: [[UInt8]]  // codes[i] = M-byte code for vector i
      private var originalVectors: [[Float]]  // keep originals for training/fallback (cleared post-training)

      public init(
          capacity: Int,
          dim: Int,
          pq: ProductQuantizer
      ) throws(ANNSError)

      public func vector(at index: Int) -> [Float]

      public func insert(vector: [Float], at index: Int) throws

      public func approximateDistance(query: [Float], to index: UInt32, metric: Metric) -> Float
  }
  ```
  - Stores M-byte codes only (not full vectors after training)
  - `vector(at:)` reconstructs from codes
  - `approximateDistance(...)` implements fast ADC table lookup
  - `insert()` encodes vector and stores codes

- [ ] 3.4 — **GREEN**: All 3 tests pass, memory reduction verified

- [ ] 3.5 — **GIT**: `git commit -m "feat: implement PQVectorBuffer with ADC distance computation"`

---

### Task 4: IVF Coarse Quantization and IVFPQIndex Actor

**Acceptance**: `IVFPQIndexTests` passes with training, add, and search. Fourth git commit.

**Checklist:**

- [ ] 4.1 — Create `Tests/MetalANNSTests/IVFPQIndexTests.swift` with tests:
  - `trainAndAdd` — train IVFPQIndex on 10K vectors, add 1K vectors, verify count=1K
  - `searchRecall` — train on 10K, add 1K, search 100 queries, verify recall@10 > 0.8
  - `nprobeEffect` — search with nprobe=1, 4, 16; verify recall increases with nprobe
  - `memoryFootprint` — measure total index size (coarse centroids + PQ codebooks + vector codes), expect < original/30

- [ ] 4.2 — **RED**: Tests fail (IVFPQIndex not defined)

- [ ] 4.3 — Create `Sources/MetalANNS/IVFPQIndex.swift`:
  ```swift
  public struct IVFPQConfiguration: Sendable, Codable {
      public var numSubspaces: Int = 8          // M: PQ subspaces
      public var numCentroids: Int = 256        // Ks: centroids per subspace (UInt8)
      public var numCoarseCentroids: Int = 256  // IVF: coarse partitions
      public var nprobe: Int = 8                // search: clusters to probe
      public var metric: Metric = .l2
      public var trainingIterations: Int = 20
  }

  public actor IVFPQIndex: Sendable {
      private let config: IVFPQConfiguration
      private let coarseQuantizer: ProductQuantizer  // single M=1 codebook (D→1 byte)
      private let pq: ProductQuantizer              // M subspaces
      private let vectorBuffer: PQVectorBuffer       // compressed vectors
      private var coarseAssignments: [UInt32]       // coarseAssignments[i] = cluster ID for vector i

      public init(capacity: Int, dimension: Int, config: IVFPQConfiguration) throws(ANNSError)

      /// Train coarse and fine codebooks on training data
      public func train(vectors: [[Float]]) async throws(ANNSError)

      /// Add vectors to index (post-training)
      public func add(vectors: [[Float]], ids: [UInt32]) async throws(ANNSError)

      /// Search: find nprobe clusters, scan, return top-k
      public func search(query: [Float], k: Int, nprobe: Int?) async -> [SearchResult]

      public var count: Int { get }
  }
  ```
  - `train()`:
    1. Run KMeans with K=numCoarseCentroids on full vectors → coarse centroids
    2. Assign each training vector to nearest coarse centroid
    3. For each coarse cluster, run KMeans per subspace on residuals (vector - coarse centroid) → PQ codebooks
  - `add()`: Assign each vector to coarse cluster, encode with PQ, store codes
  - `search()`:
    1. Compute distances from query to all coarse centroids
    2. Find nprobe nearest clusters
    3. For each cluster, ADC scan (compute distance table, sum code lookups)
    4. Merge top-k across clusters

- [ ] 4.4 — **INTEGRATION NOTES**:
  - IVFPQIndex is a **standalone actor** — does NOT integrate into ANNSIndex
  - Intended for large-scale indices where compression is critical
  - No modification to existing ANNSIndex code

- [ ] 4.5 — **GREEN**: All 4 tests pass, recall@10 > 0.8, memory < original/30

- [ ] 4.6 — **GIT**: `git commit -m "feat: implement IVFPQIndex with coarse and fine quantization"`

---

### Task 5: Metal ADC Distance Kernels (GPU Acceleration)

**Acceptance**: `IVFPQGPUTests` passes. Fifth git commit.

**Checklist:**

- [ ] 5.1 — Create `Tests/MetalANNSTests/IVFPQGPUTests.swift` with tests:
  - `gpuVsCpuDistances` — 100 queries × 1000 vectors, GPU ADC vs CPU, tolerance 1e-3
  - Skip on simulator

- [ ] 5.2 — **RED**: Tests fail (GPU kernels not implemented)

- [ ] 5.3 — Create `Sources/MetalANNSCore/Shaders/PQDistance.metal`:
  ```metal
  #include <metal_stdlib>
  using namespace metal;

  /// Compute distance table: M × Ks floats (subspace × centroid distances from query residual)
  /// buffer(0) = query residual (D floats)
  /// buffer(1) = codebooks (concatenated: subspace0[Ks][D/M] + subspace1[Ks][D/M] + ...)
  /// buffer(2) = output distance table (M × Ks floats)
  /// buffer(3) = M (uint)
  /// buffer(4) = Ks (uint)
  /// buffer(5) = D/M subspace dimension (uint)
  kernel void pq_compute_distance_table(
      device const float *query [[buffer(0)]],
      device const float *codebooks [[buffer(1)]],
      device float *distTable [[buffer(2)]],
      device const uint &M [[buffer(3)]],
      device const uint &Ks [[buffer(4)]],
      device const uint &subspaceDim [[buffer(5)]],
      uint2 gid [[thread_position_in_grid]]
  ) {
      uint subspace = gid.x;
      uint centroid = gid.y;

      if (subspace >= M || centroid >= Ks) return;

      // Query residual for this subspace
      const float *queryResidual = query + subspace * subspaceDim;

      // Centroid for this (subspace, centroid) pair
      uint cbOffset = subspace * Ks * subspaceDim + centroid * subspaceDim;
      const float *cb = codebooks + cbOffset;

      // Compute L2 distance
      float dist = 0.0f;
      for (uint d = 0; d < subspaceDim; d++) {
          float diff = queryResidual[d] - cb[d];
          dist += diff * diff;
      }

      // Store in distance table
      distTable[subspace * Ks + centroid] = dist;
  }

  /// ADC scan: PQ codes × distance table → approximate distances
  /// buffer(0) = PQ codes (vectorCount × M bytes)
  /// buffer(1) = distance table (M × Ks floats, in threadgroup memory after broadcast)
  /// buffer(2) = output distances (vectorCount floats)
  /// buffer(3) = M (uint)
  /// buffer(4) = Ks (uint)
  /// buffer(5) = vectorCount (uint)
  kernel void pq_adc_scan(
      device const uchar *codes [[buffer(0)]],
      device const float *distTable [[buffer(1)]],
      device float *distances [[buffer(2)]],
      device const uint &M [[buffer(3)]],
      device const uint &Ks [[buffer(4)]],
      device const uint &vectorCount [[buffer(5)]],
      threadgroup float *tgDistTable [[threadgroup(0)]],
      uint tid [[thread_index_in_threadgroup]],
      uint gid [[thread_position_in_grid]]
  ) {
      uint vectorIdx = gid;
      if (vectorIdx >= vectorCount) return;

      // Load distance table to threadgroup memory (broadcast to all threads).
      // Host side must set threadgroup memory length to `M * Ks * sizeof(float)` at index 0.
      for (uint i = tid; i < M * Ks; i += 32) {
          if (i < M * Ks) {
              tgDistTable[i] = distTable[i];
          }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Scan PQ codes for this vector
      float dist = 0.0f;
      for (uint m = 0; m < M; m++) {
          uchar code = codes[vectorIdx * M + m];
          dist += tgDistTable[m * Ks + code];
      }

      distances[vectorIdx] = sqrt(dist);
  }
  ```

- [ ] 5.4 — Update `IVFPQIndex` to use GPU kernels when `MetalContext` available:
  - In `search()`, if GPU available: call GPU ADC scan via MetalContext
  - Otherwise: CPU ADC scan using native loops

- [ ] 5.5 — **GREEN**: GPU vs CPU tests pass, tolerance 1e-3

- [ ] 5.6 — **GIT**: `git commit -m "feat: add Metal ADC distance kernels for GPU-accelerated PQ search"`

---

### Task 6: Persistence and Round-Trip Tests

**Acceptance**: `IVFPQPersistenceTests` passes. Sixth git commit.

**Checklist:**

- [ ] 6.1 — Create `Tests/MetalANNSTests/IVFPQPersistenceTests.swift` with tests:
  - `saveThenLoad` — build IVFPQIndex, save to disk, load, verify count and search results
  - `roundTripAccuracy` — save then load, run same search before/after, verify results identical

- [ ] 6.2 — **RED**: Tests fail (serialization not implemented)

- [ ] 6.3 — Extend `IVFPQIndex` with persistence:
  ```swift
  public func save(to path: String) async throws(ANNSError)
  public static func load(from path: String) async throws(ANNSError) -> IVFPQIndex
  ```
  - Format: header (magic: "IVFP", version: 1) + config + coarse codebook + PQ codebooks + vector codes + coarse assignments
  - Use `Codable` for structured data, binary for vector/code arrays

- [ ] 6.4 — **GREEN**: Both tests pass

- [ ] 6.5 — **GIT**: `git commit -m "feat: add IVFPQIndex persistence (save/load)"`

---

### Task 7: Comprehensive Test Suite and Performance Validation

**Acceptance**: Full IVFPQ test suite passes. Seventh git commit.

**Checklist:**

- [ ] 7.1 — Create `Tests/MetalANNSTests/IVFPQComprehensiveTests.swift` consolidating:
  - All QuantizedStorage tests
  - All ProductQuantizer tests
  - All PQVectorBuffer tests
  - All IVFPQIndex tests
  - All GPU ADC tests (skip on simulator)
  - All persistence tests
  - **New**: Performance benchmarks:
    - `benchmarkSearchThroughput` — measure queries/sec for 1M-vector index, record QPS
    - `benchmarkMemoryUsage` — measure peak memory during search, record MB
    - `benchmarkRecallVsQPS` — sweep nprobe=1..16, plot recall vs QPS

- [ ] 7.2 — **RED**: Some tests may fail if implementations incomplete

- [ ] 7.3 — **GREEN**: All tests pass, benchmarks recorded

- [ ] 7.4 — **EXPECTED RESULTS**:
  - Recall@10: > 0.80 (with nprobe=8)
  - QPS: 1000-5000 queries/sec (depending on hardware, index size)
  - Memory: < 17 MB for 1M 128-dim vectors (512 MB uncompressed)
  - GPU speedup: 2-5x vs CPU on large batches

- [ ] 7.5 — **REGRESSION**: All Phase 13-15 tests still pass

- [ ] 7.6 — **GIT**: `git commit -m "feat: add comprehensive IVFPQ test suite with performance validation"`

---

### Task 8: Full Suite and Completion Signal

**Acceptance**: Full test suite passes. Eighth git commit.

**Checklist:**

- [ ] 8.1 — Run: `xcodebuild build -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation`
  - Expected: **BUILD SUCCEEDED**

- [ ] 8.2 — Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation`
  - Expected: All IVFPQ tests pass, Phase 13-15 tests unchanged
  - Allow known baseline `MmapTests` failure

- [ ] 8.3 — Verify git log shows exactly 8 commits with conventional messages

- [ ] 8.4 — Update Phase Complete Signal section in `tasks/todo.md`

- [ ] 8.5 — **GIT**: `git commit -m "chore: phase 16 complete - IVFPQ quantization and compression"`

---

## Success Criteria

✅ **Compression**: PQVectorBuffer reduces memory by 30-64x
✅ **Recall**: recall@10 > 0.80 with nprobe=8
✅ **Speed**: search > 1000 QPS on 100K vectors
✅ **GPU**: ADC kernels 2-5x faster than CPU
✅ **Isolation**: IVFPQIndex is standalone (no ANNSIndex changes)
✅ **Persistence**: Round-trip save/load verified
✅ **No regressions**: All Phase 13-15 tests pass

---

## Anti-Patterns

❌ **Don't** integrate IVFPQIndex into ANNSIndex — it's intentionally separate
❌ **Don't** use Ks != 256 (UInt8 codes require exactly 256)
❌ **Don't** forget to handle M not dividing D evenly
❌ **Don't** skip GPU ADC testing — performance critical
❌ **Don't** use float reconstruction in production (lossy by design)
❌ **Don't** over-train coarse k-means (20 iterations sufficient)
❌ **Don't** store original vectors post-training (only codes)
❌ **Don't** assume nprobe=8 is optimal (benchmark your use case)

---

## Files Summary

| File | Purpose | Size Est. |
|------|---------|-----------|
| `QuantizedStorage.swift` | Protocol for AD-based search | 50 lines |
| `ProductQuantizer.swift` | PQ training, encoding, reconstruction | 250 lines |
| `PQVectorBuffer.swift` | ADC vector storage | 200 lines |
| `IVFPQIndex.swift` | Coarse + fine quantization actor | 400 lines |
| `PQDistance.metal` | GPU ADC kernels | 150 lines |
| `IVFPQConfiguration.swift` | Config struct | 50 lines |
| `*Tests.swift` (6 test files) | Comprehensive test suite | 1500 lines |

**Total new code: ~2600 lines (including tests)**

---

## Commits Expected

1. QuantizedStorage protocol
2. ProductQuantizer training/encoding
3. PQVectorBuffer with ADC
4. IVFPQIndex actor
5. Metal ADC kernels
6. Persistence
7. Comprehensive tests
8. Phase complete signal
