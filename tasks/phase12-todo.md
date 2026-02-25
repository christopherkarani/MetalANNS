# Phase 12: Large-Scale (Combined Phase 11+12) — Todo

**Branch**: `Phase-12` (current)
**Baseline**: 33 commits, 78+ tests (after Phase 10)
**Target**: 38 commits, 89+ tests, all passing (except known MmapTests baseline issue)

---

## Known Baseline Issue

MmapTests has a pre-existing failure from Phase 10 (index-capacity on insert-after-load). Do NOT fix. Acceptance is focused on new tests + non-regression of other existing tests.

---

## Task 31: Filtered Search with Metadata Predicates

- [ ] 31.1 Create `Sources/MetalANNSCore/SearchFilter.swift` — recursive enum with Sendable conformance
- [ ] 31.2 Create `Sources/MetalANNSCore/MetadataStore.swift` — column-oriented storage (string, float, int columns)
- [ ] 31.3 Implement `MetadataStore.set()` overloads for String, Float, Int64
- [ ] 31.4 Implement `MetadataStore.matches(id:filter:)` — recursive filter evaluation
- [ ] 31.5 Implement `MetadataStore.remapped(using:)` for compaction compatibility
- [ ] 31.6 Implement `MetadataStore.remove(id:)` for deletion cleanup
- [ ] 31.7 Create `Tests/MetalANNSTests/FilteredSearchTests.swift` with 3 tests (RED)
- [ ] 31.8 Add `metadataStore: MetadataStore` private property to `ANNSIndex`
- [ ] 31.9 Add `setMetadata(_:value:for:)` overloads (String, Float, Int64) to `ANNSIndex`
- [ ] 31.10 Modify `ANNSIndex.search()` signature: add `filter: SearchFilter? = nil`
- [ ] 31.11 Implement over-fetch: when filter != nil, use `effectiveK = min(vectors.count, k * 4 + deletedCount)`
- [ ] 31.12 Apply metadata filter after softDeletion filter: `filtered.filter { metadataStore.matches(...) }`
- [ ] 31.13 Update `PersistedMetadata` struct: add `metadataStore: MetadataStore?` (optional for backward compat)
- [ ] 31.14 Update `applyLoadedState` signature: add `metadataStore: MetadataStore = MetadataStore()` param
- [ ] 31.15 Update `save()`: include metadataStore in `.meta.json`
- [ ] 31.16 Update `load()` and `loadMmap()`: read metadataStore from `.meta.json` (default empty if nil)
- [ ] 31.17 Run `FilteredSearchTests` — verify all 3 tests PASS (GREEN)
- [ ] 31.18 Run full test suite — verify 0 new regressions
- [ ] 31.19 Commit: `feat: add filtered search with metadata predicates`

---

## Task 32: Range Search

- [ ] 32.1 Create `Tests/MetalANNSTests/RangeSearchTests.swift` with 2 tests (RED)
- [ ] 32.2 Add `rangeSearch(query:maxDistance:limit:filter:)` to `ANNSIndex`
- [ ] 32.3 Implement: use large ef to explore broadly, filter by `score <= maxDistance`
- [ ] 32.4 Integrate metadata filter support in range search (reuse from Task 31)
- [ ] 32.5 Guard: maxDistance > 0, limit > 0 → return empty array
- [ ] 32.6 Run `RangeSearchTests` — verify both tests PASS (GREEN)
- [ ] 32.7 Run full test suite — verify 0 new regressions
- [ ] 32.8 Commit: `feat: add range search with distance threshold`

---

## Task 33: Runtime Metric Selection

- [ ] 33.1 Create `Tests/MetalANNSTests/RuntimeMetricTests.swift` with 2 tests (RED)
- [ ] 33.2 Add `metric: Metric? = nil` parameter to `ANNSIndex.search()`
- [ ] 33.3 Add `metric: Metric? = nil` parameter to `ANNSIndex.rangeSearch()`
- [ ] 33.4 Add `metric: Metric? = nil` parameter to `ANNSIndex.batchSearch()`
- [ ] 33.5 Implement: `let searchMetric = metric ?? configuration.metric` — pass to GPU/CPU search
- [ ] 33.6 Verify: no shader changes needed (metricType already a runtime parameter)
- [ ] 33.7 Run `RuntimeMetricTests` — verify both tests PASS (GREEN)
- [ ] 33.8 Run full test suite — verify 0 new regressions
- [ ] 33.9 Commit: `feat: support runtime metric selection at query time`

---

## Task 34: Disk-Backed Index

- [ ] 34.1 Create `Tests/MetalANNSTests/DiskBackedTests.swift` with `diskBackedSearchWorks` test (RED)
- [ ] 34.2 Create `Tests/MetalANNSTests/DiskBackedTests.swift` with `diskBackedWorksWithV3` test (RED)
- [ ] 34.3 Create `Sources/MetalANNSCore/DiskBackedVectorBuffer.swift` — mmap-backed, read-only, LRU cache
- [ ] 34.4 Implement `DiskBackedVectorBuffer.vector(at:)` with LRU cache (read from mmap pointer + offset)
- [ ] 34.5 Implement `DiskBackedVectorBuffer.insert()` / `batchInsert()` → throw read-only error
- [ ] 34.6 Conform `DiskBackedVectorBuffer` to `VectorStorage`
- [ ] 34.7 Create `DiskBackedIndexLoader` — parse header (v1/v2/v3), compute section offsets
- [ ] 34.8 Implement: create `DiskBackedVectorBuffer` pointing at vector section in mmap
- [ ] 34.9 Implement: copy adjacency + distance sections into regular `GraphBuffer` (graph stays in RAM)
- [ ] 34.10 Implement: parse trailer for IDMap + entryPoint from mmap
- [ ] 34.11 Keep mmap region alive via `mmapLifetime`
- [ ] 34.12 Add `loadDiskBacked(from:)` static method to `ANNSIndex.swift`
- [ ] 34.13 Set `isReadOnlyLoadedIndex = true` for disk-backed loaded index
- [ ] 34.14 Run `DiskBackedTests` — verify both tests PASS (GREEN)
- [ ] 34.15 Run full test suite — verify 0 new regressions
- [ ] 34.16 Commit: `feat: add disk-backed index for memory-constrained devices`

---

## Task 35: Sharded Indices (IVF-Style)

- [ ] 35.1 DECISION: Persistence — no persistence v1 (A) vs full save/load (B). Recommended: A
- [ ] 35.2 DECISION: Mutation — build-only (A) vs delegate insert to nearest shard (B). Recommended: A
- [ ] 35.3 Create `Tests/MetalANNSTests/ShardedIndexTests.swift` with `shardedSearchRecall` test (RED)
- [ ] 35.4 Create `Tests/MetalANNSTests/ShardedIndexTests.swift` with `shardedDistribution` test (RED)
- [ ] 35.5 Create `Sources/MetalANNSCore/KMeans.swift` — Lloyd's algorithm with k-means++ init
- [ ] 35.6 Implement `KMeans.cluster()`: k-means++ init → assign → recompute → iterate
- [ ] 35.7 Implement k-means++ initialization: first centroid random, rest proportional to distance
- [ ] 35.8 Create `RandomNumberGenerator64` seeded RNG for reproducibility
- [ ] 35.9 Create `Sources/MetalANNS/ShardedIndex.swift` — actor with `numShards`, `nprobe`, `configuration`
- [ ] 35.10 Implement `ShardedIndex.build()`: k-means → group vectors → build per-shard ANNSIndex
- [ ] 35.11 Handle empty shards: skip shards with zero assigned vectors
- [ ] 35.12 Implement `ShardedIndex.search()`: find top-nprobe centroids → search each → merge top-k
- [ ] 35.13 Support `filter` and `metric` parameters (pass through to per-shard search)
- [ ] 35.14 Implement `ShardedIndex.count` — sum across all shards
- [ ] 35.15 Run `ShardedIndexTests` — verify both tests PASS (GREEN)
- [ ] 35.16 Run full test suite — verify 0 new regressions
- [ ] 35.17 Commit: `feat: add sharded index with IVF-style partitioning`

---

## Review Checklist

### R1–R8: Task 31 (Filtered Search)
- [ ] R1: `SearchFilter` enum is `Sendable` (all associated values are Sendable)
- [ ] R2: `MetadataStore` is `Sendable` and `Codable`
- [ ] R3: `MetadataStore.matches()` handles all 7 filter cases: equals, greaterThan, lessThan, in, and, or, not
- [ ] R4: `.or` uses `contains { ... }` (not `allSatisfy`), `.and` uses `allSatisfy`
- [ ] R5: Over-fetch multiplier applied only when filter is non-nil (no perf regression for unfiltered search)
- [ ] R6: `PersistedMetadata.metadataStore` is optional (`MetadataStore?`) for backward compatibility
- [ ] R7: Existing save/load tests still pass (backward compat with pre-metadata `.meta.json`)
- [ ] R8: Metadata filter applied AFTER soft deletion filter (correct order)

### R9–R12: Task 32 (Range Search)
- [ ] R9: Range search returns empty array (not error) for maxDistance <= 0 or limit <= 0
- [ ] R10: All returned results have `score <= maxDistance`
- [ ] R11: Results are sorted by distance ascending
- [ ] R12: Range search integrates with metadata filter (reuses Task 31 infrastructure)

### R13–R16: Task 33 (Runtime Metric)
- [ ] R13: `metric: nil` defaults to `configuration.metric` (backward compatible)
- [ ] R14: No changes to Metal shaders (metricType already dispatched at runtime)
- [ ] R15: `batchSearch` passes metric through to individual `search()` calls
- [ ] R16: Results with override metric are sorted by the override metric's distance

### R17–R24: Task 34 (Disk-Backed Index)
- [ ] R17: `DiskBackedVectorBuffer` conforms to `VectorStorage` — all required methods implemented
- [ ] R18: `insert()` and `batchInsert()` throw clear "read-only" error
- [ ] R19: LRU cache evicts oldest entry when capacity exceeded
- [ ] R20: `DiskBackedIndexLoader` handles v1, v2, AND v3 format files (test at least v2 and v3)
- [ ] R21: For v3: section offsets account for page padding (same math as MmapIndexLoader)
- [ ] R22: For v1: no storageType field in header (defaults to float32)
- [ ] R23: `mmapLifetime` keeps MmapRegion alive — no use-after-free
- [ ] R24: `isReadOnlyLoadedIndex = true` prevents insert/delete/compact on disk-backed index

### R25–R32: Task 35 (Sharded Index)
- [ ] R25: `KMeans.cluster()` validates inputs: non-empty vectors, k > 0, k <= vectors.count
- [ ] R26: K-means++ initialization avoids duplicate centroids (probability-based selection)
- [ ] R27: Empty clusters handled: keep previous centroid, don't divide by zero
- [ ] R28: `ShardedIndex.build()` handles edge case: numShards > vectors.count → effectiveShards = vectors.count
- [ ] R29: `ShardedIndex.search()` probes `min(nprobe, shards.count)` shards (not more than exist)
- [ ] R30: Results from multiple shards merged correctly: sorted by score ascending, top-k returned
- [ ] R31: `ShardedIndex` is an actor — thread-safe for concurrent search
- [ ] R32: `SIMDDistance.distance()` used for centroid distance computation (not a custom function)

### R33–R40: Cross-Task Verification
- [ ] R33: All 89+ tests pass (except known MmapTests baseline failure)
- [ ] R34: No regressions in existing search tests (ANNSIndexTests, FullGPUSearchTests)
- [ ] R35: No regressions in persistence tests (PersistenceTests)
- [ ] R36: No regressions in batch search tests
- [ ] R37: search() signature backward compatible: `search(query:k:)` still works (filter/metric default to nil)
- [ ] R38: batchSearch() signature backward compatible: `batchSearch(queries:k:)` still works
- [ ] R39: `@unchecked Sendable` only on classes with MTLBuffer/mmap pointers
- [ ] R40: No force-unwraps except in test assertions

---

## Status

**STATUS**: pending
**Commits**: 33 (baseline, after Phase 10)
**Tests**: 78+ (baseline, after Phase 10)
**Decisions**:
- 35.1: _pending_ (recommended: Option A — no persistence v1)
- 35.2: _pending_ (recommended: Option A — build-only)
