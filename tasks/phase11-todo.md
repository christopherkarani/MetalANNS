# Phase 11: Advanced Search — Todo

**Branch**: `Phase-11` (create from Phase-10 completion)
**Baseline**: 35 commits, 71+ tests, all passing (after Phase 10)
**Target**: 38 commits, 78+ tests, all passing

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
- [ ] 31.14 Update `save()`: include metadataStore in `.meta.json`
- [ ] 31.15 Update `load()`: read metadataStore from `.meta.json` (default empty if nil)
- [ ] 31.16 Run `FilteredSearchTests` — verify all 3 tests PASS (GREEN)
- [ ] 31.17 Run full test suite — verify 0 regressions
- [ ] 31.18 Commit: `feat: add filtered search with metadata predicates`

---

## Task 32: Range Search

- [ ] 32.1 Create `Tests/MetalANNSTests/RangeSearchTests.swift` with 2 tests (RED)
- [ ] 32.2 Add `rangeSearch(query:maxDistance:limit:filter:)` to `ANNSIndex`
- [ ] 32.3 Implement: use large ef to explore broadly, filter by `score <= maxDistance`
- [ ] 32.4 Integrate metadata filter support in range search (reuse from Task 31)
- [ ] 32.5 Guard: maxDistance > 0, limit > 0
- [ ] 32.6 Run `RangeSearchTests` — verify both tests PASS (GREEN)
- [ ] 32.7 Run full test suite — verify 0 regressions
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
- [ ] 33.8 Run full test suite — verify 0 regressions
- [ ] 33.9 Commit: `feat: support runtime metric selection at query time`

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
- [ ] R16: Results with override metric are sorted by the override metric's distance (not build metric)

### R17–R22: Cross-Task Verification
- [ ] R17: All 78+ tests pass: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'`
- [ ] R18: No regressions in ANNSIndex tests (search signature backward compatible via defaults)
- [ ] R19: No regressions in persistence tests (PersistedMetadata backward compatible)
- [ ] R20: No regressions in GPU search tests
- [ ] R21: No regressions in batch search tests
- [ ] R22: `@unchecked Sendable` not added for any new types (MetadataStore, SearchFilter are value types)

---

## Status

**STATUS**: pending
**Commits**: 35 (baseline, after Phase 10)
**Tests**: 71+ (baseline, after Phase 10)
**Decisions**: None required — all design choices are specified in the prompt
