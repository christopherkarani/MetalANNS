# Phase 10: Scalability Primitives — Todo

**Branch**: `Phase-10`
**Baseline**: 32 commits, 65 tests, all passing
**Target**: 35 commits, 71+ tests, all passing

---

## Task 28: Batch Insert

- [ ] 28.1 DECISION: Sequential forward + batched reverse (Option A) vs fully parallel (Option B)
- [ ] 28.2 Create `Tests/MetalANNSTests/BatchInsertTests.swift` with `batchInsertRecall` test (RED)
- [ ] 28.3 Create `Tests/MetalANNSTests/BatchInsertTests.swift` with `batchInsertMatchesSequential` test (RED)
- [ ] 28.4 Create `Sources/MetalANNSCore/BatchIncrementalBuilder.swift` with `batchInsert()` static method
- [ ] 28.5 Implement forward pass: sequential beam search for each new vector
- [ ] 28.6 Implement reverse pass: batch update existing nodes' neighbor lists
- [ ] 28.7 Add `batchInsert(_:ids:)` method to `ANNSIndex.swift`
- [ ] 28.8 Validate: duplicate ID detection (within batch + against existing)
- [ ] 28.9 Validate: capacity overflow check before inserting
- [ ] 28.10 Run `BatchInsertTests` — verify both tests PASS (GREEN)
- [ ] 28.11 Run full test suite — verify 0 regressions
- [ ] 28.12 Commit: `feat: add batch insert for efficient bulk addition`

---

## Task 29: Hard Deletion + Compaction

- [ ] 29.1 DECISION: Full NNDescent rebuild (Option A) vs remap existing graph (Option B)
- [ ] 29.2 Add `allDeletedIDs: Set<UInt32>` public accessor to `SoftDeletion.swift`
- [ ] 29.3 Create `Tests/MetalANNSTests/CompactionTests.swift` with `compactReducesMemory` test (RED)
- [ ] 29.4 Create `Tests/MetalANNSTests/CompactionTests.swift` with `compactMaintainsRecall` test (RED)
- [ ] 29.5 Create `Sources/MetalANNSCore/IndexCompactor.swift` with `CompactionResult` struct
- [ ] 29.6 Implement compaction: enumerate surviving IDs, build old→new mapping
- [ ] 29.7 Implement compaction: create new VectorStorage, copy surviving vectors
- [ ] 29.8 Implement compaction: rebuild IDMap with contiguous internal IDs
- [ ] 29.9 Implement compaction: rebuild graph via NNDescentGPU.build() + GraphPruner.prune()
- [ ] 29.10 Implement compaction: CPU fallback path via NNDescentCPU if no MetalContext
- [ ] 29.11 Add `compact()` method to `ANNSIndex.swift`
- [ ] 29.12 Guard: early return if `softDeletion.deletedCount == 0`
- [ ] 29.13 Run `CompactionTests` — verify both tests PASS (GREEN)
- [ ] 29.14 Run full test suite — verify 0 regressions
- [ ] 29.15 Commit: `feat: add hard deletion via index compaction`

---

## Task 30: Memory-Mapped I/O

- [ ] 30.1 DECISION: Wrapper types (Option A) vs copy into existing types (Option B)
- [ ] 30.2 DECISION: New saveMmap method v3 (Option A) vs always page-align (Option B)
- [ ] 30.3 Create `Tests/MetalANNSTests/MmapTests.swift` with `mmapProducesSameResults` test (RED)
- [ ] 30.4 Create `Tests/MetalANNSTests/MmapTests.swift` with `mmapRoundtrip` test (RED)
- [ ] 30.5 Add `saveMmapCompatible()` to `IndexSerializer.swift` — version 3, page-aligned sections
- [ ] 30.6 Update `IndexSerializer.load()` to handle version 3 format (skip padding between sections)
- [ ] 30.7 Create `Sources/MetalANNSCore/MmapIndexLoader.swift` — mmap file, parse header
- [ ] 30.8 Implement: compute page-aligned offsets, create MTLBuffers via `makeBuffer(bytesNoCopy:)`
- [ ] 30.9 Implement: read-only VectorStorage wrapper for mmap'd buffer (or alternative approach per decision 30.1)
- [ ] 30.10 Handle lifetime: keep FileHandle/mmap pointer alive in result struct
- [ ] 30.11 Add `loadMmap(from:)` static method to `ANNSIndex.swift`
- [ ] 30.12 Add `saveMmapCompatible(to:)` method to `ANNSIndex.swift`
- [ ] 30.13 Run `MmapTests` — verify both tests PASS (GREEN)
- [ ] 30.14 Run full test suite — verify 0 regressions
- [ ] 30.15 Commit: `feat: add memory-mapped index loading for large indices`

---

## Review Checklist

### R1–R6: Task 28 (Batch Insert)
- [ ] R1: `BatchIncrementalBuilder.batchInsert()` processes vectors sequentially (forward) then batches reverse-update
- [ ] R2: New vectors can discover previously-inserted batch vectors as neighbors
- [ ] R3: Duplicate ID detection works both within-batch and against existing index
- [ ] R4: Capacity overflow detected before any mutations
- [ ] R5: `ANNSIndex.batchInsert` updates both `vectorStorage.count` and `graph.nodeCount`
- [ ] R6: Tests use Swift Testing framework (`@Suite`, `@Test`, `#expect`)

### R7–R14: Task 29 (Compaction)
- [ ] R7: `SoftDeletion.allDeletedIDs` returns the full set, not a copy that misses concurrent updates
- [ ] R8: `IndexCompactor` creates new buffers with `capacity = max(2, survivingCount * 2)` (room for future inserts)
- [ ] R9: Old→new ID mapping is contiguous (0, 1, 2, ..., survivingCount-1)
- [ ] R10: Graph rebuilt via `NNDescentGPU.build()` + `GraphPruner.prune()` (not just remapped)
- [ ] R11: CPU fallback path works when MetalContext is nil
- [ ] R12: `ANNSIndex.compact()` resets `softDeletion` to empty after compaction
- [ ] R13: `ANNSIndex.compact()` early-returns when `deletedCount == 0`
- [ ] R14: Post-compaction search still resolves external IDs correctly

### R15–R22: Task 30 (Mmap I/O)
- [ ] R15: Page alignment padding is `(pageSize - (cursor % pageSize)) % pageSize` (correct modular arithmetic)
- [ ] R16: `makeBuffer(bytesNoCopy:)` uses `.storageModeShared`
- [ ] R17: File handle / mmap pointer kept alive for entire lifetime of the loaded index
- [ ] R18: `saveMmapCompatible` writes version 3 header
- [ ] R19: `IndexSerializer.load()` correctly handles version 3 (skips page padding)
- [ ] R20: Read-only mmap buffers: `insert()` throws appropriate error
- [ ] R21: `loadMmap` result can be searched immediately after loading
- [ ] R22: Normal `save()`/`load()` still work unchanged (v2 format preserved)

### R23–R28: Cross-Task Verification
- [ ] R23: All 71+ tests pass: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'`
- [ ] R24: No regressions in Float16 tests
- [ ] R25: No regressions in persistence tests
- [ ] R26: No regressions in GPU search tests
- [ ] R27: `@unchecked Sendable` only on classes with MTLBuffer (not new value types)
- [ ] R28: No force-unwraps except in test assertions

---

## Status

**STATUS**: pending
**Commits**: 32 (baseline)
**Tests**: 65 (baseline)
**Decisions**:
- 28.1: _pending_
- 29.1: _pending_
- 30.1: _pending_
- 30.2: _pending_
