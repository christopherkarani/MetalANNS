# MetalANNS Code Review Report

**Date:** 2026-02-28  
**Type:** Deep audit + remediation status  
**Current Status:** Remediation implemented for reported critical/high issues

## Summary

This report consolidates the deep-audit findings and the subsequent fix pass.

- Original deep-audit score: **72/100**.
- Reported critical/high issues were addressed in code.
- Remaining full-suite failures are environment-bound Metal library load failures, not new logic regressions.

## Validation Snapshot

### Targeted regression runs (post-fix)

- `swift test --filter "(ANNSIndexTests|BatchInsertTests|IncrementalTests|GraphRepairTests|MetadataTests)"` → pass
- `swift test --filter "(PersistenceTests|MmapTests|DiskBackedTests|StreamingIndexPersistenceTests|IVFPQPersistenceTests)"` → pass
- `swift test --filter "(ANNSIndexTests|FilteredSearchTests|GPUADCSearchTests|BatchInsertTests|GraphRepairTests|PersistenceTests|MmapTests|StreamingIndexPersistenceTests)"` → pass (`41 tests / 9 suites`)

### Full suite (post-fix)

- `swift test` → **208 tests**, **196 passed**, **12 failed**
- All 12 failures are in GPU/Metal suites with the same environment error:
  - `MTLLibraryErrorDomain Code=6` (`no default library was found`)

## Fixed Findings

1. `ANNSIndex` transactional safety and ID commit ordering.
   - ID assignment now commits after insert/batch mutation succeeds.
   - Pending repair IDs are preserved/requeued on repair failure.
   - Files:
     - `Sources/MetalANNS/ANNSIndex.swift`
     - `Sources/MetalANNSCore/IDMap.swift`

2. Persistence atomicity and corruption hardening.
   - Atomic replacement for index file writes.
   - Checked integer arithmetic for serialization and mmap parsing paths.
   - Load-time invariant validation (`idMap` count, `entryPoint` bounds, truncation safety).
   - Files:
     - `Sources/MetalANNSCore/IndexSerializer.swift`
     - `Sources/MetalANNSCore/MmapIndexLoader.swift`
     - `Sources/MetalANNS/StreamingIndex.swift`

3. Streaming save consistency and background merge observability.
   - Save now stages to temp directory and swaps to avoid split snapshots.
   - Metadata structural validation on load/save.
   - Background merge errors are recorded and surfaced.
   - File:
     - `Sources/MetalANNS/StreamingIndex.swift`

4. GPU NN-Descent/PQ ADC safety fixes.
   - Guarded `nodeCount >= 2` for GPU NN-Descent init/build.
   - Distance ordering logic corrected for inner-product/negative values by comparing floats directly.
   - Local join updates made per-node update path to avoid cross-node pair races.
   - PQ ADC now validates code bounds in host path and shader.
   - Files:
     - `Sources/MetalANNSCore/NNDescentGPU.swift`
     - `Sources/MetalANNSCore/Shaders/NNDescent.metal`
     - `Sources/MetalANNSCore/Shaders/NNDescentFloat16.metal`
     - `Sources/MetalANNSCore/GPUADCSearch.swift`
     - `Sources/MetalANNSCore/Shaders/PQDistance.metal`

5. Full GPU search guard behavior (no silent truncation/degradation).
   - Explicit failure for unsupported `k/ef`/visited/degree limits.
   - ANNSIndex falls back to CPU/HNSW path when GPU constraints are not met.
   - Files:
     - `Sources/MetalANNSCore/FullGPUSearch.swift`
     - `Sources/MetalANNS/ANNSIndex.swift`

6. Search correctness and API contract fixes.
   - HNSW search now validates jagged vector dimensions.
   - Greedy layer traversal chooses best neighbor per pass (stable greedy step).
   - HNSW only used when runtime metric matches build metric.
   - Build now validates minimum node count and degree invariants up front.
   - Files:
     - `Sources/MetalANNSCore/HNSWSearchCPU.swift`
     - `Sources/MetalANNS/ANNSIndex.swift`

7. Metadata filtering precision.
   - Added `Int64`-safe filter operators:
     - `greaterThanInt(column:value:)`
     - `lessThanInt(column:value:)`
   - Integrated in both index and streaming metadata evaluation paths.
   - Files:
     - `Sources/MetalANNSCore/SearchFilter.swift`
     - `Sources/MetalANNSCore/MetadataStore.swift`
     - `Sources/MetalANNS/StreamingIndex.swift`

## Added/Updated Tests

- `Tests/MetalANNSTests/ANNSIndexTests.swift`
  - Build validation for invalid node-count/degree combinations.
- `Tests/MetalANNSTests/FilteredSearchTests.swift`
  - Int64 precision filter behavior.
- `Tests/MetalANNSTests/GPUADCSearchTests.swift`
  - Out-of-range PQ code validation.
- `Tests/MetalANNSTests/NNDescentGPUTests.swift`
  - GPU random-init rejects `nodeCount < 2`.

## Residual Risks

1. Full GPU suites remain blocked in this environment due default Metal shader library availability.
2. Some performance-oriented follow-ups from the original review remain candidates for future tuning (not correctness blockers).

