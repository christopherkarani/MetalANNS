# Codebase Audit Report: MetalANNS

**Date:** February 28, 2026
**Target:** MetalANNS (GPU-native Approximate Nearest Neighbor Search)

## Executive Summary

MetalANNS provides a Swift-based, GPU-accelerated Approximate Nearest Neighbor Search (ANNS) library for Apple platforms. It utilizes a hybrid architecture, aiming to combine the parallel construction benefits of a CAGRA-style NN-Descent graph on the GPU with the search efficiency of HNSW on the CPU. 

While the project features a clean public API and leverages native Apple frameworks (Metal, Accelerate), the audit revealed several critical bottlenecks regarding memory management, algorithmic scaling, and architectural consistency that hinder its ability to process large-scale datasets efficiently.

---

## 1. Architecture & Performance Bottlenecks

### Memory Inefficiency
*   **Redundant In-Memory Copies:** `StreamingIndex` permanently maintains a full copy of the entire vector corpus in system RAM as `[[Float]]`, consuming massive Swift array heap overhead. Furthermore, while `ANNSIndex` correctly relies on `VectorStorage` for permanent storage, it temporarily materializes full `[[Float]]` copies during CPU operations (like HNSW building and CPU search fallbacks). For a 1M vector dataset, this can cause transient memory spikes exceeding 1.5GB, risking OOM crashes.
*   **VectorBuffer Residency:** Despite being intended for the GPU, `VectorBuffer` uses `.storageModeShared`, meaning it resides in system memory and relies on the unified memory architecture. Because this data is natively accessible by the CPU, the current CPU/HNSW APIs that explicitly demand `[[Float]]` arrays should be refactored to read directly from `VectorStorage` to eliminate the temporary allocations mentioned above.

### Scaling Limitations
*   **GPU Search Capped:** The `FullGPUSearch` kernel (`beam_search`) relies on a small, shared-memory visited hash table capped at `MAX_VISITED = 4096`. This severely limits GPU search capabilities for larger graphs, forcing a fallback to CPU search.
*   **Serial GPU Execution:** The `beam_search` kernel explores candidates sequentially, using threads only to parallelize neighbor distance calculations. This fails to utilize the full parallel compute potential of Apple Silicon GPUs.
*   **$O(N^2)$ Skip Layer Construction:** `HNSWBuilder.buildLayers` builds skip layers using a brute-force neighbor search. For large datasets, this results in billions of unnecessary distance computations, making index construction extremely slow.

---

## 2. Technical Implementation Issues

### Unoptimized CPU Paths
*   **Manual Distance Calculations:** `NNDescentCPU.build` ignores the optimized, `vDSP`-accelerated `SIMDDistance` module and instead uses a manual, unoptimized loop for distance calculations. 
*   **Redundant Sorting:** The CPU NNDescent implementation performs redundant array sorting on every neighbor insertion within its inner loops, heavily degrading CPU construction performance.

### Metal Shader Limitations
*   **Rigid Graph Degree Requirements:** The `NNDescentGPU` construction and its associated bitonic sort kernel (`bitonic_sort_neighbors`) strictly require the graph degree to be a power of two, capped at 64. While the code explicitly catches this and throws a runtime error, this limitation is not documented at the API level (e.g., in the configuration structs), leading to runtime failures rather than compile-time safety or graceful fallback handling.
*   **Atomic Race Conditions:** In `local_join`, the pattern of comparing-and-swapping (CAS) the distance and then subsequently writing the neighbor ID can lead to transient inconsistencies between IDs and distances during concurrent GPU execution.

### Stalled Migrations
*   **Storage Migration:** A comprehensive plan exists (`2026-02-28-grdb-storage-migration.md`) to migrate JSON-based metadata and ID mapping to SQLite (GRDB) for better performance and ACID guarantees. However, this migration has not yet been executed in the codebase.

---

## 3. Strengths

*   **Clean Public API:** The `ANNSIndex` actor provides a straightforward, user-friendly, and thread-safe interface for integrating developers.
*   **Native Ecosystem Focus:** The exclusion of third-party dependencies in favor of Metal and Accelerate aligns perfectly with idiomatic Apple platform development.
*   **Strict Concurrency:** The project adheres to Swift 6 strict concurrency rules, correctly utilizing `Sendable` protocols and actor isolation to prevent data races at the API boundary.

---

## 4. Actionable Recommendations

1.  **Reduce Host-Side Materialization:** Minimize persistent `[[Float]]` storage in `StreamingIndex` and replace repeated temporary snapshots in `ANNSIndex` CPU/HNSW paths with interfaces that can read contiguous storage directly.
2.  **Overhaul GPU Search:** Rewrite `FullGPUSearch` to use a global-memory visited array with a generation counter, removing the 4096 node limit. Implement a parallel beam search that evaluates multiple candidates concurrently per threadgroup.
3.  **Optimize HNSW Building:** Replace the brute-force skip-layer construction in `HNSWBuilder` with an algorithm that traverses the existing base graph to find neighbors, drastically reducing distance computations.
4.  **Fix CPU NNDescent:** Refactor `NNDescentCPU` to utilize `SIMDDistance` for all metric calculations and optimize the neighbor candidate insertion logic to avoid constant sorting.
5.  **Execute GRDB Migration:** Implement the structured storage migration plan to improve the reliability and load times of the index metadata and ID maps.
6.  **Surface Shader Constraints Earlier:** Validate and document GPU constraints (`degree <= 64`, power-of-two requirement for bitonic sort) at configuration time, while keeping existing runtime checks as a safety net.
