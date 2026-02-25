# MetalANNS Design Document

**Date:** 2026-02-25
**Status:** Approved

## Summary

MetalANNS is a Swift package implementing GPU-native Approximate Nearest Neighbor Search on Apple Silicon using Metal compute shaders. It uses a CAGRA-style fixed out-degree directed graph built via NN-Descent, which is 2.2-27x faster to construct and 33-77x faster to query than HNSW on GPU (ICDE 2024).

## Architecture Decision: CAGRA over HNSW

HNSW's sequential insertion creates data dependencies that prevent GPU parallelism. CAGRA's flat directed graph with fixed out-degree `d` enables:
- Uniform memory layout for coalesced GPU access
- No hierarchy/branching between layers
- Fully parallelizable construction via NN-Descent

## Package Structure

Three targets:
- **MetalANNSCore** — Metal shaders, buffer management, compute backends
- **MetalANNS** — Public async Swift API (zero Metal knowledge required)
- **MetalANNSBenchmarks** — Benchmark harness (separate executable)

Platforms: iOS 17+, macOS 14+, visionOS 1.0+

## Key Design Addition: Backend Protocol Abstraction

```swift
protocol ComputeBackend: Sendable {
    func computeDistances(query: [Float], corpus: VectorBuffer, metric: Metric) async -> [Float]
    func randomInitGraph(nodeCount: Int, degree: Int, seed: UInt32) async
    func localJoin(updateCounter: MTLBuffer) async -> Int
    func beamSearch(query: [Float], k: Int, ef: Int) async -> [(UInt32, Float)]
}
```

- **MetalBackend** — GPU compute via Metal shaders
- **AccelerateBackend** — CPU fallback via vDSP/BLAS (simulator, testing, CI)
- Backend selected at init based on GPU availability
- Tests validate both backends produce equivalent results within FP tolerance

## Algorithm: NN-Descent Construction

1. Random initialization — assign each node `d` random neighbors
2. Reverse edge computation — for each edge u->v, add v->u to reverse list
3. Local join — for each node, compute distances between forward x reverse neighbor pairs, keep best `d`
4. Repeat until convergence (~10-20 iterations)
5. Sort each node's neighbor list by distance ascending

## Data Structures (GPU-Resident)

- **GraphBuffer** — flat `nodeCount x degree` adjacency matrix (UInt32) + companion distance matrix (Float32)
- **VectorBuffer** — flat `nodeCount x dim` vector storage (Float32 or Float16)
- **MetadataBuffer** — entryPointID, nodeCount, degree, dim, iterationCount

## Search: Greedy Beam Search

- Maintain candidate priority queue of size `ef_search` in threadgroup shared memory
- Visited bitset (generation counter for nodeCount > 100k)
- One threadgroup per query for batch search

## Public API

```swift
public actor ANNSIndex {
    public init(configuration: IndexConfiguration = .default)
    public func build(vectors: [[Float]], ids: [String]) async throws
    public func insert(_ vector: [Float], id: String) async throws
    public func delete(id: String) throws
    public func search(query: [Float], k: Int) async throws -> [SearchResult]
    public func batchSearch(queries: [[Float]], k: Int) async throws -> [[SearchResult]]
    public func save(to url: URL) throws
    public static func load(from url: URL) throws -> ANNSIndex
    public var count: Int { get }
}
```

## Memory Footprint (100k vectors, dim=384, degree=32, FP32)

- Vector buffer: 153.6 MB
- Adjacency + distance buffers: 25.6 MB
- Total at rest: 179.2 MB
- Peak during construction: 204.8 MB
- With FP16: 102.4 MB total

## Zero Dependencies

Pure Apple frameworks only: Metal, MetalPerformanceShaders, Accelerate, Foundation, OSLog.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Disconnected graph from NN-Descent | Bridge edges between components + extra iterations |
| Atomic contention during local join | Partition nodes into non-overlapping groups |
| Simulator has no GPU | AccelerateBackend CPU fallback (same API) |
| MTLBuffer 256MB limit on older devices | Chunked buffer strategy with transparent sharding |
| Recall degrades with incremental inserts | Track insertions, auto-rebuild at 10% threshold |

## Source Spec

Full implementation spec: `/Users/chriskarani/Desktop/MetalANNS_Implementation_Plan.md`
