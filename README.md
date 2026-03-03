# MetalANNS

GPU-native approximate nearest neighbor search for Apple platforms, built in Swift + Metal.

MetalANNS is designed for production vector search on-device and on Apple Silicon servers: mutable indexes, streaming ingest, persistence, filtering, and benchmark tooling.

## Why MetalANNS

- Metal-first ANN search path with CPU fallback for portability.
- High-recall graph search via `VectorIndex` (backed by `Advanced.GraphIndex`).
- Real update lifecycle: `build`, `insert`, `batchInsert`, `delete`, `compact`, `save`, `load`.
- Optional advanced modes in `Advanced.*` for explicit streaming, sharding, and IVFPQ.
- Multiple persistence modes: binary, mmap read-only, disk-backed read-only.
- Swift 6 concurrency-safe actor API.

## Performance Snapshot

The benchmark harness is in-repo (`swift run MetalANNSBenchmarks`). Numbers below are from recorded local harness runs with synthetic data unless dataset paths are supplied.

### Baseline (graph index)

| Metric | Value |
|---|---:|
| Build time | 21,967.6 ms |
| Query mean | 9.607 ms |
| p50 / p95 / p99 | 9.62 / 9.97 / 10.06 ms |
| Throughput | 102.97 QPS |
| Recall@1 / @10 / @100 | 1.000 / 1.000 / 1.000 |

### efSearch Sweep (graph index, queryCount=100, runs=2)

| efSearch | Recall@10 | QPS | p95 (ms) |
|---|---:|---:|---:|
| 16 | 1.000 | 60 | 36.64 |
| 32 | 1.000 | 64 | 29.75 |
| 64 | 1.000 | 57 | 33.00 |
| 128 | 1.000 | 56 | 34.53 |
| 256 | 1.000 | 44 | 38.80 |

### ANS vs IVFPQ (in-repo comparison)

| Scenario | Index | Recall@10 | QPS |
|---|---|---:|---:|
| Speed-first | Graph | 1.000 | 57 |
| Speed-first | IVFPQ | 0.406 | 581 |
| High-recall | Graph | 1.000 | 65 |
| High-recall | IVFPQ | 0.997 | 19 |

What this shows:

1. The graph index can hold perfect recall across `efSearch` settings in this synthetic workload.
2. IVFPQ gives a speed/accuracy dial: very high QPS at lower recall, or near-graph recall at lower throughput.
3. At comparable high recall in this snapshot, the graph index was faster.

## Install

```swift
// Package.swift
.package(url: "https://github.com/<your-org>/MetalANNS.git", from: "0.1.0")
```

```swift
// target dependencies
.product(name: "MetalANNS", package: "MetalANNS")
```

## Quick Start

```swift
import MetalANNS

let index = VectorIndex<String, VectorIndexState.Unbuilt>(
    configuration: IndexConfiguration(degree: 32, metric: .cosine, efSearch: 64)
)

let ready = try await index.build(
    records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) }
)

let results = try await ready.search(query: query, topK: 10) {
    QueryFilter.equals(Field<String>("category"), "docs")
    QueryFilter.greaterThan(Field<Float>("score"), 0.8)
}
for hit in results {
    print("\(hit.id) -> \(hit.score)")
}
```

## Mutability + Persistence

```swift
try await index.insert(newVector, id: "doc_123")
try await index.batchInsert(batchVectors, ids: batchIDs)
try await index.delete(id: "doc_99")
try await index.compact()

let fileURL = URL(fileURLWithPath: "/tmp/my-index.mann")
try await index.save(to: fileURL)

let loaded = try await VectorIndex<String, VectorIndexState.Unbuilt>.load(from: fileURL)
let mmapLoaded = try await VectorIndex<String, VectorIndexState.Unbuilt>.loadReadOnly(from: fileURL, mode: .mmap)
let diskLoaded = try await VectorIndex<String, VectorIndexState.Unbuilt>.loadReadOnly(from: fileURL, mode: .diskBacked)
```

## API Surface

- `VectorIndex<Key, State>`: minimal state-typed public API (`Unbuilt -> Ready -> ReadOnly`).
- `Advanced.*`: explicit power-user escape hatch for low-level index types.
  - `Advanced.GraphIndex`
  - `Advanced.StreamingIndex`
  - `Advanced.ShardedIndex`
  - `Advanced.IVFPQIndex`

Legacy top-level types were removed. Use `VectorIndex` for default usage and `Advanced.*` for low-level control.

## Power Users

```swift
import MetalANNS

let raw = Advanced.GraphIndex(configuration: .default)
try await raw.build(vectors: vectors, ids: ids)
let hits = try await raw.search(query: query, k: 10)
```

## Architecture

- `MetalANNSCore`: data structures, kernels, graph build/search, serialization.
- `MetalANNS`: public actor-based API and storage integrations.

## Benchmark Commands

```bash
# baseline
swift run MetalANNSBenchmarks

# efSearch sweep
swift run MetalANNSBenchmarks --sweep --runs 2 --query-count 100 --sweep-efsearch 16,32,64,128,256

# ANS vs IVFPQ
swift run MetalANNSBenchmarks --ivfpq --query-count 50
swift run MetalANNSBenchmarks --ivfpq --query-count 100 --ivfpq-subspaces 8 --ivfpq-centroids 256 --ivfpq-coarse-centroids 256 --ivfpq-nprobe 1 --ivfpq-iterations 10
swift run MetalANNSBenchmarks --ivfpq --query-count 100 --ivfpq-subspaces 8 --ivfpq-centroids 256 --ivfpq-coarse-centroids 512 --ivfpq-nprobe 64 --ivfpq-iterations 10
```

Export raw benchmark data:

```bash
swift run MetalANNSBenchmarks --csv-out results.csv --json-out results.json
```

## Requirements

- iOS 17+
- macOS 14+
- visionOS 1.0+
- Apple Silicon recommended for GPU acceleration

## License

MIT
