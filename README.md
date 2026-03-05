# MetalANNS

**GPU-native vector search for Apple Silicon.** Pure Swift + Metal. No C++. No cloud. No compromise.

MetalANNS brings production-grade approximate nearest neighbor search to iOS, macOS, and visionOS — running entirely on-device with GPU acceleration via Metal compute shaders.

*English | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Portugues](README.pt-BR.md) | [Espanol](README.es.md)*

## Why MetalANNS?

Most ANN libraries are C++ ports bolted onto Apple platforms. MetalANNS was designed from scratch for Metal's memory model and compute architecture.

- **CAGRA, not HNSW** — Fixed out-degree directed graphs are fully GPU-parallelizable. No sequential insert bottleneck. [2.2-27x faster construction, 33-77x faster queries](https://arxiv.org/abs/2308.15136) vs. HNSW.
- **Dual backend** — Metal shaders on Apple Silicon, Accelerate (vDSP/BLAS) fallback on simulators and CI. Same API, same results.
- **Mutable indexes** — Insert, delete, batch update, compact. Not just build-once-query-forever.
- **Multiple persistence modes** — Binary save/load, zero-copy mmap, disk-backed streaming for large indexes.
- **Type-safe filtering** — Rich query DSL with boolean logic, range queries, and set membership.
- **Swift 6 concurrency** — Actor-based thread safety with `Sendable` enforced at compile time.

## Quick Start

```swift
// Package.swift
.package(url: "https://github.com/<your-org>/MetalANNS.git", from: "0.1.0")
```

```swift
import MetalANNS

// Build an index
let index = VectorIndex<String, VectorIndexState.Unbuilt>(
    configuration: IndexConfiguration(degree: 32, metric: .cosine, efSearch: 64)
)

let ready = try await index.build(
    records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) }
)

// Search with filters
let results = try await ready.search(query: queryVector, topK: 10) {
    QueryFilter.equals(Field<String>("category"), "docs")
    QueryFilter.greaterThan(Field<Float>("score"), 0.8)
}

for hit in results {
    print("\(hit.id) -> \(hit.score)")
}
```

## Mutability + Persistence

```swift
// Live mutations
try await index.insert(newVector, id: "doc_123")
try await index.batchInsert(batchVectors, ids: batchIDs)
try await index.delete(id: "doc_99")
try await index.compact()

// Save and load
try await index.save(to: fileURL)
let loaded = try await VectorIndex<String, VectorIndexState.Ready>.load(from: fileURL)

// Zero-copy for read-heavy workloads
let mmap = try await VectorIndex<String, VectorIndexState.ReadOnly>.loadReadOnly(from: fileURL, mode: .mmap)
```

## Performance

Benchmarks from the in-repo harness (`swift run MetalANNSBenchmarks`), synthetic data:

| | Graph Index | IVFPQ |
|---|---|---|
| **Recall@10** | 1.000 | 0.406 - 0.997 |
| **Throughput** | 57-102 QPS | 19-581 QPS |
| **Trade-off** | Accuracy-first | Speed dial |

The graph index holds **perfect recall** across `efSearch` settings. IVFPQ gives a tunable speed/accuracy knob — up to 10x throughput when you can trade recall.

## Architecture

```
MetalANNS (public API)          MetalANNSCore (internals)
┌─────────────────────┐         ┌─────────────────────────────┐
│ VectorIndex<K,State> │────────▶│ NN-Descent graph build      │
│ QueryFilter DSL      │         │ Beam search (GPU + CPU)     │
│ Persistence layer    │         │ Metal shaders / Accelerate  │
│ Metadata (GRDB)      │         │ FP16 / Binary / PQ codecs   │
└─────────────────────┘         │ Binary + mmap serialization  │
                                 └─────────────────────────────┘
```

**`VectorIndex<Key, State>`** — The main API. Type-state machine: `Unbuilt` → `Ready` → `ReadOnly`.

**`Advanced.*`** — Power-user escape hatch for direct access to low-level index types:

| Type | Use Case |
|---|---|
| `Advanced.GraphIndex` | Raw CAGRA-style graph |
| `Advanced.StreamingIndex` | Continuous ingest with background merges |
| `Advanced.ShardedIndex` | Large datasets with k-means routing |
| `Advanced.IVFPQIndex` | Product quantization for speed/memory trade-off |

## Distance Metrics

`cosine` · `l2` · `innerProduct` · `hamming`

## Benchmarks

```bash
swift run MetalANNSBenchmarks                        # baseline
swift run MetalANNSBenchmarks --sweep                 # efSearch sweep
swift run MetalANNSBenchmarks --ivfpq                 # graph vs IVFPQ
swift run MetalANNSBenchmarks --dataset path/to/data  # real dataset
swift run MetalANNSBenchmarks --csv-out results.csv   # export
```

## Requirements

| Platform | Minimum |
|---|---|
| macOS | 14+ |
| iOS | 17+ |
| visionOS | 1.0+ |

Apple Silicon recommended for GPU acceleration. Falls back to Accelerate on Intel / simulators.

## License

MIT
