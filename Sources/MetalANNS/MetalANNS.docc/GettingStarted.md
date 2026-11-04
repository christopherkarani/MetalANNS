# Getting Started with MetalANNS

Build a vector index, search it, and persist it to disk.

## Overview

MetalANNS uses a type-state pattern to guide you through the index lifecycle:
**Unbuilt** → **Ready** → **ReadOnly**. The compiler enforces which operations
are available at each stage — you can't search an unbuilt index or mutate a
read-only one.

## Add the Dependency

Add MetalANNS to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/MetalANNS.git", from: "0.4.0")
]
```

Then add it to your target:

```swift
.target(name: "MyApp", dependencies: [
    .product(name: "MetalANNS", package: "MetalANNS")
])
```

## Build an Index

Create a ``VectorIndex`` in the `Unbuilt` state, then call ``VectorIndex/build(records:)``
to construct the graph and transition to `Ready`:

```swift
import MetalANNS

let index = VectorIndex<String, VectorIndexState.Unbuilt>(
    configuration: IndexConfiguration(degree: 32, metric: .cosine, efSearch: 64)
)

let ready = try await index.build(
    records: zip(ids, vectors).map { VectorRecord(id: $0.0, vector: $0.1) }
)
```

MetalANNS automatically selects a GPU (Metal) or CPU (Accelerate) backend
based on hardware availability.

Any non-zero count works. With no more vectors than `degree`, the exact
neighbour graph is built directly and fills in as you insert. Vectors with
NaN or infinite components are rejected with ``ANNSError/nonFiniteVector(index:)``.
Cancelling the calling task stops construction with `CancellationError`.

If you already hold the vectors in one row-major buffer, build from it
directly. The buffer is copied once, into the index's storage, and
`[[Float]]` is never materialized:

```swift
let ready = try await index.build(flatVectors: flat, dimension: 384, ids: ids)
```

## Search with Filters

Use the result builder DSL to compose filters at search time:

```swift
let results = try await ready.search(query: queryVector, topK: 10) {
    QueryFilter.equals(Field<String>("category"), "docs")
    QueryFilter.greaterThan(Field<Float>("score"), 0.8)
}

for hit in results {
    print("\(hit.id) — distance: \(hit.distance), similarity: \(hit.similarity)")
}
```

## Read Scores

Results are ordered nearest first. Each ``VectorNeighbor`` carries the
`distance` the index ranks by (lower is closer), the `metric` it was computed
with, and a `similarity` (higher is closer):

| Metric | `distance` | `similarity` |
|---|---|---|
| `.cosine` | `1 − cos θ`, in `[0, 2]` | `cos θ`, in `[−1, 1]` |
| `.innerProduct` | `−(a · b)` | `a · b` |
| `.l2` | squared Euclidean `‖a − b‖²` | `−‖a − b‖²` |
| `.hamming` | differing bits | `−(differing bits)` |

Cosine is computed from the stored vectors as given. The exact scan and the
graph search use the same formula, so they agree to float rounding for vectors
of any norm. `rangeSearch(maxDistance:)` takes a `distance`.

## Mutate the Index

A `Ready` index supports live mutations:

```swift
try await ready.insert(VectorRecord(id: "doc_new", vector: newVector))
try await ready.delete(id: "doc_old")
try await ready.insert(VectorRecord(id: "doc_old", vector: updatedVector)) // re-add after delete
try await ready.compact()
```

Storage grows as you insert; there is no fixed capacity. A deleted id can be
inserted again and takes its old slot back. `compact()` rebuilds the graph
without the remaining deleted vectors.

## Concurrency

``VectorIndex`` is `Sendable` and wraps an actor, so you can share it across
tasks. Searches run concurrently with each other. `insert`, `batchInsert`,
`compact` and `build` are exclusive: they wait for in-flight searches, and
searches issued while one runs wait for it. A search never sees a half-applied
mutation, and an insert issued during `compact()` is not lost.

## Persist and Load

Save the index to disk and reload it later. Choose the load mode that fits
your access pattern:

```swift
// Save
let url = URL(fileURLWithPath: "/tmp/my-index.mann")
try await ready.save(to: url)

// Full load (mutable)
let loaded = try await VectorIndex<String, VectorIndexState.Unbuilt>.load(from: url)

// Zero-copy mmap (read-only, instant load)
let mmap = try await VectorIndex<String, VectorIndexState.Unbuilt>.loadReadOnly(
    from: url, mode: .mmap
)

// Disk-backed streaming (read-only, low memory)
let disk = try await VectorIndex<String, VectorIndexState.Unbuilt>.loadReadOnly(
    from: url, mode: .diskBacked
)
```

## Power-User Access

For fine-grained control, use the ``Advanced`` namespace to work directly with
low-level index types:

```swift
let graph = GraphIndex(configuration: .default)
try await graph.build(vectors: vectors, ids: ids)
let hits = try await graph.search(query: query, k: 10)
```

Available types:

| Type | Use Case |
|---|---|
| ``GraphIndex`` | Raw CAGRA-style directed graph |
| ``Advanced/StreamingIndex`` | Continuous ingest with background merges |
| ``Advanced/ShardedIndex`` | Large datasets with k-means routing |
| ``Advanced/IVFPQIndex`` | Product quantization for speed/memory trade-offs |
