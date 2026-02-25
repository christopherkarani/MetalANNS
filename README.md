# MetalANNS

GPU-native approximate nearest neighbor search for Apple platforms, built in Swift and Metal.

## Features

- GPU-accelerated ANN search on Apple Silicon
- CAGRA-style NN-Descent graph construction
- Dual backend behavior (Metal-first with CPU algorithm fallback paths)
- Incremental single-vector insert
- Soft deletion filtering
- Binary index persistence
- Swift 6 strict concurrency with `actor`-based public API
- Zero external dependencies

## Requirements

- iOS 17+
- macOS 14+
- visionOS 1.0+
- Apple Silicon recommended for GPU acceleration
- Frameworks used: `Metal`, `Accelerate`, `Foundation`, `OSLog`

## Quick Start

```swift
import MetalANNS

let index = ANNSIndex()
try await index.build(vectors: myVectors, ids: myIDs)

let results = try await index.search(query: queryVector, k: 10)
for result in results {
    print("\(result.id): \(result.score)")
}
```

## Configuration

`ANNSIndex` accepts `IndexConfiguration`:

```swift
let config = IndexConfiguration(
    degree: 32,
    metric: .cosine,
    efConstruction: 100,
    efSearch: 64,
    maxIterations: 20,
    useFloat16: false,
    convergenceThreshold: 0.001
)

let index = ANNSIndex(configuration: config)
```

## Incremental Operations

```swift
try await index.insert(newVector, id: "doc_123")
try await index.delete(id: "doc_99")

let activeCount = await index.count
```

## Persistence

```swift
let fileURL = URL(fileURLWithPath: "/tmp/my-index.mann")

try await index.save(to: fileURL)
let loaded = try await ANNSIndex.load(from: fileURL)
```

`save` writes:

- Core binary index: `*.mann`
- Metadata sidecar: `*.mann.meta.json` (configuration + soft deletion state)

## Architecture

MetalANNS is split into two layers:

- `MetalANNSCore`: Internal data structures, Metal compute kernels, graph construction/search, serialization, and mutation primitives.
- `MetalANNS`: Public API facade exposing the `ANNSIndex` actor and public types.

## Benchmarks

Benchmark methodology and sample numbers are in [BENCHMARKS.md](BENCHMARKS.md).

## License

MIT (or project-specific license to be finalized).
