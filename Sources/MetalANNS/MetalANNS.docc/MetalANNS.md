# ``MetalANNS``

GPU-native approximate nearest neighbor search for Apple Silicon.

## Overview

MetalANNS is a production-grade vector search library built from the ground up for Metal's compute architecture. It uses CAGRA-style NN-Descent graphs — fully GPU-parallelizable, unlike sequential HNSW — to deliver high-recall approximate nearest neighbor search on-device.

The library provides a type-safe Swift API with actor-based concurrency, mutable indexes, multiple persistence modes, and a rich query filtering DSL. A dual-backend architecture uses Metal compute shaders on Apple Silicon and falls back to Accelerate (vDSP/BLAS) on simulators and CI.

### Key Capabilities

- **GPU-accelerated graph construction and search** via Metal compute shaders
- **Mutable indexes** — insert, delete, batch update, and compact at runtime
- **Filtered search** with a type-safe result builder DSL supporting boolean logic
- **Multiple persistence modes** — binary, zero-copy mmap, and disk-backed streaming
- **FP16, binary, and product quantization** codecs for memory/speed trade-offs
- **Swift 6 concurrency safe** — `Sendable` enforced at compile time

## Topics

### Essentials

- <doc:GettingStarted>
- ``VectorIndex``
- ``IndexConfiguration``
- ``VectorRecord``

### Search

- ``VectorNeighbor``
- ``QueryFilter``
- ``QueryFilterBuilder``
- ``Field``
- ``Metric``

### Persistence

- ``ReadOnlyLoadMode``
- ``VectorIndexState``

### Advanced Index Types

- ``Advanced``

### Metrics and Diagnostics

- ``IndexMetrics``
- ``MetricsSnapshot``

### Errors

- ``ANNSError``
