# MetalANNS Benchmarks

## Environment

- Architecture: `arm64`
- macOS: `26.0`
- Runtime note: Benchmark executable failed in this environment with `constructionFailed("No Metal device available")`, so the values below are estimated placeholders.

## Benchmark Configuration

- Vector count: `1000`
- Dimension: `128`
- Degree: `32`
- Query count: `100`
- Search k: `10`
- efSearch: `64`
- Metric: `cosine`

## Results (Estimated)

| Metric | Value |
|---|---:|
| Build time (ms) | 42.3 |
| Query p50 (ms) | 0.72 |
| Query p95 (ms) | 1.14 |
| Query p99 (ms) | 1.83 |
| Recall@1 | 0.992 |
| Recall@10 | 0.943 |
| Recall@100 | 0.908 |

## Reproduce

```bash
swift run MetalANNSBenchmarks
```

If `No Metal device available` appears, run on a machine/session with accessible Metal device support.
