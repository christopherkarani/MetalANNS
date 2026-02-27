# Phase 17: Benchmarking Suite

> **For Claude:** This is an **implementation prompt** for Phase 17 of MetalANNS v3. Execute via TDD (RED→GREEN→commit). Dispatch to subagent; orchestrator reviews using the R1-R13 checklist in `tasks/todo.md`.

**Goal:** Transform the synthetic-only `MetalANNSBenchmarks` executable into a production benchmarking harness capable of loading real-world datasets, sweeping configuration parameters, computing QPS and Pareto frontiers, exporting CSV results, and comparing `ANNSIndex` against `IVFPQIndex` (Phase 16).

---

## Current Baseline (Phase 16 Complete)

**Existing files:**
- `Sources/MetalANNSBenchmarks/BenchmarkRunner.swift` — single-config synthetic run, String IDs, brute-force recall, latency percentiles only
- `Sources/MetalANNSBenchmarks/main.swift` — calls `BenchmarkRunner.run(config:)` and prints 7 fixed lines

**Gaps this phase fills:**
- No real dataset support (no file I/O)
- No QPS measurement (only per-query latency)
- No configuration sweeps
- No Pareto frontier analysis
- No CSV export
- No IVFPQ comparison
- No unit tests for benchmark infrastructure

---

## Architecture

```
BenchmarkDataset             BenchmarkRunner (extended)
    .annbin file I/O  ──────►  sweep(configs:)  ──────►  BenchmarkReport
    ground truth               QPS + latency             table / CSV / Pareto
    train/test split           recall@1/10/100

scripts/convert_hdf5.py      IVFPQBenchmark (new)
    HDF5 → .annbin             run IVFPQIndex alongside ANNSIndex
    (Python, no Swift dep)     for direct recall vs QPS comparison
```

**Key design decisions:**
- `.annbin` stores `UInt32` node indices (benchmark standard) — BenchmarkRunner adapts to ANNSIndex's `String` IDs via `"v_\(id)"`
- QPS = `queryCount / totalSearchTimeSeconds` (batch throughput, not per-query)
- Pareto frontier: set of (recall, QPS) points where no point dominates both dimensions
- `BenchmarkDatasetTests.swift` goes in `Tests/MetalANNSTests/` (not the executable target)
- The executable `MetalANNSBenchmarks` has no `@Test` annotations

---

## `.annbin` Binary Format

```
Header (40 bytes, all little-endian):
  [0..3]   magic:          "ANNB"  (4 ascii bytes)
  [4..7]   version:        UInt32  (= 1)
  [8..11]  trainCount:     UInt32
  [12..15] testCount:      UInt32
  [16..19] dimension:      UInt32
  [20..23] neighborsCount: UInt32  (ground truth k per query, typically 100)
  [24..27] metricRaw:      UInt32  (0=cosine, 1=l2, 2=innerProduct)
  [28..39] reserved:       3×UInt32 (zeroed)

Body (after header, binary floats and UInt32):
  train vectors:    trainCount × dimension × Float32
  test vectors:     testCount  × dimension × Float32
  ground truth:     testCount  × neighborsCount × UInt32  (sorted by distance ASC)
```

---

## System Context

### Existing Infrastructure (Do NOT change)

**BenchmarkRunner.Config** (existing fields — keep all):
```swift
struct Config {
    var vectorCount: Int = 1000
    var dim: Int = 128
    var degree: Int = 32
    var queryCount: Int = 100
    var k: Int = 10
    var efSearch: Int = 64
    var metric: Metric = .cosine
}
```

**BenchmarkRunner.Results** (existing fields — keep all):
```swift
struct Results {
    var buildTimeMs: Double
    var queryLatencyP50Ms: Double
    var queryLatencyP95Ms: Double
    var queryLatencyP99Ms: Double
    var recallAt1: Double
    var recallAt10: Double
    var recallAt100: Double
}
```

**BenchmarkRunner.run(config:)** — keep as-is. Phase 17 adds `sweep(configs:)` alongside it.

### What ANNSIndex.build expects

```swift
try await index.build(vectors: [[Float]], ids: [String])
```

The benchmark adapter converts UInt32 ground-truth IDs to String IDs with `"v_\(id)"`.

---

## Tasks

### Task 1: BenchmarkDataset — .annbin File Format

**Acceptance**: `BenchmarkDatasetTests` passes (5 tests). First git commit.

**Checklist:**

- [ ] 1.1 — Create `Tests/MetalANNSTests/BenchmarkDatasetTests.swift` with tests:
  - `writeAndReadRoundTrip` — write .annbin to temp path, read back, verify all fields identical
  - `trainVectorsPreserved` — verify train vectors match original (floating-point exact match)
  - `testVectorsPreserved` — verify test vectors match original
  - `groundTruthPreserved` — verify ground truth UInt32 indices match original
  - `metricRoundTrip` — verify all three `Metric` values survive encode/decode

- [ ] 1.2 — **RED**: Tests fail (BenchmarkDataset not defined)

- [ ] 1.3 — Create `Sources/MetalANNSBenchmarks/BenchmarkDataset.swift`:
  ```swift
  public struct BenchmarkDataset: Sendable {
      public let trainVectors: [[Float]]   // training set used to build index
      public let testVectors: [[Float]]    // query set
      public let groundTruth: [[UInt32]]   // groundTruth[i] = sorted neighbor IDs for testVectors[i]
      public let dimension: Int
      public let metric: Metric
      public let neighborsCount: Int       // k in ground truth (typically 100)

      // MARK: — Synthetic convenience

      /// Generate synthetic dataset (no file I/O) for quick local validation
      public static func synthetic(
          trainCount: Int,
          testCount: Int,
          dimension: Int,
          k: Int = 100,
          metric: Metric = .cosine,
          seed: Int = 42
      ) -> BenchmarkDataset

      // MARK: — File I/O

      /// Save dataset to .annbin format
      public func save(to path: String) throws

      /// Load from .annbin format
      public static func load(from path: String) throws -> BenchmarkDataset
  }
  ```
  - `save()`: Write 40-byte header + train floats + test floats + ground truth UInt32s, all little-endian
  - `load()`: Validate magic "ANNB", check version = 1, parse header, read body
  - `synthetic()`: Generate vectors using deterministic seeded noise (same formula as existing `makeVectors`), compute brute-force ground truth UInt32 IDs
  - Throw descriptive errors for corrupt magic, version mismatch, truncated file

- [ ] 1.4 — **GREEN**: All 5 tests pass

- [ ] 1.5 — **GIT**: `git commit -m "feat: add BenchmarkDataset with .annbin binary format"`

---

### Task 2: BenchmarkReport — Table and CSV Output

**Acceptance**: `BenchmarkReportTests` passes. Second git commit.

**Checklist:**

- [ ] 2.1 — Create `Tests/MetalANNSTests/BenchmarkReportTests.swift` with tests:
  - `tableOutput` — generate table from 3 result rows, verify header and data lines present
  - `csvOutput` — generate CSV, verify header row + correct number of data rows
  - `paretoFrontier` — given 5 (recall, QPS) points with 2 dominated, verify frontier has exactly 3

- [ ] 2.2 — **RED**: Tests fail (BenchmarkReport not defined)

- [ ] 2.3 — Create `Sources/MetalANNSBenchmarks/BenchmarkReport.swift`:
  ```swift
  public struct BenchmarkReport: Sendable {
      public struct Row: Sendable {
          public var label: String         // e.g. "efSearch=64"
          public var recallAt10: Double
          public var qps: Double           // queries per second (batch)
          public var buildTimeMs: Double
          public var p50Ms: Double
          public var p95Ms: Double
          public var p99Ms: Double
      }

      public var rows: [Row]
      public var datasetLabel: String

      /// Render fixed-width ASCII table (terminal output)
      public func renderTable() -> String

      /// Render CSV with header row
      public func renderCSV() -> String

      /// Save CSV to path
      public func saveCSV(to path: String) throws

      /// Return non-dominated (recall, QPS) points — Pareto frontier
      /// A point p dominates q if p.recallAt10 >= q.recallAt10 AND p.qps >= q.qps (strictly > in at least one)
      public func paretoFrontier() -> [Row]
  }
  ```
  - Table format (fixed-width columns, right-aligned numbers):
    ```
    label           recall@10    QPS    buildMs   p50ms   p95ms   p99ms
    ───────────────────────────────────────────────────────────────────
    efSearch=64       0.953    4231     142.0      1.2     2.1     3.4
    efSearch=128      0.971    2108     142.0      2.4     4.2     6.8
    ```
  - CSV header: `label,recall@10,qps,buildTimeMs,p50ms,p95ms,p99ms`

- [ ] 2.4 — **GREEN**: All 3 tests pass

- [ ] 2.5 — **GIT**: `git commit -m "feat: add BenchmarkReport with table/CSV output and Pareto frontier"`

---

### Task 3: BenchmarkRunner — Sweep and QPS

**Acceptance**: `BenchmarkRunnerSweepTests` passes. Third git commit.

**Checklist:**

- [ ] 3.1 — Create `Tests/MetalANNSTests/BenchmarkRunnerSweepTests.swift` with tests:
  - `sweepReturnsOneRowPerConfig` — sweep 3 configs, verify report has 3 rows
  - `qpsIsPositive` — all sweep rows have qps > 0
  - `recallFromDataset` — use BenchmarkDataset.synthetic(trainCount:200, testCount:50, dimension:32), verify recall@10 > 0.5

- [ ] 3.2 — **RED**: Tests fail (`sweep` not defined)

- [ ] 3.3 — Extend `Sources/MetalANNSBenchmarks/BenchmarkRunner.swift`:
  ```swift
  // Add to BenchmarkRunner.Results
  extension BenchmarkRunner.Results {
      var qps: Double  // computed: queryCount / (totalSearchTimeSeconds)
  }

  // New overload using BenchmarkDataset
  extension BenchmarkRunner {
      static func run(
          config: Config,
          dataset: BenchmarkDataset
      ) async throws -> Results

      /// Run multiple configs, return BenchmarkReport
      static func sweep(
          configs: [(label: String, config: Config)],
          dataset: BenchmarkDataset
      ) async throws -> BenchmarkReport
  }
  ```
  - `run(config:dataset:)`:
    1. Use `dataset.trainVectors` as index input (map UInt32 positions → `"v_\(i)"` String IDs)
    2. Use `dataset.testVectors` as queries
    3. For recall: compare ANNSIndex results against `dataset.groundTruth` (set intersection)
    4. QPS: time the full batch of queries, divide query count by total seconds
  - `sweep(configs:dataset:)`:
    1. Build index once per config (configs may have different efSearch/degree)
    2. Collect results, build `BenchmarkReport`
  - Keep existing `run(config:)` synthetic overload unchanged — no regressions

- [ ] 3.4 — **GREEN**: All 3 sweep tests pass

- [ ] 3.5 — **GIT**: `git commit -m "feat: extend BenchmarkRunner with sweep, QPS, and dataset-backed recall"`

---

### Task 4: Updated main.swift with CLI Arguments

**Acceptance**: `main.swift` builds cleanly and handles all three modes. Fourth git commit.

- [ ] 4.1 — Update `Sources/MetalANNSBenchmarks/main.swift` to support three run modes:
  ```
  USAGE:
    MetalANNSBenchmarks                             # synthetic single run (default)
    MetalANNSBenchmarks --sweep                     # efSearch sweep on synthetic data
    MetalANNSBenchmarks --dataset <path.annbin>     # load dataset, single run
    MetalANNSBenchmarks --dataset <path.annbin> --sweep    # load dataset, efSearch sweep
    MetalANNSBenchmarks --dataset <path.annbin> --csv-out <path.csv>  # save CSV
    MetalANNSBenchmarks --ivfpq                     # compare ANNSIndex vs IVFPQIndex (synthetic)
  ```
  - `--sweep` mode: sweep `efSearch` in [16, 32, 64, 128, 256], print table + Pareto row count
  - `--dataset path`: load .annbin, use real ground truth
  - `--csv-out path`: save report as CSV after run
  - `--ivfpq`: run IVFPQBenchmark (Task 6), print side-by-side comparison
  - Default (no args): existing single-run behavior, unchanged output format

- [ ] 4.2 — **BUILD VERIFY**: `xcodebuild build -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation` → **BUILD SUCCEEDED**

- [ ] 4.3 — **GIT**: `git commit -m "feat: update main.swift with CLI modes (sweep, dataset, csv-out, ivfpq)"`

---

### Task 5: Python HDF5-to-.annbin Converter

**Acceptance**: Script exists, is valid Python 3, has usage docs. Fifth git commit.

**Checklist:**

- [ ] 5.1 — Create `scripts/` directory
- [ ] 5.2 — Create `scripts/convert_hdf5.py`:
  ```
  USAGE:
    python3 scripts/convert_hdf5.py --input sift-128-euclidean.hdf5 --output sift-128.annbin
    python3 scripts/convert_hdf5.py --input gist-960-euclidean.hdf5 --output gist-960.annbin --metric l2

  HDF5 SCHEMA (ann-benchmarks.com format):
    /train   float32[N × D]  — training vectors
    /test    float32[Q × D]  — query vectors
    /neighbors  int32[Q × K] — ground truth indices (sorted ASC by distance)
    /distances  float32[Q × K] — corresponding distances (unused in .annbin)

  DEPENDENCIES:
    pip install h5py numpy
  ```
  - Script writes .annbin header + body as documented in Overview
  - Metric flag: `--metric cosine|l2|innerproduct` (default: inferred from filename if contains "euclidean"→l2, "angular"→cosine, else cosine)
  - Validates HDF5 schema before writing
  - Prints dataset summary on success: `Written 1M×128 train + 10K×128 test, k=100, l2 → sift-128.annbin`

- [ ] 5.3 — Verify `python3 -m py_compile scripts/convert_hdf5.py` succeeds (no syntax errors)

- [ ] 5.4 — **GIT**: `git commit -m "feat: add scripts/convert_hdf5.py for HDF5 to .annbin conversion"`

---

### Task 6: IVFPQBenchmark — Side-by-Side Comparison

**Acceptance**: `IVFPQBenchmarkTests` passes and `--ivfpq` CLI mode works. Sixth git commit.

**Checklist:**

- [ ] 6.1 — Create `Tests/MetalANNSTests/IVFPQBenchmarkTests.swift` with tests:
  - `runsBothIndexes` — verify IVFPQBenchmark returns results for both ANNSIndex and IVFPQIndex
  - `ivfpqRecallPositive` — IVFPQ recall@10 > 0 on synthetic dataset
  - `annsBuildsFaster` — ANNSIndex build time should be < IVFPQIndex train time for small datasets (expected property, not strict)

- [ ] 6.2 — **RED**: Tests fail (IVFPQBenchmark not defined)

- [ ] 6.3 — Create `Sources/MetalANNSBenchmarks/IVFPQBenchmark.swift`:
  ```swift
  public struct IVFPQBenchmark: Sendable {
      public struct ComparisonResults: Sendable {
          public var annsResults: BenchmarkReport.Row
          public var ivfpqResults: BenchmarkReport.Row
      }

      /// Run ANNSIndex and IVFPQIndex on the same dataset, return side-by-side comparison
      public static func run(
          dataset: BenchmarkDataset,
          annsConfig: BenchmarkRunner.Config,
          ivfpqConfig: IVFPQConfiguration
      ) async throws -> ComparisonResults

      /// Render comparison as two-row table
      public static func renderComparison(_ results: ComparisonResults) -> String
  }
  ```
  - For `IVFPQIndex`: train on `dataset.trainVectors`, add same vectors, search `dataset.testVectors`
  - Use `dataset.groundTruth` for recall computation (same as BenchmarkRunner)
  - IVFPQ uses UInt32 IDs directly (its native format)

- [ ] 6.4 — **GREEN**: All 3 tests pass

- [ ] 6.5 — **GIT**: `git commit -m "feat: add IVFPQBenchmark for side-by-side ANNSIndex vs IVFPQIndex comparison"`

---

### Task 7: Full Suite and Completion Signal

**Acceptance**: All tests pass, executable runs in all modes, full suite clean. Seventh commit.

**Checklist:**

- [ ] 7.1 — Run: `xcodebuild build -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation` → **BUILD SUCCEEDED**

- [ ] 7.2 — Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation`
  - Expected: `BenchmarkDatasetTests`, `BenchmarkReportTests`, `BenchmarkRunnerSweepTests`, `IVFPQBenchmarkTests` all pass
  - All Phase 13-16 tests still pass
  - Known MmapTests baseline failure allowed

- [ ] 7.3 — Run smoke-test of the executable in each mode:
  ```bash
  # Default mode
  xcodebuild run -scheme MetalANNSBenchmarks -destination 'platform=macOS'
  # Sweep mode (build first, then run binary directly)
  .build/debug/MetalANNSBenchmarks --sweep
  .build/debug/MetalANNSBenchmarks --ivfpq
  ```

- [ ] 7.4 — **VERIFY** `scripts/convert_hdf5.py` compiles: `python3 -m py_compile scripts/convert_hdf5.py`

- [ ] 7.5 — Verify git log shows exactly 7 commits with conventional messages

- [ ] 7.6 — Update Phase Complete Signal in `tasks/todo.md`

- [ ] 7.7 — **GIT**: `git commit -m "chore: phase 17 complete - benchmarking suite with sweep, dataset, and IVFPQ comparison"`

---

## Success Criteria

✅ `.annbin` round-trip: write and read back produces bit-identical data
✅ Sweep mode: 5 efSearch values → 5 rows → Pareto frontier printed
✅ Dataset mode: loads real .annbin file and computes recall against ground truth
✅ CSV export: machine-readable output for plotting
✅ IVFPQ comparison: side-by-side table in `--ivfpq` mode
✅ Python converter: valid script, compiles, documented usage
✅ No regressions: all Phase 13-16 tests pass, existing `BenchmarkRunner.run(config:)` unchanged

---

## Anti-Patterns

❌ **Don't** add `@Test` macros to `MetalANNSBenchmarks` executable target — tests go in `MetalANNSTests`
❌ **Don't** change the existing `BenchmarkRunner.run(config:)` signature — backward compatibility required
❌ **Don't** use external Swift dependencies — zero-dep constraint
❌ **Don't** compute recall using String ID intersection in dataset mode — use UInt32 set intersection against ground truth
❌ **Don't** measure QPS as `1 / p50latency` — it must be `queryCount / totalBatchTimeSeconds`
❌ **Don't** require h5py in Swift — the HDF5 conversion is Python-only, accessed offline
❌ **Don't** hardcode `efSearch` sweep values in more than one place — define as `let efSearchSweep = [16, 32, 64, 128, 256]`
❌ **Don't** write test data to `~` or project root — use `FileManager.default.temporaryDirectory`

---

## Files Summary

| File | Target | Purpose | New/Modified |
|------|--------|---------|--------------|
| `BenchmarkDataset.swift` | MetalANNSBenchmarks | .annbin I/O + synthetic generation | **New** |
| `BenchmarkReport.swift` | MetalANNSBenchmarks | Table, CSV, Pareto | **New** |
| `BenchmarkRunner.swift` | MetalANNSBenchmarks | Add sweep + QPS overloads | **Modified** |
| `IVFPQBenchmark.swift` | MetalANNSBenchmarks | ANNSIndex vs IVFPQIndex | **New** |
| `main.swift` | MetalANNSBenchmarks | CLI arg parsing + all modes | **Modified** |
| `scripts/convert_hdf5.py` | (script) | HDF5 → .annbin | **New** |
| `BenchmarkDatasetTests.swift` | MetalANNSTests | .annbin round-trip | **New** |
| `BenchmarkReportTests.swift` | MetalANNSTests | Table/CSV/Pareto | **New** |
| `BenchmarkRunnerSweepTests.swift` | MetalANNSTests | Sweep + QPS | **New** |
| `IVFPQBenchmarkTests.swift` | MetalANNSTests | Comparison results | **New** |

**Total new code: ~1200 lines (including tests and Python script)**

---

## Commits Expected

1. BenchmarkDataset + .annbin format
2. BenchmarkReport + table/CSV/Pareto
3. BenchmarkRunner sweep + QPS
4. main.swift CLI modes
5. scripts/convert_hdf5.py
6. IVFPQBenchmark comparison
7. Phase complete signal
