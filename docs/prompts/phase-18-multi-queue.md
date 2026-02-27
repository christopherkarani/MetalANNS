# Phase 18: Multi-Queue Parallelism

> **For Claude:** This is an **implementation prompt** for Phase 18 of MetalANNS v3 — the final phase. Execute via TDD (RED→GREEN→commit). Dispatch to subagent; orchestrator reviews using the R1-R13 checklist in `tasks/todo.md`.

**Goal:** Eliminate serial bottlenecks in GPU command submission and shard execution. Deliver measurable QPS improvements for batch search and multi-shard builds without breaking the existing actor-isolation model.

---

## Current Bottlenecks (Verified from Codebase)

### 1. Single MTLCommandQueue (MetalDevice.swift)
```swift
// TODAY — one queue for everything
public final class MetalContext: @unchecked Sendable {
    public let commandQueue: MTLCommandQueue  // single queue
    public func execute(_ encode:) async throws { ... }
}
```
Every concurrent `search()` call in `batchSearch` creates command buffers from the same queue. On Apple Silicon, a single queue serialises GPU work submission from CPU side, adding unnecessary round-trips.

### 2. batchSearch hardcodes concurrency=4
```swift
// ANNSIndex.batchSearch — today
let maxConcurrency = context != nil ? 4 : max(1, ProcessInfo.processInfo.activeProcessorCount)
```
4 is correct for M1, too low for M2 Pro/Max/Ultra (up to 16+ performance cores), and untested.

### 3. ShardedIndex builds and searches shards sequentially
```swift
// ShardedIndex — today
for shardIndex in 0..<effectiveShards {
    try await shard.build(...)   // sequential await — each shard must finish before next starts
}
for shardIndex in probeIndices {
    let shardResults = try await shards[shardIndex].search(...)  // sequential
}
```
For N shards this is O(N × shard_build_time). With TaskGroup it becomes O(max_shard_build_time).

---

## Architecture

```
Phase 18 changes:

MetalContext (modified)
  + CommandQueuePool actor — N queues, round-robin dispatch
  + executeOnPool() — pick queue from pool for each call

ANNSIndex.batchSearch (modified)
  - hardcoded 4
  + queuePool.count or CPU core count

ShardedIndex (modified)
  build():   sequential for → withThrowingTaskGroup (parallel shard construction)
  search():  sequential for → withThrowingTaskGroup (parallel shard queries)
```

**Key design constraints:**
- `MTLComputePipelineState` is thread-safe — shared across queues (no change to PipelineCache)
- `MTLCommandQueue` is thread-safe for `makeCommandBuffer()` from multiple threads
- The existing `MetalContext.execute()` is kept unchanged for backward compatibility — new `executeOnPool()` is additive
- `ANNSIndex` is an `actor` — concurrent searches within one `ANNSIndex` go through actor isolation; multi-queue helps the Metal layer beneath
- `ShardedIndex` builds each shard as a separate `ANNSIndex` instance — full task parallelism is safe

---

## System Context

### MetalContext today (do not break)
```swift
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue  // keep — backward compat
    public let library: MTLLibrary
    public let pipelineCache: PipelineCache

    public func execute(_ encode: (MTLCommandBuffer) throws -> Void) async throws
}
```

### ShardedIndex.build() today (sequential loop to parallelise)
```swift
for shardIndex in 0..<effectiveShards {
    guard !shardVectors[shardIndex].isEmpty else { continue }
    var shardConfiguration = configuration
    shardConfiguration.degree = min(configuration.degree, max(1, shardVectors[shardIndex].count - 1))
    let shard = ANNSIndex(configuration: shardConfiguration)
    try await shard.build(vectors: shardVectors[shardIndex], ids: shardIDs[shardIndex])
    builtShards.append(shard)
    builtCentroids.append(kmeans.centroids[shardIndex])
}
```

### ShardedIndex.search() today (sequential loop to parallelise)
```swift
for shardIndex in probeIndices {
    let shardResults = try await shards[shardIndex].search(
        query: query, k: k, filter: filter, metric: metric
    )
    mergedResults.append(contentsOf: shardResults)
}
```

### ANNSIndex.batchSearch() today (hardcoded concurrency)
```swift
let maxConcurrency = context != nil ? 4 : max(1, ProcessInfo.processInfo.activeProcessorCount)
return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
    // work-stealing pattern, maxConcurrency tasks in flight
}
```

---

## Tasks

### Task 1: CommandQueuePool Actor and Tests

**Acceptance**: `CommandQueuePoolTests` passes (4 tests). First git commit.

**Checklist:**

- [ ] 1.1 — Create `Tests/MetalANNSTests/CommandQueuePoolTests.swift` with tests:
  - `createsNQueues` — init pool with count=4, verify `pool.queues.count == 4`
  - `queuesAreDistinct` — all 4 queues are different object references
  - `nextIsRoundRobin` — call `next()` 8 times, verify first 4 == next 4 (cycle wraps)
  - `concurrentNextIsSafe` — call `next()` from 8 concurrent tasks, verify no crashes
  - All tests skip on simulator with `#if targetEnvironment(simulator) return #endif`

- [ ] 1.2 — **RED**: Tests fail (CommandQueuePool not defined)

- [ ] 1.3 — Create `Sources/MetalANNSCore/CommandQueuePool.swift`:
  ```swift
  /// A fixed-size pool of MTLCommandQueues for pipelining GPU work across concurrent searches.
  public actor CommandQueuePool: Sendable {
      public let queues: [MTLCommandQueue]   // immutable after init, safe to share
      private var nextIndex: Int = 0

      public init(device: MTLDevice, count: Int = 4) throws(ANNSError) {
          var qs: [MTLCommandQueue] = []
          qs.reserveCapacity(count)
          for _ in 0..<count {
              guard let q = device.makeCommandQueue() else {
                  throw ANNSError.deviceNotSupported
              }
              qs.append(q)
          }
          self.queues = qs
      }

      /// Round-robin queue selection. Thread-safe via actor isolation.
      public func next() -> MTLCommandQueue {
          let q = queues[nextIndex % queues.count]
          nextIndex &+= 1
          return q
      }
  }
  ```

- [ ] 1.4 — **GREEN**: All 4 tests pass on device, skip gracefully on simulator

- [ ] 1.5 — **GIT**: `git commit -m "feat: add CommandQueuePool actor for round-robin GPU queue selection"`

---

### Task 2: MetalContext Multi-Queue Integration

**Acceptance**: `MetalContextMultiQueueTests` passes. Existing tests pass. Second git commit.

**Checklist:**

- [ ] 2.1 — Create `Tests/MetalANNSTests/MetalContextMultiQueueTests.swift` with tests:
  - `poolInitialisedOnContext` — create MetalContext, verify `context.queuePool` is non-nil
  - `executeOnPoolUsesPoolQueue` — call `executeOnPool` twice concurrently, verify both complete without error
  - `legacyExecuteUnchanged` — verify original `context.execute()` still works (backward compat)
  - All tests skip on simulator

- [ ] 2.2 — **RED**: Tests fail (queuePool/executeOnPool not defined)

- [ ] 2.3 — Modify `Sources/MetalANNSCore/MetalDevice.swift`:
  - Add `public let queuePool: CommandQueuePool` property
  - In `MetalContext.init()`, after creating `commandQueue`, create the pool:
    ```swift
    self.queuePool = try CommandQueuePool(device: device, count: 4)
    ```
  - Add new method:
    ```swift
    /// Like execute(), but picks a queue from the pool for pipelining concurrent calls.
    public func executeOnPool(_ encode: (MTLCommandBuffer) throws -> Void) async throws {
        let queue = await queuePool.next()
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ANNSError.constructionFailed("Failed to create command buffer from pool queue")
        }
        try encode(commandBuffer)
        commandBuffer.commit()
        await commandBuffer.completed()
        if let error = commandBuffer.error {
            throw ANNSError.constructionFailed("Command buffer failed: \(error.localizedDescription)")
        }
    }
    ```
  - **KEEP** `execute()` unchanged — it still uses `self.commandQueue`

- [ ] 2.4 — **GREEN**: All 3 new tests pass

- [ ] 2.5 — **REGRESSION**: Existing `MetalDeviceTests` still pass — `execute()` unbroken

- [ ] 2.6 — **GIT**: `git commit -m "feat: add CommandQueuePool to MetalContext with executeOnPool() API"`

---

### Task 3: ShardedIndex Parallel Build

**Acceptance**: `ShardedIndexParallelBuildTests` passes with verified speedup. Third git commit.

**Checklist:**

- [ ] 3.1 — Create `Tests/MetalANNSTests/ShardedIndexParallelBuildTests.swift` with tests:
  - `parallelBuildMatchesSequentialResults` — build ShardedIndex with 4 shards (200 vectors each), parallel and sequential, run 20 queries, verify recall@10 is identical (within floating-point tolerance)
  - `parallelBuildCompletesWithoutError` — build 8-shard index, verify no errors and correct total vector count
  - `parallelBuildFasterThanSequential` — build 4 shards, time parallel vs sequential, log speedup factor (don't assert strictly — hardware varies)

- [ ] 3.2 — **RED**: `parallelBuildMatchesSequentialResults` fails on current sequential impl (or passes but sequential — verify in notes)

- [ ] 3.3 — Modify `Sources/MetalANNS/ShardedIndex.swift` build loop:
  ```swift
  // BEFORE:
  for shardIndex in 0..<effectiveShards {
      guard !shardVectors[shardIndex].isEmpty else { continue }
      var shardConfiguration = configuration
      shardConfiguration.degree = min(...)
      let shard = ANNSIndex(configuration: shardConfiguration)
      try await shard.build(vectors: shardVectors[shardIndex], ids: shardIDs[shardIndex])
      builtShards.append(shard)
      builtCentroids.append(kmeans.centroids[shardIndex])
  }

  // AFTER:
  var indexedShards: [(index: Int, shard: ANNSIndex)] = []
  try await withThrowingTaskGroup(of: (Int, ANNSIndex).self) { group in
      for shardIndex in 0..<effectiveShards {
          guard !shardVectors[shardIndex].isEmpty else { continue }
          let sv = shardVectors[shardIndex]
          let si = shardIDs[shardIndex]
          var shardConfig = configuration
          shardConfig.degree = min(configuration.degree, max(1, sv.count - 1))
          group.addTask {
              let shard = ANNSIndex(configuration: shardConfig)
              try await shard.build(vectors: sv, ids: si)
              return (shardIndex, shard)
          }
      }
      for try await (idx, shard) in group {
          indexedShards.append((idx, shard))
      }
  }
  // Sort by original shard index to preserve ordering
  indexedShards.sort { $0.index < $1.index }
  builtShards = indexedShards.map(\.shard)
  builtCentroids = indexedShards.map { kmeans.centroids[$0.index] }
  ```

- [ ] 3.4 — **GREEN**: All 3 tests pass

- [ ] 3.5 — **REGRESSION**: Existing `ShardedIndexTests` from Phase 12 still pass (same recall/results)

- [ ] 3.6 — **GIT**: `git commit -m "feat: parallelise ShardedIndex shard construction with TaskGroup"`

---

### Task 4: ShardedIndex Parallel Search

**Acceptance**: `ShardedIndexParallelSearchTests` passes. Fourth git commit.

**Checklist:**

- [ ] 4.1 — Create `Tests/MetalANNSTests/ShardedIndexParallelSearchTests.swift` with tests:
  - `parallelSearchMatchesSequential` — build 4-shard index, run 50 queries, parallel and sequential, verify results identical (same top-k IDs and distances within 1e-5)
  - `parallelBatchSearchCorrect` — call `batchSearch` on ShardedIndex with 100 queries, verify recall@10 > 0.6
  - `parallelSearchFasterThanSequential` — time 100 queries parallel vs sequential on 4-shard index, log speedup (no strict assert)

- [ ] 4.2 — **RED**: `parallelSearchMatchesSequential` may pass or fail depending on result ordering

- [ ] 4.3 — Modify `Sources/MetalANNS/ShardedIndex.swift` search loop:
  ```swift
  // BEFORE:
  for shardIndex in probeIndices {
      let shardResults = try await shards[shardIndex].search(query: query, k: k, filter: filter, metric: metric)
      mergedResults.append(contentsOf: shardResults)
  }

  // AFTER:
  try await withThrowingTaskGroup(of: [SearchResult].self) { group in
      for shardIndex in probeIndices {
          let shard = shards[shardIndex]
          group.addTask {
              try await shard.search(query: query, k: k, filter: filter, metric: metric)
          }
      }
      for try await results in group {
          mergedResults.append(contentsOf: results)
      }
  }
  ```
  - Merge is unordered (collected from group) — final sort by distance still applied after merge (verify this is already done)

- [ ] 4.4 — **GREEN**: All 3 tests pass

- [ ] 4.5 — **REGRESSION**: Existing `ShardedIndexTests` still pass

- [ ] 4.6 — **GIT**: `git commit -m "feat: parallelise ShardedIndex shard search with TaskGroup"`

---

### Task 5: batchSearch Adaptive Concurrency

**Acceptance**: `BatchSearchAdaptiveConcurrencyTests` passes. Fifth git commit.

**Checklist:**

- [ ] 5.1 — Create `Tests/MetalANNSTests/BatchSearchAdaptiveConcurrencyTests.swift` with tests:
  - `gpuModeUsesQueuePoolCount` — build GPU-backed ANNSIndex, verify `batchSearch` uses `queuePool.queues.count` (or higher) as maxConcurrency, not a literal 4
  - `cpuModeUsesProcessorCount` — build CPU-backed ANNSIndex (Accelerate), verify concurrency = `ProcessInfo.processInfo.activeProcessorCount`
  - `batchSearchResultsUnchanged` — 100 queries, verify results same before and after concurrency change

- [ ] 5.2 — **RED**: `gpuModeUsesQueuePoolCount` fails (hardcoded 4 today)

- [ ] 5.3 — Modify `Sources/MetalANNS/ANNSIndex.swift` in `batchSearch()`:
  ```swift
  // BEFORE:
  let maxConcurrency = context != nil ? 4 : max(1, ProcessInfo.processInfo.activeProcessorCount)

  // AFTER:
  let maxConcurrency: Int
  if let ctx = context {
      // Align concurrency with queue pool size so each concurrent search can use a distinct queue
      maxConcurrency = await ctx.queuePool.queues.count
  } else {
      maxConcurrency = max(1, ProcessInfo.processInfo.activeProcessorCount)
  }
  ```

- [ ] 5.4 — Update `MetalBackend.computeDistances()` to use `context.executeOnPool()` instead of `context.execute()` when available, so concurrent batch searches use distinct queues:
  ```swift
  // In MetalBackend.computeDistances():
  // BEFORE:
  try await context.execute { commandBuffer in ... }

  // AFTER:
  try await context.executeOnPool { commandBuffer in ... }
  ```

- [ ] 5.5 — **GREEN**: All 3 tests pass

- [ ] 5.6 — **REGRESSION**: All Phase 6-16 GPU tests still pass

- [ ] 5.7 — **GIT**: `git commit -m "feat: adaptive batchSearch concurrency using CommandQueuePool count"`

---

### Task 6: Performance Verification

**Acceptance**: `MultiQueuePerformanceTests` passes with documented speedup. Sixth git commit.

**Checklist:**

- [ ] 6.1 — Create `Tests/MetalANNSTests/MultiQueuePerformanceTests.swift` with tests:
  - `shardedBuildSpeedup` — build 8-shard index with parallelism, measure wall time, log speedup vs estimated sequential (should be > 2x on multi-core Mac)
  - `batchSearchQPS` — 200 queries on 10K-vector GPU-backed index, compute QPS, verify > 1000 QPS (or > baseline)
  - `shardedSearchQPS` — 100 queries on 4-shard index, compute QPS, log result
  - All on-device tests skip on simulator

- [ ] 6.2 — **EXPECTED RESULTS** (document actuals in Task Notes 6):
  - ShardedIndex build: 2-4x speedup for N=4 shards on MacBook Pro M-series
  - batchSearch GPU: 1.5-2x QPS improvement from queue pool vs single queue on large batches
  - ShardedIndex search: 2-4x QPS for N=4 probeShards

- [ ] 6.3 — **GREEN**: Tests pass (QPS/speedup assertions are soft — log results, don't hard-fail on timing)

- [ ] 6.4 — **REGRESSION SWEEP**: Run complete test suite, document any timing regressions

- [ ] 6.5 — **GIT**: `git commit -m "test: add multi-queue performance verification tests"`

---

### Task 7: Full Suite and Completion Signal

**Acceptance**: Full suite passes. Final commit.

**Checklist:**

- [ ] 7.1 — Run: `xcodebuild build -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation` → **BUILD SUCCEEDED**

- [ ] 7.2 — Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation`
  - New tests pass: CommandQueuePoolTests, MetalContextMultiQueueTests, ShardedIndexParallelBuildTests, ShardedIndexParallelSearchTests, BatchSearchAdaptiveConcurrencyTests, MultiQueuePerformanceTests
  - All Phase 13-17 tests unchanged
  - Known MmapTests baseline failure allowed

- [ ] 7.3 — Verify git log shows exactly 7 commits with conventional messages

- [ ] 7.4 — Update Phase Complete Signal below

- [ ] 7.5 — **GIT**: `git commit -m "chore: phase 18 complete - multi-queue parallelism"`

---

## Success Criteria

✅ `CommandQueuePool` — N distinct queues, round-robin, actor-safe
✅ `MetalContext.executeOnPool()` — additive API, `execute()` unchanged
✅ `ShardedIndex.build()` — parallel TaskGroup, results identical to sequential
✅ `ShardedIndex.search()` — parallel TaskGroup, results identical to sequential
✅ `batchSearch` — adaptive concurrency = `queuePool.count` for GPU backend
✅ `MetalBackend` — uses `executeOnPool()` for distance compute calls
✅ No regressions across all Phase 1-17 tests

---

## Anti-Patterns

❌ **Don't** create a new `MetalContext` per concurrent search — `init()` allocates GPU resources
❌ **Don't** change `MetalContext.execute()` signature — backward compat for all existing callers
❌ **Don't** use `@unchecked Sendable` on CommandQueuePool — it's an actor, already Sendable
❌ **Don't** assert hard timing thresholds in performance tests — hardware varies; log and verify > 0
❌ **Don't** share a single `MTLCommandBuffer` across concurrent tasks — create one per `executeOnPool()` call
❌ **Don't** sort ShardedIndex results inside the TaskGroup — collect all then sort after
❌ **Don't** change `PipelineCache` — `MTLComputePipelineState` is already thread-safe
❌ **Don't** set `maxConcurrency > queuePool.count` for GPU backend — it won't improve throughput beyond queue count

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `CommandQueuePool.swift` | **New** | Actor-based queue pool |
| `MetalDevice.swift` | **Modified** | Add `queuePool`, `executeOnPool()` |
| `ShardedIndex.swift` | **Modified** | Parallel build + search TaskGroups |
| `ANNSIndex.swift` | **Modified** | Adaptive batchSearch concurrency |
| `MetalBackend.swift` | **Modified** | Use `executeOnPool()` for distances |
| `CommandQueuePoolTests.swift` | **New** | Pool creation and round-robin tests |
| `MetalContextMultiQueueTests.swift` | **New** | executeOnPool API tests |
| `ShardedIndexParallelBuildTests.swift` | **New** | Parallel build correctness + timing |
| `ShardedIndexParallelSearchTests.swift` | **New** | Parallel search correctness + QPS |
| `BatchSearchAdaptiveConcurrencyTests.swift` | **New** | Adaptive concurrency tests |
| `MultiQueuePerformanceTests.swift` | **New** | End-to-end QPS verification |

**Total new code: ~800 lines (including tests)**

---

## Commits Expected

1. CommandQueuePool actor
2. MetalContext + executeOnPool()
3. ShardedIndex parallel build
4. ShardedIndex parallel search
5. batchSearch adaptive concurrency + MetalBackend executeOnPool
6. Performance verification tests
7. Phase complete signal
