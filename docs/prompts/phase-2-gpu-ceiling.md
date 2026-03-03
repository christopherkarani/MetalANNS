# Phase 2: Lift 4096-Node GPU Search Ceiling

## Role

You are a senior Swift/Metal engineer working in MetalANNS — a GPU-native ANN search library for Apple Silicon. You have deep expertise in Metal compute pipelines, atomic memory operations, and Swift 6 strict concurrency.

## Context

**Project:** MetalANNS at `/Users/chriskarani/CodingProjects/MetalANNS`
**Branch:** `grdb4`
**Language:** Swift 6.0 with strict concurrency
**Build system:** SPM via xcodebuild (NOT `swift build` — Metal shaders require xcodebuild)
**Task tracker:** `tasks/wax-readiness-todo.md` — check off items as you complete each step
**Full plan:** `docs/plans/2026-02-28-metalanns-wax-readiness.md`

### What is ALREADY done — do NOT re-implement these

Read these files before making any changes to fully understand the current state:

- `Sources/MetalANNSCore/Shaders/Search.metal` — already uses `try_visit_global()` with a device-memory CAS (`atomic_compare_exchange_weak_explicit`) per nodeID + a `generation` parameter at buffer index 13. No threadgroup hash table. The 4096 ceiling from the old design is gone at the shader level.
- `Sources/MetalANNSCore/Shaders/SearchFloat16.metal` — identical visited-set approach via `try_visit_global_f16()`.
- `Sources/MetalANNSCore/FullGPUSearch.swift` — already allocates `visitedGenerationBuffer` with `length = max(nodeCount * MemoryLayout<UInt32>.stride, MemoryLayout<UInt32>.stride)`. No 4096-node guard anywhere. Phase 1's `SearchBufferPool` is wired in for the 3 search buffers.
- `Sources/MetalANNSCore/SearchBufferPool.swift` — Phase 1 complete. Pools query/output buffers with LRU eviction.

### The remaining gap — what you WILL implement

`FullGPUSearch` currently allocates `visitedGenerationBuffer` fresh on every search call and **zero-initializes it with O(n) CPU work**:

```swift
// Current code in FullGPUSearch.swift (lines ~65-76)
guard let visitedGenerationBuffer = context.device.makeBuffer(
    length: visitedLength,
    options: .storageModeShared
) else { ... }
visitedGenerationBuffer.contents()
    .bindMemory(to: UInt32.self, capacity: max(nodeCount, 1))
    .initialize(repeating: 0, count: max(nodeCount, 1))

var visitGenerationValue: UInt32 = 1
```

At 100k nodes × 4 bytes = 400KB of CPU memset on every search call. This defeats the purpose of the buffer pool.

**The fix:** Extend `SearchBufferPool` with a visited-buffer slot that uses a **generation counter** — a monotonically incrementing UInt32 that identifies each search. A node is "visited in the current search" iff `visited[nodeID] == currentGeneration`. Since generations advance, old marks from prior searches are invisible without any zero-init. This gives O(1) setup cost regardless of index size.

## Constraints (READ FIRST)

- Swift 6.0 strict concurrency — all new code must be `Sendable`
- `.storageModeShared` for all `MTLBuffer` allocations (Apple Silicon unified memory)
- Do NOT modify any `.metal` shader file — shaders are already correct
- Do NOT change buffer binding indices in `FullGPUSearch` — the encoder uses fixed indices 0-13
- Do NOT change the public signature of `FullGPUSearch.search()`
- Use Swift Testing (`import Testing`, `@Test`, `#expect`) — NOT XCTest
- Metal is unavailable on simulator — all tests must guard with `guard MTLCreateSystemDefaultDevice() != nil`
- **Commit after each task. Do not batch tasks into one commit.**

## Definition of Done

Phase 2 is complete when ALL of these are true:
1. `searchAbove4096NodesReturnsResults` test passes at `nodeCount = 5000`
2. `gpuSearchMatchesCPUAtSmallScale` parity test passes with recall >= 0.70
3. `SearchBufferPool` has `acquireVisited(nodeCount:)` and `releaseVisited(_:capacity:)` with 2+ unit tests
4. `FullGPUSearch` uses visited-buffer pooling — zero calls to `.initialize(repeating: 0)` in the search path
5. Full suite passes: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
6. Two clean commits on the branch
7. All Phase 2 checkboxes in `tasks/wax-readiness-todo.md` marked `[x]`

## Progress Tracking

After each commit, mark completed items in `tasks/wax-readiness-todo.md` with `[x]`. The todo list is the source of truth.

## Anti-Patterns to Avoid

- **Do NOT zero-initialize the visited buffer** when reusing from pool — that's exactly what we're eliminating. Trust the generation counter.
- **Generation 0 is forbidden as an active generation** — the buffer contents default to 0 on allocation, so if generation == 0 a freshly allocated buffer would appear fully visited. Skip 0 by starting the counter at 1 and wrapping: `counter = counter == UInt32.max ? 1 : counter + 1`.
- **Do NOT use an `actor` for visited buffer management** — it would force `await` in a sync context. Use `NSLock`.
- **Do NOT share a single generationCounter across all pool instances** — the counter must be per-pool (it's a property on `SearchBufferPool`).
- **Do NOT add the visited buffer to the existing `Buffers` struct** — it has different sizing semantics (nodeCount, not queryDim). Keep it as a separate struct `VisitedBuffers`.

---

## Task 2.1: Write and Run Verification Tests

**Depends on:** Nothing (write these first to confirm the ceiling is already lifted)
**Modifies:** `Tests/MetalANNSTests/FullGPUSearchTests.swift`

The tests at nodeCount=200 and 500 already exist. Add two new tests that prove correctness at and above the old 4096-node limit.

### Step 1 — Add the two tests

Add to `Tests/MetalANNSTests/FullGPUSearchTests.swift` inside the `FullGPUSearchTests` suite:

```swift
@Test("GPU search works above 4096-node old limit")
func searchAbove4096NodesReturnsResults() async throws {
    guard MTLCreateSystemDefaultDevice() != nil else { return }

    let nodeCount = 5000  // Above old 4096 limit
    let dim = 32
    let degree = 16
    let context = try MetalContext()
    let vectors = randomVectors(count: nodeCount, dim: dim)
    let (vectorBuffer, graphBuffer, entryPoint) = try await buildBuffers(
        context: context,
        vectors: vectors,
        degree: degree,
        metric: .cosine
    )

    let results = try await FullGPUSearch.search(
        context: context,
        query: vectors[0],
        vectors: vectorBuffer,
        graph: graphBuffer,
        entryPoint: Int(entryPoint),
        k: 10,
        ef: 64,
        metric: .cosine
    )

    #expect(results.count == 10, "GPU search at 5000 nodes should return 10 results")
    // The query vector is in the index, so the top result should be near-zero distance
    #expect(results[0].score < 0.05, "Top result should be near-exact match at large scale")
}

@Test("GPU search recall matches CPU at small scale")
func gpuSearchMatchesCPUAtSmallScale() async throws {
    guard MTLCreateSystemDefaultDevice() != nil else { return }

    let nodeCount = 200
    let dim = 32
    let degree = 16
    let k = 10
    let ef = 64
    let context = try MetalContext()
    let vectors = randomVectors(count: nodeCount, dim: dim)
    let (vectorBuffer, graphBuffer, entryPoint) = try await buildBuffers(
        context: context,
        vectors: vectors,
        degree: degree,
        metric: .cosine
    )

    let query = vectors[5]
    let gpuResults = try await FullGPUSearch.search(
        context: context,
        query: query,
        vectors: vectorBuffer,
        graph: graphBuffer,
        entryPoint: Int(entryPoint),
        k: k,
        ef: ef,
        metric: .cosine
    )

    let exactTopK = Set(
        vectors.enumerated()
            .map { (idx, v) in (UInt32(idx), SIMDDistance.cosine(query, v)) }
            .sorted { $0.1 < $1.1 }
            .prefix(k)
            .map(\.0)
    )
    let gpuIDs = Set(gpuResults.map(\.internalID))
    let recall = Float(exactTopK.intersection(gpuIDs).count) / Float(k)

    #expect(recall >= 0.70, "GPU-vs-brute-force recall \(recall) is below threshold 0.70")
}
```

### Step 2 — Run both tests

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/FullGPUSearchTests/searchAbove4096NodesReturnsResults \
  -only-testing MetalANNSTests/FullGPUSearchTests/gpuSearchMatchesCPUAtSmallScale \
  2>&1 | tail -30
```

**Expected:** Both PASS. The ceiling is already lifted in the shaders and Swift code.

**If `searchAbove4096NodesReturnsResults` fails:** The kernel may be crashing or returning garbage. Check:
1. Is `Search.metal` really using device-memory generation counter? Run `grep -n "MAX_VISITED\|try_visit_global" Sources/MetalANNSCore/Shaders/Search.metal`
2. Is the `visitedGenerationBuffer` bound at index 12? Check the encoder calls in `FullGPUSearch.swift`.
3. Look for any GPU command buffer errors in the test output.

**If `gpuSearchMatchesCPUAtSmallScale` fails (recall < 0.70):** The graph quality may be low. Check that `buildBuffers` in `FullGPUSearchTests.swift` uses `NNDescentCPU.build` with `maxIterations: 10` or more.

### Step 3 — Commit

```bash
git add Tests/MetalANNSTests/FullGPUSearchTests.swift
git commit -m "test: verify GPU search works above 4096-node limit and matches CPU recall"
```

### Step 4 — Update todo

Mark Task 2.1 items in `tasks/wax-readiness-todo.md`.

---

## Task 2.2: Add Visited Buffer Pooling to SearchBufferPool

**Depends on:** Task 2.1 committed
**Modifies:** `Sources/MetalANNSCore/SearchBufferPool.swift`, `Tests/MetalANNSTests/SearchBufferPoolTests.swift`

### Step 1 — Write the failing tests

Add to `Tests/MetalANNSTests/SearchBufferPoolTests.swift`:

```swift
@Test func acquireAndReleaseVisitedReturnsSameBuffer() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Skipping: no Metal device"); return
    }
    let pool = SearchBufferPool(device: device)

    let v1 = try pool.acquireVisited(nodeCount: 1000)
    let ptr1 = v1.buffer.gpuAddress
    pool.releaseVisited(v1.buffer, capacity: 1000)

    let v2 = try pool.acquireVisited(nodeCount: 1000)
    #expect(v2.buffer.gpuAddress == ptr1, "Pooled visited buffer should be reused")
    pool.releaseVisited(v2.buffer, capacity: 1000)
}

@Test func visitedGenerationsAreMonotonicallyIncreasing() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Skipping: no Metal device"); return
    }
    let pool = SearchBufferPool(device: device)

    let v1 = try pool.acquireVisited(nodeCount: 100)
    let gen1 = v1.generation
    pool.releaseVisited(v1.buffer, capacity: 100)

    let v2 = try pool.acquireVisited(nodeCount: 100)
    let gen2 = v2.generation
    pool.releaseVisited(v2.buffer, capacity: 100)

    #expect(gen2 > gen1, "Each acquire should return a strictly higher generation")
    #expect(gen1 != 0, "Generation must never be 0 (reserved as unvisited sentinel)")
    #expect(gen2 != 0, "Generation must never be 0 (reserved as unvisited sentinel)")
}

@Test func concurrentVisitedAcquireReturnsDistinctBuffers() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Skipping: no Metal device"); return
    }
    let pool = SearchBufferPool(device: device)

    let v1 = try pool.acquireVisited(nodeCount: 100)
    let v2 = try pool.acquireVisited(nodeCount: 100)
    #expect(v1.buffer.gpuAddress != v2.buffer.gpuAddress,
        "Concurrent acquires must return distinct buffers")
    pool.releaseVisited(v1.buffer, capacity: 100)
    pool.releaseVisited(v2.buffer, capacity: 100)
}
```

### Step 2 — Run to verify they fail

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/SearchBufferPoolTests/acquireAndReleaseVisitedReturnsSameBuffer \
  2>&1 | tail -20
```

**Expected:** Compilation error — `acquireVisited` does not exist.

### Step 3 — Implement visited buffer pooling in SearchBufferPool

Add the following to `Sources/MetalANNSCore/SearchBufferPool.swift`:

```swift
// MARK: - Visited Buffer Support

public struct VisitedBuffers: Sendable {
    public let buffer: MTLBuffer
    public let generation: UInt32
}

private var visitedAvailable: [(buffer: MTLBuffer, capacity: Int)] = []
private var generationCounter: UInt32 = 0

/// Acquires a visited-generation buffer sized for `nodeCount` nodes.
/// Returns a buffer and a unique non-zero generation value.
/// The buffer is NOT zeroed — callers rely on the generation counter for isolation.
public func acquireVisited(nodeCount: Int) throws -> VisitedBuffers {
    lock.lock()
    defer { lock.unlock() }

    // Advance generation counter, skipping 0 (reserved as "unvisited" sentinel)
    generationCounter = generationCounter == UInt32.max ? 1 : generationCounter + 1
    let generation = generationCounter

    if let index = visitedAvailable.firstIndex(where: { $0.capacity >= nodeCount }) {
        let entry = visitedAvailable.remove(at: index)
        return VisitedBuffers(buffer: entry.buffer, generation: generation)
    }

    let length = max(nodeCount * MemoryLayout<UInt32>.stride, MemoryLayout<UInt32>.stride)
    guard let buf = device.makeBuffer(length: length, options: .storageModeShared) else {
        throw ANNSError.searchFailed("Failed to allocate visited generation buffer")
    }
    // Zero-initialize only on first allocation so generation 0 == "never visited"
    buf.contents().initializeMemory(as: UInt32.self, repeating: 0, count: nodeCount)
    return VisitedBuffers(buffer: buf, generation: generation)
}

/// Returns a visited buffer to the pool. Pass the nodeCount used during acquire as `capacity`.
public func releaseVisited(_ buffer: MTLBuffer, capacity: Int) {
    lock.lock()
    defer { lock.unlock() }
    visitedAvailable.append((buffer: buffer, capacity: capacity))
}
```

### Step 4 — Run the 3 new tests

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/SearchBufferPoolTests/acquireAndReleaseVisitedReturnsSameBuffer \
  -only-testing MetalANNSTests/SearchBufferPoolTests/visitedGenerationsAreMonotonicallyIncreasing \
  -only-testing MetalANNSTests/SearchBufferPoolTests/concurrentVisitedAcquireReturnsDistinctBuffers \
  2>&1 | tail -20
```

**Expected:** 3 tests PASS.

### Step 5 — Commit

```bash
git add Sources/MetalANNSCore/SearchBufferPool.swift Tests/MetalANNSTests/SearchBufferPoolTests.swift
git commit -m "feat: add visited-buffer pooling with generation counter to SearchBufferPool"
```

---

## Task 2.3: Wire Visited Pool into FullGPUSearch

**Depends on:** Task 2.2 committed
**Modifies:** `Sources/MetalANNSCore/FullGPUSearch.swift`

### Step 1 — Replace direct alloc + zero-init with pool

Open `Sources/MetalANNSCore/FullGPUSearch.swift`.

Find the block that allocates `visitedGenerationBuffer` (the `guard let visitedGenerationBuffer = context.device.makeBuffer(...)` block and the `.initialize(repeating: 0, ...)` call below it, plus the `var visitGenerationValue: UInt32 = 1` line).

**Replace that entire section** with:

```swift
let visited = try context.searchBufferPool.acquireVisited(nodeCount: nodeCount)
defer { context.searchBufferPool.releaseVisited(visited.buffer, capacity: nodeCount) }

let visitedGenerationBuffer = visited.buffer
var visitGenerationValue = visited.generation
```

**Key points:**
- The `defer` must be after both `acquire` calls (search buffers + visited buffers) so both are released together after GPU work completes
- `visitGenerationValue` is now from the pool (unique per-search, non-zero) instead of hardcoded `1`
- No `.initialize(repeating: 0)` — do NOT add it back

### Step 2 — Run the full suite

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|Passed|Failed|error:)'
```

**Expected:** All tests pass. Specifically verify:
- `SearchBufferPoolTests` — all 10 tests pass
- `FullGPUSearchTests` — all 4 tests pass (including the 2 you added in Task 2.1)
- `NNDescentGPUTests` — all pass (regression check)
- `ANNSIndexTests` — all pass

**If any test fails:** First check that `visitGenerationValue` is still being bound to the encoder at index 13. The encoder setup in `FullGPUSearch` must not have changed. Check that the `defer` for `releaseVisited` appears before `context.execute` so it executes after the GPU command buffer completes (not before).

### Step 3 — Verify zero memset in search path

```bash
grep -n "initialize(repeating: 0" Sources/MetalANNSCore/FullGPUSearch.swift
```

**Expected:** No output — zero occurrences.

### Step 4 — Commit

```bash
git add Sources/MetalANNSCore/FullGPUSearch.swift
git commit -m "perf: replace per-search visited-buffer alloc+memset with generation-counter pool"
```

### Step 5 — Update todo

Mark Task 2.2 items and Phase 2 exit criteria in `tasks/wax-readiness-todo.md`.

---

## Key Files Reference

| File | Lines | Role | Touch? |
|------|-------|------|--------|
| `Sources/MetalANNSCore/SearchBufferPool.swift` | all | Add `VisitedBuffers`, `acquireVisited`, `releaseVisited`, `generationCounter` | YES |
| `Sources/MetalANNSCore/FullGPUSearch.swift` | ~65-84 | Replace visited alloc+memset+hardcoded-gen with pool | YES |
| `Tests/MetalANNSTests/FullGPUSearchTests.swift` | end of suite | Add 2 new tests | YES |
| `Tests/MetalANNSTests/SearchBufferPoolTests.swift` | end of suite | Add 3 visited buffer tests | YES |
| `Sources/MetalANNSCore/Shaders/Search.metal` | all | Already correct | **NO — do not touch** |
| `Sources/MetalANNSCore/Shaders/SearchFloat16.metal` | all | Already correct | **NO — do not touch** |

## Verification Checklist

Before marking Phase 2 complete, confirm:

- [ ] `searchAbove4096NodesReturnsResults` test at 5000 nodes passes
- [ ] `gpuSearchMatchesCPUAtSmallScale` test passes with recall >= 0.70
- [ ] `SearchBufferPool` has `acquireVisited(nodeCount:)` returning a unique non-zero generation
- [ ] `SearchBufferPool` has `releaseVisited(_:capacity:)`
- [ ] `generationCounter` never returns 0 (skips 0, wraps to 1 at UInt32.max)
- [ ] `FullGPUSearch.search()` has zero `.initialize(repeating: 0)` calls — confirmed with grep
- [ ] `visitGenerationValue` comes from the pool, not hardcoded as `1`
- [ ] Both `defer` blocks (search buffers + visited buffer) are before `context.execute`
- [ ] Full test suite passes: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
- [ ] Three clean commits: verification tests, pool extension, FullGPUSearch wiring
