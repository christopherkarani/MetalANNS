# Phase 1: Buffer Pool for GPU Search

## Role

You are a senior Swift/Metal engineer implementing a performance optimization in MetalANNS — a GPU-native ANN search library for Apple Silicon. You have deep expertise in Metal compute pipelines, Swift 6 concurrency, and TDD.

## Context

**Project:** MetalANNS at `/Users/chriskarani/CodingProjects/MetalANNS`
**Branch:** `grdb4` (current working branch)
**Language:** Swift 6.0 with strict concurrency
**Build system:** SPM via xcodebuild (NOT `swift build` — Metal shaders require xcodebuild)
**Task tracker:** `tasks/wax-readiness-todo.md` — check off items as you complete each step
**Full plan:** `docs/plans/2026-02-28-metalanns-wax-readiness.md`

**The problem:** `FullGPUSearch.search()` at `Sources/MetalANNSCore/FullGPUSearch.swift:53-77` allocates 4 new `MTLBuffer` objects on every call in a single guard chain (queryBuffer, outputDistanceBuffer, outputIDBuffer, visitedGenerationBuffer). Metal buffer allocation involves page table modifications. Under concurrent `batchSearch` (which calls `search()` N times), this creates significant allocator pressure and is the dominant non-compute cost.

**The fix:** Introduce a `SearchBufferPool` that pre-allocates and reuses 3 of those buffers across search calls (queryBuffer, outputDistanceBuffer, outputIDBuffer). The `visitedGenerationBuffer` is sized by `nodeCount` (not query dimensions) and will be pooled separately in Phase 2 — leave it as a regular allocation for now. Wire the pool into `MetalContext` so `FullGPUSearch` can acquire/release without API changes.

## Constraints (READ FIRST)

- Swift 6.0 strict concurrency — every new type must be `Sendable`
- All `MTLBuffer` objects MUST use `.storageModeShared` (Apple Silicon unified memory)
- Do NOT modify any `.metal` shader files or kernel buffer binding indices
- Do NOT change the public signature of `FullGPUSearch.search()` — this is an internal refactor only
- Do NOT change `Package.swift` — shaders load via `Bundle.module`
- Metal is unavailable on simulator — tests must guard with `MTLCreateSystemDefaultDevice()` and skip gracefully
- Use the Swift Testing framework (`import Testing`, `@Test`, `#expect`), NOT XCTest
- Commit after each task. Do not batch tasks into one commit.

## Definition of Done

Phase 1 is complete when ALL of these are true:
1. `SearchBufferPool` exists with acquire/release semantics and 3 passing unit tests
2. `FullGPUSearch.search()` uses the pool instead of `device.makeBuffer` — zero per-search allocations
3. `MetalContext` exposes `searchBufferPool` as a public property
4. The FULL test suite passes with zero regressions: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
5. Two clean commits on the current branch
6. All Phase 1 checkboxes in `tasks/wax-readiness-todo.md` are marked `[x]`

## Progress Tracking

As you complete each step, update `tasks/wax-readiness-todo.md` by changing `- [ ]` to `- [x]` for the corresponding item. Do this after each commit, not at the end. The todo list is the source of truth for what's done.

## Anti-Patterns to Avoid

- Do NOT use an `actor` for the pool — it would force `await` on every acquire/release in a synchronous context. Use `NSLock` instead.
- Do NOT store `MTLBuffer` in a `Set` — `MTLBuffer` does not conform to `Hashable`. Use an `Array`.
- Do NOT release buffers before the GPU command buffer completes — use `defer` after acquire so release happens after `context.execute` returns.
- Do NOT assume `queryDim` or `maxK` are constant across calls — different searches may have different dimensions or k values. The pool must match by capacity, not exact size.

---

## Task 1.1: Create SearchBufferPool

**Depends on:** Nothing (start here)
**Creates:** `Sources/MetalANNSCore/SearchBufferPool.swift`, `Tests/MetalANNSTests/SearchBufferPoolTests.swift`

### Step 1 — Write failing tests

Create `Tests/MetalANNSTests/SearchBufferPoolTests.swift` with these 3 tests:

```swift
import Testing
import Metal
@testable import MetalANNSCore

@Suite("SearchBufferPool Tests")
struct SearchBufferPoolTests {

    @Test func acquireAndReleaseReturnsSameBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device"); return
        }
        let pool = SearchBufferPool(device: device)

        let b1 = try pool.acquire(queryDim: 128, maxK: 10)
        let ptr1 = b1.queryBuffer.gpuAddress
        pool.release(b1)

        let b2 = try pool.acquire(queryDim: 128, maxK: 10)
        // After release + re-acquire with same dimensions, should return the same buffer
        #expect(b2.queryBuffer.gpuAddress == ptr1)
        pool.release(b2)
    }

    @Test func acquireLargerDimAllocatesNew() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device"); return
        }
        let pool = SearchBufferPool(device: device)

        let small = try pool.acquire(queryDim: 64, maxK: 10)
        pool.release(small)

        // Larger dim needs a bigger buffer — pool entry is too small, must allocate fresh
        let large = try pool.acquire(queryDim: 512, maxK: 10)
        #expect(large.queryBuffer.length >= 512 * MemoryLayout<Float>.stride)
        pool.release(large)
    }

    @Test func concurrentAcquireReturnsDistinctBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device"); return
        }
        let pool = SearchBufferPool(device: device)

        // Acquire two without releasing — must get distinct buffers
        let b1 = try pool.acquire(queryDim: 128, maxK: 10)
        let b2 = try pool.acquire(queryDim: 128, maxK: 10)
        #expect(b1.queryBuffer.gpuAddress != b2.queryBuffer.gpuAddress)
        pool.release(b1)
        pool.release(b2)
    }
}
```

### Step 2 — Verify tests fail

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/SearchBufferPoolTests 2>&1 | tail -20
```

**Expected output:** Compilation error — `SearchBufferPool` does not exist. This confirms the test is wired correctly.

### Step 3 — Implement SearchBufferPool

Create `Sources/MetalANNSCore/SearchBufferPool.swift`:

```swift
import Foundation
import Metal

/// Thread-safe pool of reusable MTLBuffer triplets for GPU search operations.
/// Eliminates per-search allocation overhead in FullGPUSearch.
public final class SearchBufferPool: @unchecked Sendable {

    public struct Buffers: Sendable {
        public let queryBuffer: MTLBuffer
        public let outputDistanceBuffer: MTLBuffer
        public let outputIDBuffer: MTLBuffer
        public let queryDim: Int
        public let maxK: Int
    }

    private let device: MTLDevice
    private var available: [Buffers] = []
    private let lock = NSLock()

    public init(device: MTLDevice) {
        self.device = device
    }

    /// Returns a buffer set with capacity >= requested dimensions.
    /// If no pooled entry fits, allocates new buffers.
    public func acquire(queryDim: Int, maxK: Int) throws -> Buffers {
        lock.lock()
        defer { lock.unlock() }

        if let index = available.firstIndex(where: {
            $0.queryDim >= queryDim && $0.maxK >= maxK
        }) {
            return available.remove(at: index)
        }

        return try allocate(queryDim: queryDim, maxK: maxK)
    }

    /// Returns buffers to the pool for future reuse.
    public func release(_ buffers: Buffers) {
        lock.lock()
        defer { lock.unlock() }
        available.append(buffers)
    }

    private func allocate(queryDim: Int, maxK: Int) throws -> Buffers {
        let floatSize = MemoryLayout<Float>.stride
        let uintSize = MemoryLayout<UInt32>.stride

        guard
            let qBuf = device.makeBuffer(length: queryDim * floatSize, options: .storageModeShared),
            let dBuf = device.makeBuffer(length: max(maxK * floatSize, floatSize), options: .storageModeShared),
            let iBuf = device.makeBuffer(length: max(maxK * uintSize, uintSize), options: .storageModeShared)
        else {
            throw ANNSError.searchFailed("Failed to allocate search buffer pool entry")
        }

        return Buffers(
            queryBuffer: qBuf,
            outputDistanceBuffer: dBuf,
            outputIDBuffer: iBuf,
            queryDim: queryDim,
            maxK: maxK
        )
    }
}
```

### Step 4 — Verify tests pass

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/SearchBufferPoolTests 2>&1 | tail -20
```

**Expected output:** 3 tests passed, 0 failures.

**If tests fail:** Check that `ANNSError` is accessible from `MetalANNSCore` (it's defined in `Sources/MetalANNSCore/Errors.swift`). Check `.storageModeShared` is correct for your platform.

### Step 5 — Commit

```bash
git add Sources/MetalANNSCore/SearchBufferPool.swift Tests/MetalANNSTests/SearchBufferPoolTests.swift
git commit -m "feat: add SearchBufferPool to eliminate per-search MTLBuffer allocation"
```

### Step 6 — Update todo

Mark Task 1.1 items complete in `tasks/wax-readiness-todo.md`.

---

## Task 1.2: Wire SearchBufferPool into FullGPUSearch

**Depends on:** Task 1.1 must be committed first.
**Modifies:** `Sources/MetalANNSCore/MetalDevice.swift`, `Sources/MetalANNSCore/FullGPUSearch.swift`
**Adds to:** `Tests/MetalANNSTests/SearchBufferPoolTests.swift`

### Step 1 — Write safety test FIRST

This test validates that `FullGPUSearch.search()` still returns correct results after the refactor. Add to `Tests/MetalANNSTests/SearchBufferPoolTests.swift`:

```swift
// Add at top of file, outside the suite:
private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

// Add inside the @Suite:
@Test func fullGPUSearchCorrectAfterPoolRefactor() async throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Skipping: no Metal device"); return
    }
    let context = try MetalContext()

    let dim = 16
    let nodeCount = 32
    let degree = 8
    var rng = SeededGenerator(state: 42)
    let vectors = (0..<nodeCount).map { _ in
        (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
    }

    let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
    for (i, v) in vectors.enumerated() { try vectorBuffer.setVector(v, at: i) }
    vectorBuffer.setCount(nodeCount)

    let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
    try await NNDescentGPU.build(
        context: context, graph: graph, vectors: vectorBuffer,
        nodeCount: nodeCount, metric: .cosine
    )

    // Run two searches — validates correctness AND that pool reuse works
    let r1 = try await FullGPUSearch.search(
        context: context, query: vectors[0], vectors: vectorBuffer,
        graph: graph, entryPoint: 0, k: 5, ef: 16, metric: .cosine
    )
    let r2 = try await FullGPUSearch.search(
        context: context, query: vectors[1], vectors: vectorBuffer,
        graph: graph, entryPoint: 0, k: 5, ef: 16, metric: .cosine
    )

    #expect(r1.count > 0, "First search returned no results")
    #expect(r2.count > 0, "Second search returned no results")
    #expect(r1[0].score < 0.1, "First result should be near-exact match (query is in index)")
}
```

### Step 2 — Verify test passes BEFORE refactoring

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/SearchBufferPoolTests/fullGPUSearchCorrectAfterPoolRefactor 2>&1 | tail -20
```

**Expected:** PASS. This establishes the baseline. If it fails now, the problem is NOT your refactor.

### Step 3 — Add pool to MetalContext

Open `Sources/MetalANNSCore/MetalDevice.swift`.

**At line 12** (after `public let pipelineCache: PipelineCache`), add:

```swift
public let searchBufferPool: SearchBufferPool
```

**At line 36** (inside `init()`, after `self.pipelineCache = PipelineCache(...)`), add:

```swift
self.searchBufferPool = SearchBufferPool(device: device)
```

### Step 4 — Replace buffer allocation in FullGPUSearch

Open `Sources/MetalANNSCore/FullGPUSearch.swift`.

Read the file first to understand the exact guard chain. You will see a single `guard` block (around lines 53-77) that allocates all 4 buffers: `queryBuffer`, `outputDistanceBuffer`, `outputIDBuffer`, and `visitedGenerationBuffer`.

**Replace only the first 3 buffer allocations** — keep `visitedGenerationBuffer` as a regular `device.makeBuffer` call. The result should look like:

```swift
// Pool-managed buffers (query, output distances, output IDs)
let buffers = try context.searchBufferPool.acquire(queryDim: query.count, maxK: kLimit)
defer { context.searchBufferPool.release(buffers) }

let queryBuffer = buffers.queryBuffer
queryBuffer.contents().copyMemory(from: query, byteCount: query.count * floatSize)
let outputDistanceBuffer = buffers.outputDistanceBuffer
let outputIDBuffer = buffers.outputIDBuffer

// visitedGenerationBuffer stays as a direct allocation (pooled in Phase 2)
let visitedLength = max(nodeCount * uintSize, uintSize)
guard let visitedGenerationBuffer = context.device.makeBuffer(
    length: visitedLength, options: .storageModeShared
) else {
    throw ANNSError.searchFailed("Failed to allocate visited generation buffer")
}
// Zero-initialize so generation 0 is never a valid active generation
visitedGenerationBuffer.contents().initializeMemory(as: UInt32.self, repeating: 0, count: nodeCount)
```

**Critical:** The `defer` ensures buffers are returned to the pool even if `context.execute` throws. Verify that `outputDistanceBuffer` and `outputIDBuffer` are still bound at the same encoder indices as before — do not renumber any buffer bindings.

### Step 5 — Run FULL test suite

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|Passed|Failed|error:)'
```

**Expected:** All test suites passed, 0 failures. Look specifically for:
- `SearchBufferPoolTests` — 4 tests passed
- `NNDescentGPUTests` — all passed
- `ANNSIndexTests` — all passed
- `GPUADCSearchTests` — all passed

**If any test fails:** The refactor broke something. Check:
1. Did you leave the old `guard let queryBuffer` code in place alongside the new code? (Remove the old code entirely.)
2. Is `buffers.queryBuffer` being written to with `copyMemory` before the encoder reads it?
3. Are `outputDistanceBuffer` and `outputIDBuffer` the same variables the encoder binds at index 3 and 4?

### Step 6 — Commit

```bash
git add Sources/MetalANNSCore/FullGPUSearch.swift Sources/MetalANNSCore/MetalDevice.swift Tests/MetalANNSTests/SearchBufferPoolTests.swift
git commit -m "refactor: wire SearchBufferPool into FullGPUSearch, eliminating per-search allocation"
```

### Step 7 — Update todo

Mark Task 1.2 items and Phase 1 exit criteria complete in `tasks/wax-readiness-todo.md`.

---

## Key Files Reference

| File | Lines | Role | Touch? |
|------|-------|------|--------|
| `Sources/MetalANNSCore/FullGPUSearch.swift` | 53-77 | 4-buffer guard chain — replace 3, keep visitedGenerationBuffer | YES — replace |
| `Sources/MetalANNSCore/FullGPUSearch.swift` | 96-125 | Command encoder setup | NO — leave as-is |
| `Sources/MetalANNSCore/FullGPUSearch.swift` | 127-141 | Result extraction from output buffers | NO — leave as-is |
| `Sources/MetalANNSCore/MetalDevice.swift` | 8-12 | MetalContext properties | YES — add pool |
| `Sources/MetalANNSCore/MetalDevice.swift` | 14-39 | MetalContext.init() | YES — init pool |
| `Sources/MetalANNSCore/MetalDevice.swift` | 41-52 | execute() method | NO — leave as-is |
| `Sources/MetalANNSCore/Errors.swift` | 3-10 | ANNSError enum (.searchFailed) | NO — just use it |
| `Sources/MetalANNSCore/Shaders/Search.metal` | all | beam_search kernel | NO — do not touch |
| `Tests/MetalANNSTests/GPUADCSearchTests.swift` | 344-353 | SeededGenerator pattern | Reference only |

## Verification Checklist

Before marking Phase 1 complete, confirm:

- [ ] `SearchBufferPool.swift` exists in `Sources/MetalANNSCore/`
- [ ] `SearchBufferPoolTests.swift` has 4 tests (3 unit + 1 integration)
- [ ] `MetalContext` has `searchBufferPool` property
- [ ] `FullGPUSearch.search()` has exactly ONE `device.makeBuffer` call (visitedGenerationBuffer only — 3 buffers now come from pool)
- [ ] `FullGPUSearch.search()` uses `defer` for buffer release
- [ ] Full test suite passes: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
- [ ] Two commits on the branch: one for pool creation, one for wiring
