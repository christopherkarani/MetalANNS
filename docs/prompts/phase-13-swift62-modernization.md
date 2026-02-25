# Phase 13 Execution Prompt: Swift 6.2 Modernization

---

## System Context

You are implementing **Phase 13** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1–12 are complete. The package compiles under `swift-tools-version: 6.0` with `.swiftLanguageMode(.v6)`. All tests pass. The codebase has 34+ source files and 32+ test files.

You are modernizing the entire codebase to Swift 6.2 — adding typed throws, `@concurrent` actor methods, and targeted `InlineArray` usage. This is a **refactoring phase**: no new features, no new files. Every change must preserve existing behavior exactly.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing a refactoring plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/todo.md`

**Your communication contract:**
1. **`tasks/todo.md` is your shared state.** Check off `[x]` items as you complete them. The orchestrator reads this file to track your progress.
2. **Write notes under every task** — especially for any decisions, compiler errors, or unexpected behaviors.
3. **Update `Last Updated`** at the top of todo.md after each task completes.
4. **When done, fill in the "Phase 13 Complete — Signal" section** at the bottom of todo.md.
5. **Do NOT modify the "Orchestrator Review Checklist"** section — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **Behavior-preserving refactor**: Every test that passes before this phase MUST pass after. Zero regressions.
2. **Build with `xcodebuild`**, never `swift build` or `swift test`. Metal shaders are not compiled by SPM CLI.
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest.
4. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
5. **Commit after every task** with the conventional commit message specified in the todo.
6. **Run the full test suite after every commit** to verify zero regressions. If tests break, fix immediately before moving on.
7. **Do NOT add new public API surface** except the new `ANNSError` cases and `@concurrent` annotations.
8. **Do NOT rename existing error cases** — only add new ones. Existing callers must not break.

---

## Xcodebuild Commands

Build:
```bash
xcodebuild build -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation 2>&1 | tail -20
```

Test:
```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation 2>&1 | tail -40
```

---

## Current Codebase Snapshot (What You're Working With)

### Package.swift (current)
```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MetalANNS",
    platforms: [.iOS(.v17), .macOS(.v14), .visionOS(.v1)],
    products: [.library(name: "MetalANNS", targets: ["MetalANNS"])],
    targets: [
        .target(name: "MetalANNSCore", resources: [.process("Shaders")], swiftSettings: [.swiftLanguageMode(.v6)]),
        .target(name: "MetalANNS", dependencies: ["MetalANNSCore"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(name: "MetalANNSTests", dependencies: ["MetalANNS", "MetalANNSCore"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .executableTarget(name: "MetalANNSBenchmarks", dependencies: ["MetalANNS", "MetalANNSCore"], swiftSettings: [.swiftLanguageMode(.v6)])
    ]
)
```

### ANNSError (current — `Sources/MetalANNSCore/Errors.swift`)
```swift
public enum ANNSError: Error, Sendable {
    case deviceNotSupported
    case dimensionMismatch(expected: Int, got: Int)
    case idAlreadyExists(String)
    case idNotFound(String)
    case corruptFile(String)
    case constructionFailed(String)
    case searchFailed(String)
    case indexEmpty
}
```

### Public API Re-export (`Sources/MetalANNS/Errors.swift`)
```swift
import MetalANNSCore
public typealias ANNSError = MetalANNSCore.ANNSError
```

---

## Task 1: Bump to Swift 6.2

### What to change

**`Package.swift`**: Change `swift-tools-version: 6.0` → `swift-tools-version: 6.2`

That's it. The `.swiftLanguageMode(.v6)` settings remain unchanged — they're still valid in 6.2 and ensure strict concurrency.

### Verify
- `xcodebuild build` succeeds with no warnings from this change
- All existing tests pass

### Why no `.v6_2` language mode?
The `.v6` language mode already enables strict concurrency. Swift 6.2 features like typed throws and `@concurrent` are available regardless of language mode — they're language features, not concurrency mode features.

---

## Task 2: Expand ANNSError

### What to change

**`Sources/MetalANNSCore/Errors.swift`** — Add three new cases:

```swift
public enum ANNSError: Error, Sendable {
    // Existing cases (DO NOT MODIFY)
    case deviceNotSupported
    case dimensionMismatch(expected: Int, got: Int)
    case idAlreadyExists(String)
    case idNotFound(String)
    case corruptFile(String)
    case constructionFailed(String)
    case searchFailed(String)
    case indexEmpty

    // New cases for Phase 13+
    case serializationFailed(String)
    case metalError(String)
    case invalidArgument(String)
}
```

### Rationale for new cases
- **`serializationFailed`**: Currently, serialization errors are thrown as `constructionFailed` or `corruptFile`, conflating construction with I/O. Used in `IndexSerializer` and `MmapIndexLoader`.
- **`metalError`**: Currently, Metal pipeline/buffer failures are thrown as `constructionFailed` or `searchFailed`. This disambiguates Metal-specific failures.
- **`invalidArgument`**: For parameter validation that doesn't fit dimension/ID patterns (e.g., "ef must be >= k", "degree out of bounds").

### Important: Do NOT migrate existing throw sites to new cases yet
Adding the cases now enables future phases to use them. Migrating existing throw sites would break the typed-throws boundary mapping (Task 3) and risk behavioral changes. The migration happens incrementally in future phases as code is touched.

### Verify
- Build succeeds
- All tests pass (new cases don't break existing `catch` patterns since Swift `catch` is exhaustive only with typed throws)

---

## Task 3: Typed Throws on Public API (`ANNSIndex.swift`)

### Theory of Typed Throws

In Swift 6.2, `throws(ANNSError)` means the function can ONLY throw `ANNSError`. The compiler enforces this:
- Every `throw` in the body must throw `ANNSError`
- Every `try` on a sub-call must either: (a) call a function that `throws(ANNSError)`, or (b) be wrapped in a `do/catch` that maps the error to `ANNSError`

### Decision Framework: Which methods get `throws(ANNSError)`?

**Rule**: A public method gets `throws(ANNSError)` if ALL of its throw sites already throw `ANNSError` AND all called functions either throw `ANNSError` or don't throw.

### Method-by-method analysis for `ANNSIndex.swift`

| Method | Current throws | Internal calls that throw | Typed throw? | Reason |
|--------|---------------|--------------------------|-------------|--------|
| `build(vectors:ids:)` | `ANNSError` | `NNDescentGPU.build()`, `NNDescentCPU.build()`, `GraphPruner.prune()`, `VectorBuffer.init()`, `GraphBuffer.init()` | **YES** — but only after Task 4 makes the internal calls typed too | All internal throws are already `ANNSError` |
| `insert(_:id:)` | `ANNSError` | `IncrementalBuilder.insert()` | **YES** — after Task 4 | Same reasoning |
| `batchInsert(_:ids:)` | `ANNSError` | `BatchIncrementalBuilder.batchInsert()` | **YES** — after Task 4 | Same reasoning |
| `delete(id:)` | `ANNSError` | None | **YES** | Pure `ANNSError` throws |
| `compact()` | `ANNSError` | `IndexCompactor.compact()` | **YES** — after Task 4 | Same reasoning |
| `setMetadata(_:value:for:)` (×3) | `ANNSError` | None | **YES** | Pure `ANNSError` throws |
| `search(query:k:filter:metric:)` | `ANNSError` | `FullGPUSearch.search()`, `BeamSearchCPU.search()` | **YES** — after Task 4 | Same reasoning |
| `rangeSearch(query:maxDistance:...)` | `ANNSError` | Same as search | **YES** — after Task 4 | Same reasoning |
| `batchSearch(queries:k:...)` | `ANNSError` | Calls `self.search()` | **YES** — after search is typed | Same reasoning |
| `save(to:)` | mixed | `IndexSerializer.save()` throws `ANNSError`, but `JSONEncoder().encode()` throws generic `Error` | **YES** — wrap JSON encoding in do/catch | Map `EncodingError` → `ANNSError.serializationFailed` |
| `saveMmapCompatible(to:)` | mixed | Same pattern as `save(to:)` | **YES** — same wrapping | Same |
| `load(from:)` | mixed | `IndexSerializer.load()` + `JSONDecoder().decode()` | **YES** — wrap JSON decoding | Map `DecodingError` → `ANNSError.corruptFile` |
| `loadMmap(from:)` | mixed | `MmapIndexLoader.load()` + JSON decoding | **YES** — same | Same |
| `loadDiskBacked(from:)` | mixed | `DiskBackedIndexLoader.load()` + JSON decoding | **YES** — same | Same |
| `count` (computed property) | non-throwing | — | N/A | Not throwing |

### Transformation pattern for `save(to:)` and similar:

**Before:**
```swift
public func save(to url: URL) async throws {
    // ...
    let metadataData = try JSONEncoder().encode(metadata)
    try metadataData.write(to: Self.metadataURL(for: url), options: .atomic)
}
```

**After:**
```swift
public func save(to url: URL) async throws(ANNSError) {
    // ...
    let metadataData: Data
    do {
        metadataData = try JSONEncoder().encode(metadata)
    } catch {
        throw .serializationFailed("Failed to encode metadata: \(error)")
    }
    do {
        try metadataData.write(to: Self.metadataURL(for: url), options: .atomic)
    } catch {
        throw .serializationFailed("Failed to write metadata: \(error)")
    }
}
```

### Transformation pattern for static `load` methods:

The `load` methods are `static` and not actor-isolated. They call `ANNSIndex(configuration:)` and `await index.applyLoadedState(...)`. The pattern:

**Before:**
```swift
public static func load(from url: URL) async throws -> ANNSIndex {
    let persistedMetadata = try loadPersistedMetadataIfPresent(from: url)
    // ...
}
```

**After:**
```swift
public static func load(from url: URL) async throws(ANNSError) -> ANNSIndex {
    let persistedMetadata: PersistedMetadata?
    do {
        persistedMetadata = try loadPersistedMetadataIfPresent(from: url)
    } catch let error as ANNSError {
        throw error
    } catch {
        throw .corruptFile("Failed to load metadata: \(error)")
    }
    // ...
}
```

Wait — `loadPersistedMetadataIfPresent` already throws generic `Error` (from `Data(contentsOf:)` and `JSONDecoder`). You need to wrap it. But actually, look at the implementation: it already catches and re-throws `ANNSError` patterns. The cleanest approach: make `loadPersistedMetadataIfPresent` itself `throws(ANNSError)` by wrapping its internals.

### The `loadPersistedMetadataIfPresent` helper:

**Before:**
```swift
private nonisolated static func loadPersistedMetadataIfPresent(from fileURL: URL) throws -> PersistedMetadata? {
    let metadataURL = metadataURL(for: fileURL)
    guard FileManager.default.fileExists(atPath: metadataURL.path) else { return nil }
    let data = try Data(contentsOf: metadataURL)
    return try JSONDecoder().decode(PersistedMetadata.self, from: data)
}
```

**After:**
```swift
private nonisolated static func loadPersistedMetadataIfPresent(from fileURL: URL) throws(ANNSError) -> PersistedMetadata? {
    let metadataURL = metadataURL(for: fileURL)
    guard FileManager.default.fileExists(atPath: metadataURL.path) else { return nil }
    let data: Data
    do {
        data = try Data(contentsOf: metadataURL)
    } catch {
        throw .corruptFile("Failed to read metadata file: \(error)")
    }
    do {
        return try JSONDecoder().decode(PersistedMetadata.self, from: data)
    } catch {
        throw .corruptFile("Failed to decode metadata: \(error)")
    }
}
```

### `ShardedIndex.swift` — same treatment

Apply typed throws to:
- `build(vectors:ids:)` → `throws(ANNSError)` — all throw sites are already `ANNSError`
- `search(query:k:filter:metric:)` → `throws(ANNSError)` — same
- `count` → non-throwing, no change

### Verify after this task
- Build succeeds
- All tests pass
- Spot-check: write a test that catches a specific `ANNSError` case to confirm typed throws work:

```swift
@Test("typedThrowsCatchSpecificError")
func typedThrowsCatchSpecificError() async {
    let index = ANNSIndex()
    do {
        try await index.search(query: [1.0], k: 5)
    } catch .indexEmpty {
        // Typed throw: compiler knows this is ANNSError
        // This pattern only compiles with typed throws
    } catch {
        Issue.record("Expected .indexEmpty, got \(error)")
    }
}
```

---

## Task 4: Typed Throws on Core Internal Types

These types are called by `ANNSIndex` methods. They must be `throws(ANNSError)` for the typed throws chain to compile end-to-end.

### `IndexSerializer.swift`

Every method in `IndexSerializer` already throws only `ANNSError`. BUT the `load` method calls `JSONDecoder().decode()` which throws generic `Error`. Wrap it.

**Methods to annotate:**
- `save(vectors:graph:idMap:entryPoint:metric:to:)` → `throws(ANNSError)` — already only throws `ANNSError`
- `saveMmapCompatible(...)` → `throws(ANNSError)` — same
- `load(from:device:)` → `throws(ANNSError)` — wrap the JSON decode call

The `load` method's JSON decode:
```swift
// Before:
let idMap: IDMap
do {
    idMap = try JSONDecoder().decode(IDMap.self, from: idMapData)
} catch {
    throw ANNSError.corruptFile("IDMap payload is corrupt")
}
```
This already catches and re-throws as `ANNSError`! So this method is already compliant. Just add the annotation.

But `save` calls `JSONEncoder().encode(idMap)` which throws generic `Error`. Wrap it:
```swift
let idMapData: Data
do {
    idMapData = try JSONEncoder().encode(idMap)
} catch {
    throw ANNSError.serializationFailed("Failed to encode IDMap: \(error)")
}
```

Also wrap `FileManager.default.createDirectory` and `filePayload.write(to:)`:
```swift
do {
    try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
} catch {
    throw ANNSError.serializationFailed("Failed to create directory: \(error)")
}
do {
    try filePayload.write(to: fileURL)
} catch {
    throw ANNSError.serializationFailed("Failed to write file: \(error)")
}
```

**Private helpers** (`readUInt32`, `metric(from:)`) — these already throw only `ANNSError`. Add `throws(ANNSError)`.

### `MmapIndexLoader.swift`

- `load(from:device:)` → `throws(ANNSError)` — already only throws `ANNSError` (JSON decode is already wrapped)
- Private helpers (`readBytes`, `readUInt32`, `metric(from:)`) → `throws(ANNSError)` — already only throw `ANNSError`
- `MmapRegion.init(fileURL:)` → `throws(ANNSError)` — already only throws `ANNSError`

### `NNDescentGPU.swift`

- `randomInit(...)` → `throws(ANNSError)` — already only throws `ANNSError`
- `computeInitialDistances(...)` → `throws(ANNSError)` — same
- `build(...)` → `throws(ANNSError)` — same
- `sortNeighborLists(...)` → `throws(ANNSError)` — same

**Watch out**: `context.execute { commandBuffer in ... }` — what does this closure throw? Check `MetalContext.execute()`. If it takes a `throws` closure, the typed throw must propagate through it. If `MetalContext.execute` is generic over its error type, this works automatically. If not, you may need to update `MetalContext.execute` to use `throws(ANNSError)` or `throws(E)`.

**Check `MetalContext.swift` for the `execute` signature** before modifying `NNDescentGPU.swift`. If `execute` currently uses untyped `throws`, you have two options:
1. Make `execute` generic: `func execute<E: Error>(_ work: (MTLCommandBuffer) throws(E) -> Void) async throws(E)`
2. Keep `execute` untyped and wrap at call sites

Option 1 is cleaner. But if `MetalContext.execute` does internal error handling that throws non-`ANNSError`, option 2 is safer. **Read the file and decide.**

### `FullGPUSearch.swift`

- `search(...)` → `throws(ANNSError)` — already only throws `ANNSError`
- Same `context.execute` consideration as `NNDescentGPU`

### Other internal types to check

Grep for `throws` in all `Sources/MetalANNSCore/` files. Any function called (directly or transitively) from a `throws(ANNSError)` function must either:
- Be `throws(ANNSError)` itself, or
- Be `throws(never)` (non-throwing), or
- Be wrapped in `do/catch` at the call site

Key files to check:
- `IncrementalBuilder.swift` — called by `ANNSIndex.insert()`
- `BatchIncrementalBuilder.swift` — called by `ANNSIndex.batchInsert()`
- `IndexCompactor.swift` — called by `ANNSIndex.compact()`
- `GraphPruner.swift` — called by `ANNSIndex.build()`
- `BeamSearchCPU.swift` — called by `ANNSIndex.search()`
- `NNDescentCPU.swift` — called by `ANNSIndex.build()`
- `VectorBuffer.swift`, `Float16VectorBuffer.swift`, `GraphBuffer.swift` — constructors called everywhere
- `KMeans.swift` — called by `ShardedIndex.build()`
- `DiskBackedVectorBuffer.swift` + `DiskBackedIndexLoader` — called by `loadDiskBacked`
- `SoftDeletion.swift`, `MetadataStore.swift`, `MetadataBuffer.swift` — generally non-throwing

**Strategy**: Work bottom-up. Start with leaf functions (no sub-calls that throw), annotate them, then move up the call tree.

### Verify
- Build succeeds
- All tests pass

---

## Task 5: `@concurrent` on Read-Only Actor Methods

### Theory

In Swift 6.2, `@concurrent` on an actor method means callers do NOT need to hop to the actor's executor to call it. The method runs on the caller's executor instead. This is safe IF the method only reads shared state (no mutations).

### Eligibility Analysis for `ANNSIndex`

| Method | Mutates state? | @concurrent? | Reason |
|--------|---------------|-------------|--------|
| `search(query:k:filter:metric:)` | NO — reads `vectors`, `graph`, `idMap`, `softDeletion`, `metadataStore`, `entryPoint`, `configuration` | **YES** | Pure read path |
| `batchSearch(queries:k:filter:metric:)` | NO — calls `search()` in task group | **YES** | Pure read path |
| `rangeSearch(query:maxDistance:...)` | NO — same as search | **YES** | Pure read path |
| `count` (computed property) | NO — reads `idMap.count` and `softDeletion.deletedCount` | **YES** | Pure read |
| `build(vectors:ids:)` | YES — writes `self.vectors`, `self.graph`, etc. | **NO** | Mutates state |
| `insert(_:id:)` | YES | **NO** | Mutates state |
| `batchInsert(_:ids:)` | YES | **NO** | Mutates state |
| `delete(id:)` | YES — writes `softDeletion` | **NO** | Mutates state |
| `compact()` | YES | **NO** | Mutates state |
| `setMetadata(...)` (×3) | YES | **NO** | Mutates state |
| `save(to:)` | NO — reads state | **YES** | Pure read |
| `saveMmapCompatible(to:)` | NO — reads state | **YES** | Pure read |

**WAIT** — `search`, `rangeSearch`, `batchSearch` access `self.context` (a `MetalContext?`). `MetalContext` likely has its own internal state (command queue, etc.). These methods pass `context` to GPU functions but don't mutate `ANNSIndex` state. The `context` object handles its own synchronization. So `@concurrent` is still safe.

**HOWEVER** — there's a subtlety. `@concurrent` methods must not access actor-isolated stored properties unless they're `Sendable` and the access is read-only. In Swift 6.2, `@concurrent` actor methods can read `let` properties and `Sendable` `var` properties. Check that `vectors`, `graph`, `idMap`, etc. are `Sendable`.

Looking at the types:
- `VectorStorage` protocol — check if it's `Sendable`
- `GraphBuffer` — should be `Sendable`
- `IDMap` — is `Codable, Sendable`
- `SoftDeletion` — is `Codable, Sendable`
- `MetadataStore` — is `Codable, Sendable`
- `MetalContext` — check its conformances

**Read the protocol/class definitions before applying `@concurrent`.** If any accessed type is not `Sendable`, the compiler will error. Fix by adding `Sendable` conformance or by not applying `@concurrent` to that method.

### Applying `@concurrent`

**Pattern:**
```swift
// Before:
public func search(query: [Float], k: Int, filter: SearchFilter? = nil, metric: Metric? = nil) async throws(ANNSError) -> [SearchResult] {

// After:
@concurrent
public func search(query: [Float], k: Int, filter: SearchFilter? = nil, metric: Metric? = nil) async throws(ANNSError) -> [SearchResult] {
```

### For `ShardedIndex`

| Method | @concurrent? |
|--------|-------------|
| `search(query:k:filter:metric:)` | **YES** — reads only |
| `count` | **YES** — reads only |
| `build(vectors:ids:)` | **NO** — mutates |

### Verify
- Build succeeds
- All tests pass
- New test: concurrent search from multiple Tasks (proves `@concurrent` enables true parallelism):

```swift
@Test("concurrentSearchFromMultipleTasks")
func concurrentSearchFromMultipleTasks() async throws {
    let vectors = (0..<100).map { i in (0..<16).map { d in sin(Float(i * 16 + d) * 0.173) } }
    let ids = (0..<100).map { "v_\($0)" }

    let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
    try await index.build(vectors: vectors, ids: ids)

    // Launch 10 concurrent searches — @concurrent allows true parallelism
    try await withThrowingTaskGroup(of: [SearchResult].self) { group in
        for i in 0..<10 {
            group.addTask {
                try await index.search(query: vectors[i], k: 5)
            }
        }
        var resultCount = 0
        for try await results in group {
            #expect(!results.isEmpty)
            resultCount += 1
        }
        #expect(resultCount == 10)
    }
}
```

---

## Task 6: InlineArray (Targeted, Optional)

### Theory

`InlineArray<Count, Element>` stores elements inline (on the stack) instead of heap-allocating. Useful when:
- The count is known at compile time
- The array is short-lived (loop iteration, not stored in long-lived data structures)
- The hot path creates/destroys many small arrays

### Where it applies in MetalANNS

The main candidate: **neighbor list iteration in graph traversal**. When iterating a node's neighbors, we often copy a `[UInt32]` slice of `degree` elements. If `degree` is a power-of-two constant (8, 16, 32, 64), `InlineArray` eliminates the heap allocation.

**BUT**: In MetalANNS, `degree` is a runtime configuration value (`IndexConfiguration.degree`). It's not a compile-time constant. `InlineArray` requires a compile-time count.

### Decision: SKIP InlineArray for now

**Reason**: The degree is runtime-configurable (8, 16, 32, or 64). Using `InlineArray` would require either:
1. A generic parameter for degree (major architectural change)
2. Switching over degree values to dispatch to different `InlineArray` sizes (ugly, fragile)
3. Fixing degree at compile time (breaks configurability)

None of these are worth the complexity for Phase 13. **InlineArray may be revisited in Phase 18 (Multi-Queue) if profiling reveals allocation bottlenecks in graph traversal.**

**If the compiler has added `InlineArray` support for runtime-known-but-bounded counts by the time you execute this, adapt accordingly. But do not force it.**

### Verify
- Note in todo.md that InlineArray was evaluated and deferred with rationale

---

## Task 7: Write Phase 13 Tests

Create **one new test file**: `Tests/MetalANNSTests/Swift62ModernizationTests.swift`

### Tests to include:

```swift
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("Swift 6.2 Modernization Tests")
struct Swift62ModernizationTests {

    // --- Typed Throws Verification ---

    @Test("typedThrowCatchesIndexEmpty")
    func typedThrowCatchesIndexEmpty() async {
        let index = ANNSIndex()
        do {
            _ = try await index.search(query: [1.0, 2.0, 3.0], k: 5)
        } catch .indexEmpty {
            // Success: typed throw allows direct pattern match without `as ANNSError`
        } catch {
            Issue.record("Expected .indexEmpty, got \(error)")
        }
    }

    @Test("typedThrowCatchesDimensionMismatch")
    func typedThrowCatchesDimensionMismatch() async throws {
        let vectors = (0..<50).map { i in (0..<8).map { d in Float(i * 8 + d) } }
        let ids = (0..<50).map { "v_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 4, metric: .l2))
        try await index.build(vectors: vectors, ids: ids)

        do {
            _ = try await index.search(query: [1.0], k: 5) // wrong dimension
        } catch .dimensionMismatch(let expected, let got) {
            #expect(expected == 8)
            #expect(got == 1)
        } catch {
            Issue.record("Expected .dimensionMismatch, got \(error)")
        }
    }

    @Test("newErrorCasesExist")
    func newErrorCasesExist() {
        // Compile-time verification that new cases exist
        let e1: ANNSError = .serializationFailed("test")
        let e2: ANNSError = .metalError("test")
        let e3: ANNSError = .invalidArgument("test")
        #expect(e1 is ANNSError)
        #expect(e2 is ANNSError)
        #expect(e3 is ANNSError)
    }

    // --- @concurrent Verification ---

    @Test("concurrentSearchesRunInParallel")
    func concurrentSearchesRunInParallel() async throws {
        let vectors = (0..<100).map { i in (0..<16).map { d in sin(Float(i * 16 + d) * 0.173) } }
        let ids = (0..<100).map { "v_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        // 10 concurrent search tasks — @concurrent enables this without actor hop serialization
        try await withThrowingTaskGroup(of: [SearchResult].self) { group in
            for i in 0..<10 {
                group.addTask {
                    try await index.search(query: vectors[i], k: 5)
                }
            }
            var completedCount = 0
            for try await results in group {
                #expect(!results.isEmpty)
                #expect(results.count == 5)
                completedCount += 1
            }
            #expect(completedCount == 10)
        }
    }

    @Test("concurrentCountAccess")
    func concurrentCountAccess() async throws {
        let vectors = (0..<50).map { i in (0..<8).map { d in Float(i * 8 + d) } }
        let ids = (0..<50).map { "v_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 4, metric: .l2))
        try await index.build(vectors: vectors, ids: ids)

        // Access count from multiple concurrent tasks
        await withTaskGroup(of: Int.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    await index.count
                }
            }
            for await count in group {
                #expect(count == 50)
            }
        }
    }
}
```

### Verify
- All new tests pass
- All existing tests still pass

---

## Task Execution Order

**You MUST execute in this exact order:**

1. **Task 1** → Package.swift bump → build → test → commit
2. **Task 2** → New error cases → build → test → commit
3. **Task 4** → Internal typed throws (bottom-up: leaf functions first) → build → test → DO NOT COMMIT YET
4. **Task 3** → Public API typed throws (now possible because internals are typed) → build → test → commit (Tasks 3+4 together)
5. **Task 5** → `@concurrent` annotations → build → test → commit
6. **Task 6** → InlineArray evaluation → note in todo → no code changes → commit note only
7. **Task 7** → New tests → build → test → commit

**Why this order?** Task 4 must come before Task 3 because `ANNSIndex` calls internal functions — if the internals aren't typed yet, the public methods can't be typed. But they're committed together because they form one logical change.

---

## Success Criteria

Phase 13 is done when ALL of the following are true:

- [ ] `swift-tools-version: 6.2` in Package.swift
- [ ] `ANNSError` has 11 cases (8 original + 3 new)
- [ ] ALL public methods on `ANNSIndex` and `ShardedIndex` that throw use `throws(ANNSError)`
- [ ] ALL internal Core functions in the throw chain use `throws(ANNSError)` or are non-throwing
- [ ] `search()`, `batchSearch()`, `rangeSearch()`, `count`, `save()`, `saveMmapCompatible()` on `ANNSIndex` are `@concurrent`
- [ ] `search()`, `count` on `ShardedIndex` are `@concurrent`
- [ ] `xcodebuild build` succeeds with zero warnings from MetalANNS code
- [ ] `xcodebuild test` passes ALL existing tests (zero regressions)
- [ ] `Swift62ModernizationTests` has 4+ passing tests verifying typed throws and concurrent access
- [ ] `tasks/todo.md` has all items checked and completion signal filled in
- [ ] Git history has 4-5 clean commits for this phase

---

## Anti-Patterns to Avoid

1. **Do NOT use `try!` or `try?` to suppress errors** when mapping throw types. Always explicitly map.
2. **Do NOT use `as! ANNSError` force-casts**. Use `catch let error as ANNSError` or typed catch patterns.
3. **Do NOT add `@concurrent` to any method that writes to actor state**. Even "just one write" is unsafe.
4. **Do NOT change `actor` to `final class`** to avoid concurrency issues. The actor model is intentional.
5. **Do NOT add `nonisolated` to methods that access `self` stored properties** unless they're `Sendable` reads.
6. **Do NOT change the behavior of any existing error path**. Same errors, same conditions, just typed.
7. **Do NOT add `InlineArray` where degree is runtime-determined**. It won't compile.
8. **Do NOT modify Metal shader files (.metal)**. This phase is Swift-only.

---

## Commit Messages

Use these exact messages:

1. `feat(package): bump swift-tools-version to 6.2`
2. `feat(errors): add serializationFailed, metalError, invalidArgument cases to ANNSError`
3. `refactor(throws): add typed throws(ANNSError) to public and internal API surface`
4. `feat(concurrent): add @concurrent to read-only actor methods for parallel search`
5. `test(swift62): add typed throws and concurrent access verification tests`
