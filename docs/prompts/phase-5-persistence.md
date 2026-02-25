# Phase 5 Execution Prompt: Persistence & Incremental Operations

---

## System Context

You are implementing **Phase 5 (Tasks 16–18)** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phases 1–4 are complete. The codebase has dual compute backends, GPU-resident data structures, CPU and GPU NN-Descent construction, and CPU and GPU beam search. Git log shows 17 commits.

You are building the **persistence and mutation layer** — serializing/deserializing the full index to disk, adding vectors incrementally after initial construction, and soft-deleting vectors from search results. These operations sit between the raw algorithmic primitives (Phases 1–4) and the public actor API (Phase 6).

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/phase5-todo.md`

**Your communication contract:**
1. **`tasks/phase5-todo.md` is your shared state.** Check off `[x]` items as you complete them.
2. **Write notes under every task** — especially for decision points and any issues you hit.
3. **Update `Last Updated`** at the top of phase5-todo.md after each task completes.
4. **When done, fill in the "Phase 5 Complete — Signal" section** at the bottom.
5. **Do NOT modify the "Orchestrator Review Checklist"** section — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **TDD cycle for every task**: Write test → RED → implement → GREEN → commit. Check off RED and GREEN separately.
2. **Swift 6 strict concurrency**: All types `Sendable`. `IndexSerializer` is a stateless enum. `IncrementalBuilder` and `SoftDeletion` should be structs or enums — no classes unless wrapping MTLBuffer.
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest or `XCTSkip`.
4. **Build and test with `xcodebuild`**. Test scheme is `MetalANNS-Package`. Never `swift build` or `swift test`.
5. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
6. **Commit after every task** with the exact conventional commit message specified in the todo.
7. **Check off todo items in real time**.
8. **Do NOT modify Phase 1–4 files** unless strictly necessary. Document any changes in notes.

---

## What Already Exists (Phases 1–4 Output)

### Key Types You'll Use

```swift
// VectorBuffer — GPU-resident vector storage
let vectors = try VectorBuffer(capacity: n, dim: dim, device: context.device)
vectors.buffer          // MTLBuffer — raw bytes for serialization
vectors.dim             // Int
vectors.capacity        // Int
vectors.count           // Int
vectors.insert(vector:at:)
vectors.batchInsert(vectors:startingAt:)
vectors.setCount(_:)
vectors.vector(at:)     // [Float]
vectors.floatPointer    // UnsafeBufferPointer<Float>

// GraphBuffer — GPU-resident adjacency
let graph = try GraphBuffer(capacity: n, degree: degree, device: context.device)
graph.adjacencyBuffer   // MTLBuffer — raw bytes for serialization
graph.distanceBuffer    // MTLBuffer — raw bytes for serialization
graph.degree            // Int
graph.capacity          // Int
graph.nodeCount         // Int
graph.setNeighbors(of:ids:distances:)
graph.neighborIDs(of:)  // [UInt32]
graph.neighborDistances(of:) // [Float]
graph.setCount(_:)

// MetadataBuffer — GPU-accessible index metadata
let meta = try MetadataBuffer()
meta.entryPointID       // UInt32
meta.nodeCount          // UInt32
meta.degree             // UInt32
meta.dim                // UInt32
meta.buffer             // MTLBuffer

// IDMap — bidirectional String↔UInt32 mapping
var idMap = IDMap()      // Sendable, Codable
idMap.assign(externalID:) -> UInt32?
idMap.internalID(for:)  -> UInt32?
idMap.externalID(for:)  -> String?
idMap.count             // Int

// IndexConfiguration
IndexConfiguration.default  // degree=32, metric=.cosine, efConstruction=100, efSearch=64, maxIterations=20, convergenceThreshold=0.001
config.degree
config.metric           // Metric enum (.cosine, .l2, .innerProduct)
config.efSearch

// Metric — Codable, Sendable
Metric.cosine / .l2 / .innerProduct

// Search
BeamSearchCPU.search(query:vectors:graph:entryPoint:k:ef:metric:) -> [SearchResult]
SearchGPU.search(context:query:vectors:graph:entryPoint:k:ef:metric:) -> [SearchResult]

// Construction
NNDescentCPU.build(vectors:degree:metric:maxIterations:convergenceThreshold:) -> (graph: [[(UInt32, Float)]], entryPoint: UInt32)
NNDescentGPU.build(context:vectors:graph:nodeCount:metric:maxIterations:convergenceThreshold:)
```

### Existing Tests (must not regress)
- Phases 1–3: 32 tests
- Phase 4: 4 tests (SearchTests ×2, MetalSearchTests ×2)
- **Total: 36 tests** — all must continue passing

---

## Success Criteria

Phase 5 is done when ALL of the following are true:

- [ ] Index can be saved to file and loaded back, producing identical search results
- [ ] Corrupt file headers are detected with `ANNSError.corruptFile`
- [ ] Vectors can be inserted incrementally after initial construction, with recall degradation < 5%
- [ ] Deleted vectors never appear in search results
- [ ] All new tests pass AND all Phase 1–4 tests still pass (zero regressions)
- [ ] Git history has exactly 20 commits (17 prior + 3 new)
- [ ] `tasks/phase5-todo.md` has all items checked and the completion signal filled in

---

## Execution Instructions

### Before You Start

1. Read `tasks/phase5-todo.md` — this is your checklist.
2. Read `docs/plans/2026-02-25-metalanns-implementation.md` (Tasks 16–18, lines ~2515–2567) — high-level guidance. This prompt provides the detailed spec.
3. Run the full test suite to confirm Phases 1–4 are green: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
4. Complete the **Pre-Flight Checks** in phase5-todo.md.

### For Each Task (16 through 18)

```
1. Read the task's items in tasks/phase5-todo.md
2. Write the test file (check off the "create test" item)
3. Run the test, verify RED (check off the "RED" item)
4. Write the implementation file(s) (check off each file item)
5. Run the test, verify GREEN (check off the "GREEN" item)
6. Run regression check — ALL prior tests still pass
7. Git commit with the specified message (check off the "GIT" item)
8. Update "Last Updated" in phase5-todo.md
9. Write any notes under the task
```

### After All 3 Tasks

1. Run full test suite: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'`
2. Run `git log --oneline` and verify 20 commits
3. Fill in the **"Phase 5 Complete — Signal"** section
4. Do NOT touch the **"Orchestrator Review Checklist"** section

---

## Task-by-Task Reference

### Task 16: Index Serialization

**Purpose**: Save and load the full index to/from a binary file. This enables persistence across app launches.

**Files to create:**
- `Sources/MetalANNSCore/IndexSerializer.swift`
- `Tests/MetalANNSTests/PersistenceTests.swift`

**Binary file format:**
```
HEADER (24 bytes):
  Bytes 0-3:    Magic 'MANN' (4 ASCII bytes: 0x4D, 0x41, 0x4E, 0x4E)
  Bytes 4-7:    version (UInt32, little-endian) — always 1
  Bytes 8-11:   nodeCount (UInt32)
  Bytes 12-15:  degree (UInt32)
  Bytes 16-19:  dim (UInt32)
  Bytes 20-23:  metric (UInt32) — 0=cosine, 1=l2, 2=innerProduct

BODY:
  vectorBuffer raw bytes  (nodeCount * dim * 4 bytes)
  adjacencyBuffer raw bytes  (nodeCount * degree * 4 bytes)
  distanceBuffer raw bytes  (nodeCount * degree * 4 bytes)
  idMap JSON length (UInt32, 4 bytes)
  idMap JSON bytes (variable length)
  entryPointID (UInt32, 4 bytes)
```

**IndexSerializer** (in `MetalANNSCore` target):
- `public enum IndexSerializer` — stateless, all static methods
- `static func save(vectors:graph:idMap:entryPoint:metric:to:) throws`
  - Parameters: `VectorBuffer`, `GraphBuffer`, `IDMap`, `UInt32`, `Metric`, `URL`
  - Writes header, then raw buffer bytes, then JSON-encoded IDMap, then entry point
  - Use `Data` and `FileManager` for I/O
  - Access raw buffer bytes via `MTLBuffer.contents()` + length
- `static func load(from:device:) throws -> (vectors: VectorBuffer, graph: GraphBuffer, idMap: IDMap, entryPoint: UInt32, metric: Metric)`
  - Reads and validates header (magic, version)
  - Allocates VectorBuffer and GraphBuffer with loaded dimensions
  - Copies raw bytes back into MTLBuffers
  - Decodes IDMap from JSON
  - Throws `ANNSError.corruptFile` on invalid magic or version

**Key implementation details:**
- Use `Data(bytes:count:)` to extract raw bytes from `MTLBuffer.contents()`
- Use `buffer.contents().copyMemory(from:byteCount:)` to write bytes back into MTLBuffer on load
- `Metric` mapping to UInt32: cosine=0, l2=1, innerProduct=2 (same as shader convention)
- The vector/adjacency/distance buffer byte lengths are computable from header fields — no need to store them explicitly
- `IDMap` is `Codable` — use `JSONEncoder`/`JSONDecoder`
- `VectorBuffer` and `GraphBuffer` both set their `count`/`nodeCount` after loading via `setCount(_:)`

**Tests — 3 tests:**
1. `saveAndLoadRoundtrip`:
   - Build a small index: 50 nodes, dim=8, degree=4 via NNDescentCPU
   - Create VectorBuffer + GraphBuffer, populate from CPU graph
   - Save to temp file, load back
   - Verify: loaded dimensions match, entry point matches
   - Run search on both original and loaded, verify same top-5 results
2. `corruptMagicThrows`:
   - Write a file with wrong magic bytes ('XXXX' instead of 'MANN')
   - Assert `IndexSerializer.load(from:)` throws `ANNSError.corruptFile`
3. `corruptVersionThrows`:
   - Write valid magic but version=99
   - Assert throws `ANNSError.corruptFile`

**DECISION POINT (16.5)**: The test needs to populate VectorBuffer/GraphBuffer from NNDescentCPU output (which returns `[[(UInt32, Float)]]`). You'll need a helper to transfer CPU graph data into GraphBuffer. You can either: (a) add a helper method on GraphBuffer, (b) do it inline in the test, or (c) create a utility function. **Document your approach.**

**Commit**: `feat: implement index serialization with binary file format`

---

### Task 17: Incremental Insert

**Purpose**: Add new vectors to an existing index without full reconstruction. Uses beam search to find nearest neighbors for the new node, then locally repairs the graph.

**Files to create:**
- `Sources/MetalANNSCore/IncrementalBuilder.swift`
- `Tests/MetalANNSTests/IncrementalTests.swift`

**IncrementalBuilder** (in `MetalANNSCore` target):
- `public enum IncrementalBuilder` — stateless
- `static func insert(vector:at:into:graph:vectors:entryPoint:metric:degree:) throws`
  - Parameters:
    - `vector: [Float]` — the new vector
    - `at internalID: Int` — the internal ID slot to insert at
    - `into graph: GraphBuffer` — the graph to update
    - `vectors: VectorBuffer` — the vector storage (new vector already inserted)
    - `entryPoint: UInt32` — current entry point
    - `metric: Metric`
    - `degree: Int`

**Algorithm — Greedy Insert:**
1. The caller has already inserted the vector into `VectorBuffer` at slot `internalID`
2. Find the `degree` nearest existing nodes to the new vector:
   - Walk the graph from `entryPoint` using a simplified beam search (ef = degree * 2)
   - Collect the top `degree` nearest nodes
3. Set the new node's neighbors to these `degree` nodes (with computed distances)
4. For each of those `degree` neighbors, try to add the new node as a reverse neighbor:
   - If the new node is closer than the neighbor's worst existing neighbor, replace it
   - This maintains bidirectionality
5. No full NN-Descent iteration — just a single greedy insertion

**Key implementation details:**
- Reuse the inline distance function (cosine/l2/innerProduct) — same pattern as BeamSearchCPU
- The simplified beam search can use BeamSearchCPU internals or be reimplemented inline (recommended: inline for simplicity since you're operating on GraphBuffer not `[[(UInt32, Float)]]`)
- For the beam search on GraphBuffer: start at entry point, expand neighbors via `graph.neighborIDs(of:)`, compute distances via inline function, maintain visited set
- For reverse neighbor update: read existing neighbors via `graph.neighborIDs(of:)` and `graph.neighborDistances(of:)`, find worst, replace if new distance is better, write back via `graph.setNeighbors(of:ids:distances:)`

**Tests — 2 tests:**
1. `insertAndFindNew`:
   - Build index with 100 vectors (dim=8, degree=4) via NNDescentCPU
   - Populate VectorBuffer + GraphBuffer
   - Insert 10 new random vectors via IncrementalBuilder
   - For each newly inserted vector, search for it — verify it appears in its own top-1 result (distance ≈ 0)
2. `insertRecallDegradation`:
   - Build index with 200 vectors (dim=16, degree=8) via NNDescentCPU
   - Measure baseline recall (10 queries, k=5, ef=32)
   - Insert 20 new vectors via IncrementalBuilder
   - Measure recall again (10 queries over all 220 vectors)
   - Assert: recall degradation < 5% (i.e., new recall > baseline - 0.05)

**DECISION POINT (17.5)**: The test needs to search on GraphBuffer (not `[[(UInt32, Float)]]`). The existing `BeamSearchCPU` operates on `[[Float]]` vectors and `[[(UInt32, Float)]]` graph. For testing, you can either: (a) extract data from GraphBuffer/VectorBuffer back to arrays, (b) use `SearchGPU.search()` if on a device with GPU, or (c) write a thin adapter. **Document your approach.**

**Commit**: `feat: implement incremental vector insertion with local graph repair`

---

### Task 18: Soft Deletion

**Purpose**: Mark vectors as deleted so they never appear in search results. Deleted nodes remain in the graph structure (they still route traffic) but are filtered from final results.

**Files to create:**
- `Sources/MetalANNSCore/SoftDeletion.swift`
- `Tests/MetalANNSTests/DeletionTests.swift`

**SoftDeletion** (in `MetalANNSCore` target):
- `public struct SoftDeletion: Sendable, Codable`
- `private var deletedIDs: Set<UInt32>`
- `public init()`
- `public mutating func markDeleted(_ internalID: UInt32)`
- `public mutating func undelete(_ internalID: UInt32)` (optional — nice to have)
- `public func isDeleted(_ internalID: UInt32) -> Bool`
- `public var deletedCount: Int`
- `public func filterResults(_ results: [SearchResult]) -> [SearchResult]`
  - Filters out any result whose `internalID` is in the deleted set

**Key implementation details:**
- The deleted set is stored as `Set<UInt32>` — efficient O(1) lookup
- `SoftDeletion` is `Codable` so it can be included in persistence (Phase 6 will handle this)
- `filterResults` is the primary integration point — callers pass search results through it
- Deleted nodes are NOT removed from the graph — they still serve as routing intermediaries during search. Only the final result list is filtered.
- If the caller needs exactly k results after filtering, they should search with a larger ef to compensate for filtered results

**Tests — 3 tests:**
1. `deletedNotInResults`:
   - Build index with 50 vectors (dim=8, degree=4) via NNDescentCPU
   - Mark internal IDs 0, 5, 10 as deleted
   - Run search (various queries), filter results through `SoftDeletion.filterResults()`
   - Assert: none of the deleted IDs appear in results
2. `deletedCountTracking`:
   - Create SoftDeletion, mark 5 IDs as deleted
   - Assert `deletedCount == 5`
   - Mark same ID again — assert `deletedCount` still 5 (idempotent)
3. `undeleteRestores`:
   - Mark ID as deleted, verify `isDeleted` returns true
   - Undelete, verify `isDeleted` returns false

**DECISION POINT (18.4)**: Should `filterResults` request extra results from search to compensate for deletions? This adds coupling. Recommended: keep it simple — `filterResults` just filters, and the caller (ANNSIndex in Phase 6) handles the "search with higher ef" logic. **Document your decision.**

**Commit**: `feat: implement soft deletion with filtered search results`

---

## Decision Points Summary

| # | Decision | Recommended Approach |
|---|----------|---------------------|
| 16.5 | How to populate GraphBuffer from CPU graph data in tests | Inline loop in test or small helper — keep it simple |
| 17.5 | How to search on GraphBuffer in incremental insert tests | Extract data back to arrays and use BeamSearchCPU, or use SearchGPU |
| 18.4 | Should filterResults request extra results to compensate? | No — keep it a pure filter. Caller handles ef adjustment. |

---

## Common Failure Modes (Read Before Starting)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Save/load produces different search results | Buffer byte ordering mismatch | Ensure little-endian consistency; use raw memcpy, not element-wise |
| `corruptFile` not thrown | Magic bytes check wrong | Compare against exact bytes `[0x4D, 0x41, 0x4E, 0x4E]` |
| Loaded VectorBuffer has wrong data | Forgot to set count after loading | Call `vectors.setCount(nodeCount)` and `graph.setCount(nodeCount)` after loading |
| Incremental insert doesn't improve recall | New node's neighbors not connected back | Verify reverse neighbor update — new node must appear in neighbors' lists too |
| Deleted IDs still in results | Filter applied to wrong result set | Ensure `filterResults` is called AFTER search, not before |
| `IDMap` JSON decode fails | UInt32 keys encoded as strings by JSONEncoder | This is expected — `[UInt32: String]` dict has string keys in JSON. Decode handles it. |
| Test needs GPU but runs on CI | GPU search used in tests that should be CPU-only | Use NNDescentCPU + BeamSearchCPU for persistence/incremental/deletion tests |
| Temp file not cleaned up in test | Test creates file but doesn't remove it | Use `FileManager.default.temporaryDirectory` and clean up in test |
| Scheme not found | Xcode scheme naming | Use `MetalANNS-Package` for `xcodebuild test` |

---

## Reference Files

| File | Purpose |
|------|---------|
| `docs/plans/2026-02-25-metalanns-implementation.md` (lines 2515–2567) | High-level guidance for Tasks 16–18 |
| `Sources/MetalANNSCore/VectorBuffer.swift` | `.buffer.contents()` for raw byte access |
| `Sources/MetalANNSCore/GraphBuffer.swift` | `.adjacencyBuffer`, `.distanceBuffer` for serialization |
| `Sources/MetalANNSCore/IDMap.swift` | `Codable` — JSON encode/decode |
| `Sources/MetalANNSCore/BeamSearchCPU.swift` | Search algorithm reference |
| `Sources/MetalANNSCore/NNDescentCPU.swift` | CPU construction for test setup |
| `Sources/MetalANNS/IndexConfiguration.swift` | Configuration struct reference |
| `tasks/phase5-todo.md` | **Your checklist** |
| `tasks/lessons.md` | Record any lessons learned |

---

## Scope Boundary (What NOT To Do)

- Do NOT implement Phase 6 code (ANNSIndex actor, integration test, benchmarks, README)
- Do NOT add hard deletion (removing nodes from the graph structure)
- Do NOT add re-indexing or compaction
- Do NOT modify Phase 1–4 files unless compilation requires it (document any changes)
- Do NOT use XCTest — Swift Testing exclusively
- Do NOT use `swift build` or `swift test` — `xcodebuild` only
- Do NOT create README.md or documentation files
- Do NOT modify the Orchestrator Review Checklist in phase5-todo.md
