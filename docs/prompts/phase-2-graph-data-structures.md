# Phase 2 Execution Prompt: GPU-Resident Graph Data Structures

---

## System Context

You are implementing **Phase 2 (Tasks 7–9)** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Phase 1 is complete. The codebase has a working Swift package with dual compute backends (CPU + GPU), verified distance kernels, MetalContext, and PipelineCache. Git log shows 6 clean commits.

You are building the **GPU-resident data structures** that will hold all index state: vectors, graph adjacency, metadata, and ID mapping. These buffers are the foundation that Phase 3 (NN-Descent construction) and Phase 4 (beam search) operate on.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/phase2-todo.md`

**Your communication contract:**
1. **`tasks/phase2-todo.md` is your shared state.** Check off `[x]` items as you complete them. The orchestrator reads this file to track your progress.
2. **Write notes under every task** — especially for decision points (8.6, 9.5) and any issues you hit. The orchestrator reviews your notes.
3. **Update `Last Updated`** at the top of phase2-todo.md after each task completes.
4. **When done, fill in the "Phase 2 Complete — Signal" section** at the bottom of phase2-todo.md. This is how the orchestrator knows you're finished.
5. **Do NOT modify the "Orchestrator Review Checklist"** section at the bottom — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **TDD cycle for every task**: Write test → run to see it fail (RED) → implement → run to see it pass (GREEN) → commit. No exceptions. Check off the RED and GREEN items separately in the todo.
2. **Swift 6 strict concurrency**: All types must be `Sendable`. Use `@unchecked Sendable` ONLY for classes that wrap `MTLBuffer` — these are thread-safe Apple framework types. `IDMap` is a value type (struct) and is naturally `Sendable`.
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest.
4. **Build with `xcodebuild`**, never `swift build` or `swift test`. Metal shaders are not compiled by SPM CLI.
5. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
6. **Commit after every task** with the exact conventional commit message specified in the todo.
7. **Check off todo items in real time** — not at the end. This is how the orchestrator tracks live progress.
8. **Do NOT modify Phase 1 code** unless strictly necessary for compilation. If you need to change a Phase 1 file, document the reason in your notes.

---

## What Already Exists (Phase 1 Output)

Before writing any code, understand the existing codebase. These are the files you'll build on:

### Source Files (MetalANNSCore)
| File | What It Provides |
|------|-----------------|
| `Sources/MetalANNSCore/Metric.swift` | `Metric` enum (`.cosine`, `.l2`, `.innerProduct`) — `Sendable`, `Codable` |
| `Sources/MetalANNSCore/Errors.swift` | `ANNSError` enum with 8 cases — use `constructionFailed(_:)` for buffer allocation failures, `dimensionMismatch(expected:got:)` for vector size errors |
| `Sources/MetalANNSCore/ComputeBackend.swift` | `ComputeBackend` protocol + `BackendFactory` |
| `Sources/MetalANNSCore/AccelerateBackend.swift` | CPU reference implementation of distance computation |
| `Sources/MetalANNSCore/MetalDevice.swift` | `MetalContext` class — provides `device: MTLDevice` for buffer allocation |
| `Sources/MetalANNSCore/MetalBackend.swift` | GPU distance computation — demonstrates buffer allocation patterns |
| `Sources/MetalANNSCore/PipelineCache.swift` | `actor PipelineCache` — shader pipeline caching |

### Key APIs You'll Use
```swift
// Creating a Metal device (needed for buffer allocation)
let device = MTLCreateSystemDefaultDevice()!

// Error types to throw
throw ANNSError.constructionFailed("Failed to allocate VectorBuffer")
throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)

// Buffer allocation pattern (from MetalBackend)
let buffer = device.makeBuffer(length: byteLength, options: .storageModeShared)
```

### Package Structure (Package.swift)
- `MetalANNSCore` — where your new files go (all GPU-resident types live here)
- `MetalANNS` — public API layer (depends on Core)
- `MetalANNSTests` — test target (depends on both)

### Existing Tests (must not regress)
- `PlaceholderTests` — 1 trivial test
- `ConfigurationTests` — 3 tests (ANNSError, Metric, IndexConfiguration)
- `BackendProtocolTests` — factory test
- `DistanceTests` — 8 CPU distance tests
- `MetalDeviceTests` — 2 GPU tests (initContext, pipelineCacheCompile)
- `MetalDistanceTests` — 2 GPU vs CPU comparison tests

---

## Success Criteria

Phase 2 is done when ALL of the following are true:

- [ ] `VectorBuffer` stores and retrieves vectors with Float32 precision, tracks count, validates dimensions
- [ ] `GraphBuffer` stores and retrieves per-node neighbor lists (IDs + distances) with correct isolation
- [ ] `MetadataBuffer` reads/writes 5 UInt32 fields (entryPointID, nodeCount, degree, dim, iterationCount) via MTLBuffer
- [ ] `IDMap` provides bidirectional String↔UInt32 mapping, rejects duplicate external IDs, is `Codable`
- [ ] All new tests pass AND all Phase 1 tests still pass (zero regressions)
- [ ] Git history has exactly 9 commits (6 from Phase 1 + 3 new)
- [ ] `tasks/phase2-todo.md` has all items checked and the completion signal filled in

---

## Execution Instructions

### Before You Start

1. Read `tasks/phase2-todo.md` — this is your checklist. Every item you must do is there.
2. Read `docs/plans/2026-02-25-metalanns-implementation.md` (Tasks 7–9 section, lines ~1008–1418) — this has the **complete code** for every file and test. Use it as your primary reference.
3. Run the full test suite to confirm Phase 1 is green: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
4. Complete the **Pre-Flight Checks** in phase2-todo.md first.

### For Each Task (7 through 9)

Follow this exact loop:

```
1. Read the task's items in tasks/phase2-todo.md
2. Write the test file (check off the "create test" item)
3. Run the test, verify RED (check off the "RED" item)
4. Write the implementation file(s) (check off each file item)
5. Run the test, verify GREEN (check off the "GREEN" item)
6. Run regression check — ALL prior tests still pass (Phase 1 + prior Phase 2)
7. Git commit with the specified message (check off the "GIT" item)
8. Update "Last Updated" in phase2-todo.md header
9. Write any notes under the task in phase2-todo.md
```

### After All 3 Tasks

1. Run full test suite: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
2. Run `git log --oneline` and verify 9 commits (6 Phase 1 + 3 Phase 2)
3. Fill in the **"Phase 2 Complete — Signal"** section in phase2-todo.md
4. Do NOT touch the **"Orchestrator Review Checklist"** section

---

## Task-by-Task Reference

The complete code for every task is in `docs/plans/2026-02-25-metalanns-implementation.md` (lines ~1008–1418). Below is a summary of each task with key gotchas. **Read the plan file for full code.**

### Task 7: VectorBuffer

**Purpose**: GPU-resident flat buffer storing `capacity` vectors of `dim` dimensions. Layout: `vector[i]` starts at offset `i * dim` in the underlying Float32 buffer.

- Create `Sources/MetalANNSCore/VectorBuffer.swift` and `Tests/MetalANNSTests/VectorBufferTests.swift`
- 3 tests: `insertSingle` (insert + readback 3-dim), `batchInsert` (100 random 128-dim, tolerance `1e-7`), `countTracking`
- `@unchecked Sendable` is correct here — wraps `MTLBuffer` which is thread-safe
- Uses `device.makeBuffer(length:options:.storageModeShared)` — shared mode for CPU+GPU access
- `rawPointer` via `buf.contents().bindMemory(to: Float.self, capacity:)` for direct memory access
- **Guard**: `max(byteLength, 4)` in makeBuffer — Metal rejects zero-length buffers
- `insert` validates dimension match, throws `ANNSError.dimensionMismatch`
- `batchInsert` delegates to `insert` in a loop
- `vector(at:)` returns `[Float]` via `UnsafeBufferPointer`
- Expose `floatPointer: UnsafeBufferPointer<Float>` for GPU operations

**Commit**: `feat: add VectorBuffer for GPU-resident vector storage`

### Task 8: GraphBuffer

**Purpose**: GPU-resident adjacency list as flat 2D arrays. Layout: `adjacency[nodeID * degree + slot]` = neighbor UInt32 ID, `distances[nodeID * degree + slot]` = Float32 distance.

- Create `Sources/MetalANNSCore/GraphBuffer.swift` and `Tests/MetalANNSTests/GraphBufferTests.swift`
- 3 tests: `setAndReadNeighbors` (4 neighbors for node 0), `nodeIndependence` (node 0 vs node 1), `capacityAndDegree`
- `@unchecked Sendable` — wraps two `MTLBuffer`s (adjacency + distance)
- **TWO separate MTLBuffers**: `adjacencyBuffer` (UInt32) and `distanceBuffer` (Float32)
- **Initialization**: All distances set to `Float.greatestFiniteMagnitude`, all IDs set to `UInt32.max` (sentinel for empty slots)
- `setNeighbors(of:ids:distances:)` validates that `ids.count == degree && distances.count == degree`
- **DECISION POINT (8.6)**: The plan initializes graph slots in a simple loop. For large graphs (100K+ nodes), this initialization could be slow. You may choose to use `memset` or `vDSP.fill` for better performance, OR keep the simple loop since construction time dominates. **Document your decision in the notes.**

**Commit**: `feat: add GraphBuffer for GPU-resident adjacency list storage`

### Task 9: MetadataBuffer and IDMap

**Purpose**: MetadataBuffer stores 5 UInt32 fields in a single MTLBuffer for GPU-accessible index metadata. IDMap provides bidirectional String↔UInt32 mapping for external IDs.

- Create `Sources/MetalANNSCore/MetadataBuffer.swift`, `Sources/MetalANNSCore/IDMap.swift`, `Tests/MetalANNSTests/MetadataTests.swift`
- 3 tests: `metadataRoundtrip` (set + read all 5 fields), `idMapMapping` (assign + lookup both directions), `idMapDuplicate` (returns nil on duplicate)
- `MetadataBuffer` is `@unchecked Sendable` (wraps MTLBuffer)
- `IDMap` is a **struct** — naturally `Sendable` and `Codable` (will be serialized to disk in Phase 5)
- MetadataBuffer layout: `[entryPointID, nodeCount, degree, dim, iterationCount]` at UInt32 offsets 0–4
- IDMap uses two dictionaries: `externalToInternal: [String: UInt32]` and `internalToExternal: [UInt32: String]`
- `assign(externalID:)` returns `nil` if the ID already exists (not an error throw — callers check)
- **DECISION POINT (9.5)**: `IDMap` uses `mutating func assign`. In Phase 6 (ANNSIndex actor), this struct will be stored as a var property on the actor. Confirm this works cleanly with actor isolation. **Document your assessment in the notes.**

**Commit**: `feat: add MetadataBuffer and bidirectional IDMap`

---

## Decision Points Summary

You MUST make and document these decisions in `tasks/phase2-todo.md` notes:

| # | Decision | Recommended Approach |
|---|----------|---------------------|
| 8.6 | GraphBuffer initialization strategy for large capacity | Simple loop is fine — construction time dominates. But note if you optimize. |
| 9.5 | IDMap struct + mutating func compatibility with actor isolation | Works — actor stores var property, mutating calls are actor-isolated. Confirm. |

---

## Common Failure Modes (Read Before Starting)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `makeBuffer` returns nil | Zero-length buffer requested | Use `max(byteLength, 4)` guard |
| Crash in `bindMemory` | Buffer contents pointer misaligned or capacity wrong | Verify byte calculation: `count * MemoryLayout<T>.stride` |
| `VectorBuffer` data corruption | Off-by-one in pointer arithmetic | `offset = index * dim`, verify `dim` not `dim - 1` |
| `GraphBuffer` wrong neighbor returned | `nodeID * degree + slot` indexing error | Print base offset during debugging |
| `@unchecked Sendable` compiler error | Applied to struct instead of class | Only classes need `@unchecked`; structs are `Sendable` if all stored properties are |
| `ANNSError` not found in `MetalANNSCore` | Error type in wrong target | Verify `Errors.swift` exists in `Sources/MetalANNSCore/` (Phase 1 placed it there) |
| Test can't `@testable import MetalANNSCore` | Test target missing dependency | Verify `MetalANNSTests` depends on `MetalANNSCore` in Package.swift |
| `IDMap` not `Codable` | Custom property types not Codable | Both `[String: UInt32]` and `[UInt32: String]` are Codable — but `UInt32` dict key encodes as String. Verify roundtrip if needed. |

---

## Reference Files

| File | Purpose |
|------|---------|
| `docs/plans/2026-02-25-metalanns-implementation.md` (lines 1008–1418) | **Complete code** for Tasks 7–9 |
| `docs/plans/2026-02-25-metalanns-design.md` | Architecture decisions and rationale |
| `Sources/MetalANNSCore/MetalBackend.swift` | Buffer allocation pattern reference (`.storageModeShared`, `bindMemory`) |
| `Sources/MetalANNSCore/Errors.swift` | Error types to throw (`constructionFailed`, `dimensionMismatch`) |
| `tasks/phase2-todo.md` | **Your checklist** — check items off as you go |
| `tasks/lessons.md` | Record any lessons learned |

---

## Scope Boundary (What NOT To Do)

- Do NOT implement Phase 3+ code (NN-Descent, BeamSearch, Persistence, ANNSIndex)
- Do NOT add features beyond the plan (no FP16 support, no resizable buffers, no GPU-side metadata read)
- Do NOT modify Phase 1 files unless compilation requires it (document any changes)
- Do NOT use XCTest — Swift Testing exclusively
- Do NOT use `swift build` or `swift test` — `xcodebuild` only
- Do NOT create README.md or documentation files
- Do NOT modify the Orchestrator Review Checklist in phase2-todo.md
- Do NOT guess at Metal buffer APIs — the implementation plan has verified code
