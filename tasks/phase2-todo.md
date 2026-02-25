# MetalANNS — Phase 2: GPU-Resident Graph Data Structures

> **Status**: NOT STARTED
> **Owner**: Subagent (dispatched by orchestrator)
> **Reviewer**: Orchestrator (main session)
> **Last Updated**: —

---

## How This File Works

This file is the **shared communication layer** between the orchestrator and executing agents.

**Executing agent**: Check off items `[x]` as you complete them. Add notes under any item if you hit issues. Update `Last Updated` timestamp after each task. Do NOT check off an item unless the verification step passes.

**Orchestrator**: Reviews checked items against actual codebase state. Unchecks items that don't pass review. Adds review comments prefixed with `> [REVIEW]`.

---

## Pre-Flight Checks

- [ ] Working directory is `/Users/chriskarani/CodingProjects/MetalANNS`
- [ ] Phase 1 code exists: `Sources/MetalANNSCore/MetalDevice.swift`, `Sources/MetalANNSCore/Errors.swift`, `Sources/MetalANNSCore/Metric.swift` all present
- [ ] `git log --oneline` shows exactly 6 commits from Phase 1
- [ ] Full test suite passes: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'` → zero failures
- [ ] Read implementation plan: `docs/plans/2026-02-25-metalanns-implementation.md` (lines 1008–1418)

---

## Task 7: VectorBuffer

**Acceptance**: `VectorBufferTests` suite passes (3 tests). Seventh git commit.

- [ ] 7.1 — Create `Tests/MetalANNSTests/VectorBufferTests.swift` — 3 tests using Swift Testing (`@Suite`, `@Test`, `#expect`):
  - `insertSingle` — insert 3-dim vector at index 0, read back, verify equality
  - `batchInsert` — 100 random 128-dim vectors, verify roundtrip within `1e-7` tolerance
  - `countTracking` — verify count starts at 0, tracks after `setCount(1)`
- [ ] 7.2 — **RED**: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/VectorBufferTests 2>&1 | grep -E '(PASS|FAIL|error:)'` → confirms FAIL (VectorBuffer not defined)
- [ ] 7.3 — Create `Sources/MetalANNSCore/VectorBuffer.swift`:
  - `public final class VectorBuffer: @unchecked Sendable`
  - Properties: `buffer: MTLBuffer`, `dim: Int`, `capacity: Int`, `count: Int` (private set)
  - `rawPointer: UnsafeMutablePointer<Float>` (private)
  - `init(capacity:dim:device:)` — allocates `capacity * dim * MemoryLayout<Float>.stride` bytes, `.storageModeShared`, guard `max(byteLength, 4)`
  - `insert(vector:at:)` — validates `vector.count == dim`, copies via `update(from:count:)`
  - `batchInsert(vectors:startingAt:)` — loops over `insert`
  - `vector(at:)` → `[Float]` via `UnsafeBufferPointer`
  - `setCount(_:)` — manual count setter
  - `floatPointer: UnsafeBufferPointer<Float>` — computed property for GPU operations
- [ ] 7.4 — **GREEN**: Same test command → ALL 3 PASS
- [ ] 7.5 — **REGRESSION**: All Phase 1 tests still pass — run full suite
- [ ] 7.6 — **GIT**: `git add Sources/MetalANNSCore/VectorBuffer.swift Tests/MetalANNSTests/VectorBufferTests.swift && git commit -m "feat: add VectorBuffer for GPU-resident vector storage"`

> **Agent notes** _(write issues/decisions here)_:

---

## Task 8: GraphBuffer

**Acceptance**: `GraphBufferTests` suite passes (3 tests). Eighth git commit.

- [ ] 8.1 — Create `Tests/MetalANNSTests/GraphBufferTests.swift` — 3 tests:
  - `setAndReadNeighbors` — set 4 neighbors (IDs + distances) for node 0, read back, verify equality
  - `nodeIndependence` — set different neighbors for node 0 and node 1, verify neither overwrites the other
  - `capacityAndDegree` — verify `.capacity == 100`, `.degree == 32` after init
- [ ] 8.2 — **RED**: Test fails (GraphBuffer not defined)
- [ ] 8.3 — Create `Sources/MetalANNSCore/GraphBuffer.swift`:
  - `public final class GraphBuffer: @unchecked Sendable`
  - TWO MTLBuffers: `adjacencyBuffer` (UInt32) and `distanceBuffer` (Float32)
  - Properties: `degree: Int`, `capacity: Int`, `nodeCount: Int` (private set)
  - `idPointer: UnsafeMutablePointer<UInt32>` and `distPointer: UnsafeMutablePointer<Float>` (private)
  - `init(capacity:degree:device:)` — allocates both buffers, initializes all distances to `Float.greatestFiniteMagnitude` and all IDs to `UInt32.max` (sentinel)
  - `setNeighbors(of:ids:distances:)` — validates `ids.count == degree && distances.count == degree`
  - `neighborIDs(of:)` → `[UInt32]`, `neighborDistances(of:)` → `[Float]`
  - `setCount(_:)` — manual node count setter
- [ ] 8.4 — **GREEN**: All 3 tests pass
- [ ] 8.5 — **REGRESSION**: All Phase 1 + Task 7 tests still pass
- [ ] 8.6 — **INITIALIZATION DECISION**: The plan initializes graph slots in a simple `for` loop. For 100K+ nodes with degree 32, that's 3.2M iterations. Evaluate whether this needs optimization (memset, vDSP.fill) or is acceptable. **Write your decision in the notes below.**
- [ ] 8.7 — **GIT**: `git add Sources/MetalANNSCore/GraphBuffer.swift Tests/MetalANNSTests/GraphBufferTests.swift && git commit -m "feat: add GraphBuffer for GPU-resident adjacency list storage"`

> **Agent notes** _(REQUIRED — document your 8.6 decision here)_:

---

## Task 9: MetadataBuffer and IDMap

**Acceptance**: `MetadataTests` suite passes (3 tests). Ninth git commit.

- [ ] 9.1 — Create `Tests/MetalANNSTests/MetadataTests.swift` — 3 tests:
  - `metadataRoundtrip` — set all 5 fields (entryPointID=42, nodeCount=1000, degree=32, dim=128), read back, verify
  - `idMapMapping` — assign "doc-a" → 0, "doc-b" → 1, verify bidirectional lookup
  - `idMapDuplicate` — assign "doc-a" twice, second returns nil
- [ ] 9.2 — **RED**: Tests fail (types not defined)
- [ ] 9.3 — Create `Sources/MetalANNSCore/MetadataBuffer.swift`:
  - `public final class MetadataBuffer: @unchecked Sendable`
  - Single MTLBuffer, 5 × `MemoryLayout<UInt32>.stride` bytes
  - `pointer: UnsafeMutablePointer<UInt32>` (private)
  - Computed properties for each field: `entryPointID`, `nodeCount`, `degree`, `dim`, `iterationCount`
  - Init zero-fills with `memset`
- [ ] 9.4 — Create `Sources/MetalANNSCore/IDMap.swift`:
  - `public struct IDMap: Sendable, Codable` (struct — naturally Sendable)
  - `externalToInternal: [String: UInt32]` and `internalToExternal: [UInt32: String]` (private)
  - `nextID: UInt32` (private)
  - `mutating func assign(externalID:) -> UInt32?` — returns nil on duplicate
  - `func internalID(for:) -> UInt32?`
  - `func externalID(for:) -> String?`
  - `var count: Int`
- [ ] 9.5 — **ACTOR ISOLATION DECISION**: `IDMap` uses `mutating func`. In Phase 6, `ANNSIndex` actor will store `IDMap` as a var property. Confirm this pattern works with actor isolation (actor-isolated var can call mutating methods). **Write your assessment in the notes below.**
- [ ] 9.6 — **GREEN**: All 3 tests pass
- [ ] 9.7 — **REGRESSION**: All Phase 1 + Phase 2 prior tests still pass
- [ ] 9.8 — **GIT**: `git add Sources/MetalANNSCore/MetadataBuffer.swift Sources/MetalANNSCore/IDMap.swift Tests/MetalANNSTests/MetadataTests.swift && git commit -m "feat: add MetadataBuffer and bidirectional IDMap"`

> **Agent notes** _(REQUIRED — document your 9.5 decision here)_:

---

## Phase 2 Complete — Signal

When all items above are checked, update this section:

```
STATUS: COMPLETE
FINAL TEST RESULT: (paste xcodebuild test summary)
TOTAL COMMITS: (paste git log --oneline)
ISSUES ENCOUNTERED: (list any)
DECISIONS MADE: (list Task 8.6 and 9.5 decisions)
```

---

## Orchestrator Review Checklist (DO NOT MODIFY — Orchestrator use only)

- [ ] R1 — Git log shows exactly 9 commits (6 Phase 1 + 3 Phase 2) with correct conventional commit messages
- [ ] R2 — Full test suite passes: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'` — zero failures including all Phase 1 tests
- [ ] R3 — `VectorBuffer` is `@unchecked Sendable` (wraps MTLBuffer — justified)
- [ ] R4 — `GraphBuffer` is `@unchecked Sendable` with TWO separate MTLBuffers (adjacency + distance)
- [ ] R5 — `GraphBuffer` initializes sentinel values: `Float.greatestFiniteMagnitude` for distances, `UInt32.max` for IDs
- [ ] R6 — `MetadataBuffer` is `@unchecked Sendable` with exactly 5 UInt32 fields at correct offsets
- [ ] R7 — `IDMap` is a **struct** (not class), conforms to `Sendable` and `Codable`
- [ ] R8 — `IDMap.assign` returns `nil` on duplicate (not throwing)
- [ ] R9 — No `import XCTest` anywhere — Swift Testing exclusively
- [ ] R10 — No Phase 3+ code leaked in (no NN-Descent, no BeamSearch, no Sort)
- [ ] R11 — Agent notes filled in for Tasks 8.6 and 9.5 decisions
- [ ] R12 — `VectorBuffer.insert` validates dimension match with `ANNSError.dimensionMismatch`
- [ ] R13 — All buffer allocations use `.storageModeShared` (required for CPU+GPU access)
- [ ] R14 — No Phase 1 files were modified (or changes are documented and justified)
