# MetalANNS — Phase 4: Beam Search & Query API

> **Status**: COMPLETE
> **Owner**: Subagent (dispatched by orchestrator)
> **Reviewer**: Orchestrator (main session)
> **Last Updated**: 2026-02-25 07:25:59 EAT

---

## How This File Works

This file is the **shared communication layer** between the orchestrator and executing agents.

**Executing agent**: Check off items `[x]` as you complete them. Add notes under any item if you hit issues. Update `Last Updated` timestamp after each task. Do NOT check off an item unless the verification step passes.

**Orchestrator**: Reviews checked items against actual codebase state. Unchecks items that don't pass review. Adds review comments prefixed with `> [REVIEW]`.

---

## Pre-Flight Checks

- [x] Working directory is `/Users/chriskarani/CodingProjects/MetalANNS`
- [x] Phase 1–3 code exists: `NNDescentCPU.swift`, `NNDescentGPU.swift`, `Distance.metal`, `NNDescent.metal`, `Sort.metal` all present
- [x] `git log --oneline` baseline verified (actual was 15 commits before Phase 4 execution)
- [x] Full test suite passes (32 tests, zero failures): `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(passed|failed)'`
- [x] Read implementation plan: `docs/plans/2026-02-25-metalanns-implementation.md` (lines 2371–2513, Tasks 14–15)
- [x] Read this phase's prompt: `docs/prompts/phase-4-search.md` (detailed spec for Task 15)

---

## Task 14: CPU Beam Search (Reference Implementation)

**Acceptance**: `SearchTests` suite passes (2 tests). Fifteenth commit.

- [x] 14.1 — Create `Tests/MetalANNSTests/SearchTests.swift` — 2 tests using Swift Testing:
  - `cpuSearchReturnsK`:
    - Build CPU graph: 100 nodes, dim=16, degree=8, maxIterations=10
    - Search: k=5, ef=32, metric=.cosine
    - Assert: `results.count == k`
    - Assert: results sorted ascending by `score` (`results[i].score >= results[i-1].score`)
  - `cpuSearchRecall`:
    - Build CPU graph: 1000 nodes, dim=32, degree=16, maxIterations=15
    - 20 random queries, k=10, ef=64
    - Compute brute-force top-k via AccelerateBackend
    - Assert: average recall > 0.90
- [x] 14.2 — **RED**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' -only-testing MetalANNSTests/SearchTests 2>&1 | grep -E '(PASS|FAIL|error:)'` → confirms FAIL
- [x] 14.3 — Create `Sources/MetalANNS/SearchResult.swift`:
  - `public struct SearchResult: Sendable`
  - Properties: `id: String`, `score: Float`, `internalID: UInt32`
  - `public init(id:score:internalID:)`
- [x] 14.4 — Create `Sources/MetalANNSCore/BeamSearchCPU.swift`:
  - `public enum BeamSearchCPU` — stateless, all static methods
  - `static func search(query:vectors:graph:entryPoint:k:ef:metric:) async throws -> [SearchResult]`
  - Algorithm:
    1. Compute distance from query to entry point → add to candidates and results
    2. Mark entry point as visited
    3. Loop: pop best unvisited candidate
    4. If candidate's distance > worst in results and results.count >= ef → break
    5. For each neighbor of candidate: if unvisited, compute distance, add to candidates/results if improvement
    6. Keep results sorted, trim to ef size
    7. Return top-k from results
  - Inline distance computation (cosine/l2/innerProduct) — same pattern as NNDescentCPU
  - Results use `id: ""` (empty string) — IDMap integration is Phase 6
  - **DATA STRUCTURE DECISION (14.3)**: Use sorted array for candidates/results. Simple and correct. Document in notes.
- [x] 14.5 — **GREEN**: Both tests pass. Specifically confirm `cpuSearchRecall` achieves > 0.90
- [x] 14.6 — **REGRESSION**: All Phase 1–3 tests still pass (32 prior tests)
- [x] 14.7 — **GIT**: `git add Sources/MetalANNS/SearchResult.swift Sources/MetalANNSCore/BeamSearchCPU.swift Tests/MetalANNSTests/SearchTests.swift && git commit -m "feat: implement CPU beam search with SearchResult type"`

> **Agent notes** _(REQUIRED — document 14.3 data structure decision)_:

- 14.3 data structure decision: used sorted arrays for both candidate and result sets to prioritize correctness and traceability over heap complexity in the CPU reference implementation.
- SearchResult placement adjustment: `SearchResult` was defined in `MetalANNSCore` and re-exported via `Sources/MetalANNS/SearchResult.swift` to avoid circular package-target dependencies while preserving public API access from `MetalANNS`.

---

## Task 15: GPU-Accelerated Beam Search

**Acceptance**: `MetalSearchTests` suite passes (2 tests). Sixteenth commit.

- [x] 15.1 — Create `Tests/MetalANNSTests/MetalSearchTests.swift` — 2 tests:
  - `gpuSearchReturnsK`:
    - Build GPU graph: 100 nodes, dim=16, degree=8
    - Search: k=5, ef=32, metric=.cosine
    - Assert: `results.count == k` and results sorted ascending by score
    - Guard with `guard MTLCreateSystemDefaultDevice() != nil else { return }`
  - `gpuSearchRecall`:
    - Build GPU graph: 500 nodes, dim=32, degree=16, maxIterations=15
    - 10 random queries, k=10, ef=64
    - Compute brute-force top-k via AccelerateBackend
    - Assert: average recall > 0.85
    - Guard with `guard MTLCreateSystemDefaultDevice() != nil else { return }`
- [x] 15.2 — **RED**: Tests fail (`SearchGPU` not defined)
- [x] 15.3 — Create `Sources/MetalANNSCore/SearchGPU.swift`:
  - `public enum SearchGPU` — stateless, all static methods
  - `static func search(context:query:vectors:graph:entryPoint:k:ef:metric:) async throws -> [SearchResult]`
  - **Hybrid CPU/GPU architecture**: CPU manages the beam traversal, GPU handles distance computation batches
  - Implementation:
    1. Compute distance from query to entry point (CPU — single vector not worth GPU dispatch)
    2. Initialize candidates/results, visited set
    3. Loop: pop best unvisited candidate, collect unvisited neighbor IDs
    4. **GPU batch**: extract neighbor vectors, dispatch distance kernel (reuse `cosine_distance`/`l2_distance`/`inner_product_distance` from Distance.metal)
    5. Read back distances, update candidates/results
    6. Return top-k
  - For the GPU dispatch: create temporary MTLBuffers for query and neighbor batch, reuse existing distance kernels via MetalContext
  - Map metric to kernel name: `cosine_distance`, `l2_distance`, `inner_product_distance`
- [x] 15.4 — Optionally create `Sources/MetalANNSCore/Shaders/Search.metal` if you need a custom kernel. If reusing Distance.metal kernels is sufficient, this file is not needed. Document decision.
- [x] 15.5 — **ARCHITECTURE DECISION**: The plan suggests a full threadgroup-per-query GPU approach. The recommended approach for Phase 4 is **hybrid CPU/GPU** (CPU traversal + GPU distance batches). A fully GPU-native search can be added as an optimization later. **Document your architecture choice in the notes below.**
- [x] 15.6 — **KERNEL REUSE DECISION**: Are you reusing the existing Distance.metal kernels or writing new search-specific kernels? Reuse is recommended — they handle all 3 metrics and are already verified. **Document in notes.**
- [x] 15.7 — **GREEN**: Both GPU tests pass. Specifically confirm `gpuSearchRecall` achieves > 0.85
- [x] 15.8 — **REGRESSION**: All prior tests still pass (32 + 2 from Task 14 = 34)
- [x] 15.9 — **FULL SUITE**: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS' 2>&1 | grep -E '(Test Suite|passed|failed)'` → **zero failures**
- [x] 15.10 — **GIT LOG**: `git log --oneline` baseline+2 verified (17 commits total after Phase 4)
- [x] 15.11 — **GIT**: `git add Sources/MetalANNSCore/SearchGPU.swift Tests/MetalANNSTests/MetalSearchTests.swift && git commit -m "feat: implement GPU-accelerated beam search"`
  _(If you created Search.metal, include it in git add)_

> **Agent notes** _(REQUIRED — document 15.5 architecture decision and 15.6 kernel reuse decision)_:

- 15.5 architecture decision: implemented hybrid CPU/GPU beam search (CPU controls traversal and candidate bookkeeping; GPU computes batched neighbor distances), deferring full GPU traversal to a future optimization phase.
- 15.6 kernel decision: reused existing `Distance.metal` kernels (`cosine_distance`, `l2_distance`, `inner_product_distance`) and did not add a new `Search.metal` kernel for Phase 4.
- 15.4 file decision: `Sources/MetalANNSCore/Shaders/Search.metal` was not created because kernel reuse covered all required metric computations.

---

## Phase 4 Complete — Signal

When all items above are checked, update this section:

```
STATUS: COMPLETE
FINAL TEST RESULT: Test run with 36 tests in 14 suites passed after 10.718 seconds.
TOTAL COMMITS: 17
ISSUES ENCOUNTERED:
- Prompt/todo assumed 14->16 commits, but repo baseline was already 15 before Phase 4; executed baseline+2 and recorded 17.
- `SearchResult` placement required dependency-safe adjustment (defined in MetalANNSCore and re-exported from MetalANNS) to avoid circular target dependencies.
DECISIONS MADE:
- 14.3: sorted arrays chosen for beam candidates/results (correctness-first CPU reference).
- 15.5: hybrid CPU/GPU search architecture selected over full GPU traversal.
- 15.6: reused existing `Distance.metal` kernels; no custom `Search.metal` kernel added.
```

---

## Orchestrator Review Checklist (DO NOT MODIFY — Orchestrator use only)

- [ ] R1 — Git log shows exactly 16 commits with correct conventional commit messages
- [ ] R2 — Full test suite passes: `xcodebuild test -scheme MetalANNS-Package -destination 'platform=macOS'` — zero failures including all Phase 1–3 tests
- [ ] R3 — `SearchResult` is a `struct` in `MetalANNS` target, conforms to `Sendable`
- [ ] R4 — `SearchResult` has exactly 3 properties: `id: String`, `score: Float`, `internalID: UInt32`
- [ ] R5 — `BeamSearchCPU` is a stateless `enum` with `static func search(...)`
- [ ] R6 — `SearchGPU` is a stateless `enum` with `static func search(...)`
- [ ] R7 — CPU beam search returns exactly k results sorted by score ascending
- [ ] R8 — CPU beam search uses visited set to prevent re-expansion
- [ ] R9 — CPU recall > 0.90 on 1000 vectors with 20 queries
- [ ] R10 — GPU recall > 0.85 on 500 vectors with 10 queries
- [ ] R11 — GPU tests guarded with `guard MTLCreateSystemDefaultDevice() != nil else { return }`
- [ ] R12 — No `import XCTest` or `XCTSkip` anywhere
- [ ] R13 — No Phase 5+ code leaked in (no Persistence, no ANNSIndex actor, no Incremental Insert)
- [ ] R14 — No Phase 1–3 files were modified (or changes are documented and justified)
- [ ] R15 — Agent notes filled in for Tasks 14.3, 15.5, and 15.6 decisions
- [ ] R16 — `ef` parameter controls beam width (ef >= k), not hardcoded
- [ ] R17 — Beam search correctly terminates (no infinite loops) — early exit when best candidate worse than worst result
