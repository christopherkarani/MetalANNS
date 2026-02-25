# Phase 1 Execution Prompt: MetalANNS Foundation

---

## System Context

You are implementing **Phase 1 (Tasks 1–6)** of the MetalANNS project — a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders.

**Working directory**: `/Users/chriskarani/CodingProjects/MetalANNS`
**Starting state**: Directory contains only `docs/` (plans, prompts) and `tasks/` (todo, lessons). No source code yet.

You are building the foundation layer: package scaffold, type system, dual compute backends (CPU + GPU), and verified distance kernels.

---

## Your Role & Relationship to the Orchestrator

You are a **senior Swift/Metal engineer executing an implementation plan**.

There is an **orchestrator** in a separate session who:
- Wrote this prompt and the implementation plan
- Will review your work after you finish
- Communicates with you through `tasks/todo.md`

**Your communication contract:**
1. **`tasks/todo.md` is your shared state.** Check off `[x]` items as you complete them. The orchestrator reads this file to track your progress.
2. **Write notes under every task** — especially for decision points (3.6, 5.6) and any issues you hit. The orchestrator reviews your notes.
3. **Update `Last Updated`** at the top of todo.md after each task completes.
4. **When done, fill in the "Phase 1 Complete — Signal" section** at the bottom of todo.md. This is how the orchestrator knows you're finished.
5. **Do NOT modify the "Orchestrator Review Checklist"** section at the bottom — that's for the orchestrator only.

---

## Constraints (Non-Negotiable)

1. **TDD cycle for every task**: Write test → run to see it fail (RED) → implement → run to see it pass (GREEN) → commit. No exceptions. Check off the RED and GREEN items separately in the todo.
2. **Swift 6 strict concurrency**: All types must be `Sendable`. Use `actor` where mutation is needed. No `@unchecked Sendable` unless wrapping a thread-safe Apple framework type (`MTLBuffer`, `MTLDevice`).
3. **Swift Testing framework** only (`import Testing`, `@Suite`, `@Test`, `#expect`). Do NOT use XCTest.
4. **Build with `xcodebuild`**, never `swift build` or `swift test`. Metal shaders are not compiled by SPM CLI.
5. **Zero external dependencies**. Only Apple frameworks: Metal, Accelerate, Foundation, OSLog.
6. **Commit after every task** with the exact conventional commit message specified in the todo.
7. **Load Metal library with** `try device.makeDefaultLibrary(bundle: Bundle.module)` — never the parameterless overload.
8. **All Metal atomics use `memory_order_relaxed`** — Metal supports nothing else.
9. **Check off todo items in real time** — not at the end. This is how the orchestrator tracks live progress.

---

## Success Criteria

Phase 1 is done when ALL of the following are true:

- [ ] Swift package compiles with `xcodebuild build` for macOS
- [ ] `AccelerateBackend` passes 8 distance tests (cosine identical/orthogonal, L2 identical/squared, inner product, batch 1000, dim=1, dim=1536)
- [ ] `MetalBackend` GPU distance output matches `AccelerateBackend` CPU output within `1e-4` tolerance for 1000 random 128-dim vectors (cosine + L2)
- [ ] `MetalContext` initializes and `PipelineCache` compiles shader functions from `Bundle.module`
- [ ] Git history has exactly 6 clean commits
- [ ] `tasks/todo.md` has all items checked and the completion signal filled in

---

## Execution Instructions

### Before You Start

1. Read `tasks/todo.md` — this is your checklist. Every item you must do is there.
2. Read `docs/plans/2026-02-25-metalanns-implementation.md` (Tasks 1–6 section) — this has the **complete code** for every file, test, and shader. Use it as your primary reference.
3. Complete the **Pre-Flight Checks** in todo.md first.

### For Each Task (1 through 6)

Follow this exact loop:

```
1. Read the task's items in tasks/todo.md
2. Write the test file (check off the "create test" item)
3. Run the test, verify RED (check off the "RED" item)
4. Write the implementation (check off each file item)
5. Run the test, verify GREEN (check off the "GREEN" item)
6. Run regression check — all prior tests still pass
7. Git commit with the specified message (check off the "GIT" item)
8. Update "Last Updated" in todo.md header
9. Write any notes under the task in todo.md
```

### After All 6 Tasks

1. Run full test suite: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS'`
2. Run `git log --oneline` and verify 6 commits
3. Fill in the **"Phase 1 Complete — Signal"** section in todo.md
4. Do NOT touch the **"Orchestrator Review Checklist"** section

---

## Task-by-Task Reference

The complete code for every task is in `docs/plans/2026-02-25-metalanns-implementation.md`. Below is a summary of each task with key gotchas. **Read the plan file for full code.**

### Task 1: Package Scaffold
- Create `Package.swift`, placeholders, `.gitignore`
- Key: `MetalANNSCore` needs `resources: [.process("Shaders")]` for Bundle.module to work
- Key: The placeholder test must use Swift Testing (`import Testing`, `@Test`), not XCTest
- Verify: `xcodebuild build` succeeds

### Task 2: Error Types and Metric Enum
- Create `Errors.swift`, `IndexConfiguration.swift`, `ConfigurationTests.swift`
- `ANNSError` has 8 cases, `Metric` has 3 cases, `IndexConfiguration` has 7 fields with defaults
- All types must be `Sendable`

### Task 3: Compute Backend Protocol
- Create `ComputeBackend.swift`, stub `AccelerateBackend.swift`, stub `MetalBackend.swift`
- **CRITICAL DECISION (3.6)**: `Metric` is defined in `MetalANNS` but needed in `MetalANNSCore`. You must resolve this cross-target dependency. Recommended: move `Metric` to `MetalANNSCore`, re-export from `MetalANNS`. **Document your decision in todo.md notes.**

### Task 4: Accelerate Distance Kernels
- Replace `fatalError` in `AccelerateBackend` with real vDSP implementations
- 8 tests covering all 3 metrics + edge cases
- Cosine: `1 - dot/(||q||*||v||)`, guard zero-norm with `denom < 1e-10 → 1.0`
- L2: squared Euclidean (no sqrt)
- Inner product: negated dot product

### Task 5: Metal Device & Pipeline Cache
- `MetalContext` is `Sendable` (immutable after init)
- `PipelineCache` is an `actor` (thread-safe lazy compilation)
- Library loaded via `Bundle.module` — the parameterless `makeDefaultLibrary()` only searches the main app bundle and will fail in SPM
- **KNOWN ISSUE (5.6)**: `pipelineCacheCompile` test needs a real kernel in Distance.metal. Document whether you handle this now or defer to Task 6.

### Task 6: Metal Distance Shaders
- Write 3 kernels in `Distance.metal`: `cosine_distance`, `l2_distance`, `inner_product_distance`
- Each kernel: buffer(0)=query, buffer(1)=corpus, buffer(2)=output, buffer(3)=dim(uint), buffer(4)=n(uint)
- Update `MetalBackend` to create context, allocate buffers, encode, dispatch, read back
- GPU vs CPU comparison tests: 1000 random vectors, assert element-wise tolerance
- **FINAL CHECK**: Full suite zero failures, git log shows 6 commits

---

## Decision Points Summary

You MUST make and document these decisions in `tasks/todo.md` notes:

| # | Decision | Recommended Approach |
|---|----------|---------------------|
| 3.6 | Where to define `Metric` for cross-target use | Move to `MetalANNSCore`, re-export from `MetalANNS` |
| 5.6 | How to handle `pipelineCacheCompile` needing a real kernel | Either add minimal kernel to placeholder, or defer test validation to Task 6 |

---

## Common Failure Modes (Read Before Starting)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `no such module 'MetalANNSCore'` | Package.swift target names don't match import | Check spelling in Package.swift targets array |
| `Bundle.module` not found | No `resources:` declaration in target | Add `.process("Shaders")` to MetalANNSCore target |
| Metal shaders not compiled | Used `swift build` instead of `xcodebuild` | Always use `xcodebuild` |
| `makeDefaultLibrary()` returns nil | Used parameterless overload | Use `makeDefaultLibrary(bundle: Bundle.module)` |
| `Metric` not visible in Core | Cross-target dependency | Move Metric to MetalANNSCore (decision 3.6) |
| Tests pass but wrong framework | Used XCTest instead of Swift Testing | `import Testing`, `@Test`, `#expect` — no XCTest |
| GPU test crashes on simulator | Metal not available | Guard with `#if targetEnvironment(simulator)` |

---

## Reference Files

| File | Purpose |
|------|---------|
| `docs/plans/2026-02-25-metalanns-implementation.md` | **Complete code** for every file, test, and shader |
| `docs/plans/2026-02-25-metalanns-design.md` | Architecture decisions and rationale |
| `/Users/chriskarani/Desktop/MetalANNS_Implementation_Plan.md` | Original spec with algorithm details |
| `tasks/todo.md` | **Your checklist** — check items off as you go |
| `tasks/lessons.md` | Record any lessons learned |

---

## Scope Boundary (What NOT To Do)

- Do NOT implement Phase 2+ code (VectorBuffer, GraphBuffer, NN-Descent, Search, Persistence)
- Do NOT add features beyond the plan (no FP16 buffer, no batch distance kernel, no DocC)
- Do NOT use XCTest — Swift Testing exclusively
- Do NOT use `swift build` or `swift test` — `xcodebuild` only
- Do NOT create README.md or documentation files
- Do NOT modify the Orchestrator Review Checklist in todo.md
- Do NOT guess at Metal APIs — the implementation plan has verified code
