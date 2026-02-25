# Orchestrator Review Protocol

> This document defines how the orchestrator reviews work completed by subagents.

---

## When to Review

Review is triggered when:
1. A subagent fills in the **"Phase 1 Complete — Signal"** section of `tasks/todo.md`
2. The user explicitly asks for a review
3. The orchestrator wants to check intermediate progress

---

## Review Process

### Step 1: Read the Todo

```
Read tasks/todo.md
```

Check:
- Are all items checked `[x]`?
- Are agent notes filled in for decision points (3.6, 5.6)?
- Is the completion signal section filled in?

### Step 2: Verify Git History

```bash
cd /Users/chriskarani/CodingProjects/MetalANNS && git log --oneline
```

Expected: 6 commits with these exact prefixes:
```
chore: initialize MetalANNS Swift package scaffold
feat: add ANNSError, Metric, and IndexConfiguration types
feat: add ComputeBackend protocol with factory and stub backends
feat: implement Accelerate distance kernels (cosine, L2, inner product)
feat: add MetalContext with device lifecycle and PipelineCache
feat: implement Metal distance shaders (cosine, L2, inner product) with GPU tests
```

### Step 3: Run Full Test Suite

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -20
```

Expected: All tests pass, zero failures.

### Step 4: Code Inspection (Parallel Subagents)

Dispatch 4 review subagents in parallel:

**Agent A — Swift Concurrency Review:**
- Grep for `@unchecked Sendable` — should only appear on MTLBuffer wrappers
- Grep for `import XCTest` — should not exist anywhere
- Verify `PipelineCache` is an `actor`
- Verify `MetalContext` is `Sendable`

**Agent B — Metal Shader Review:**
- Read `Distance.metal` — verify 3 kernels with correct buffer indices
- Verify buffer layout matches Swift encoder calls in `MetalBackend`
- Check for common Metal bugs: missing `if (tid >= n) return`, wrong buffer types

**Agent C — AccelerateBackend Review:**
- Read `AccelerateBackend.swift` — verify vDSP usage
- Check zero-norm guard in cosine distance
- Verify L2 returns squared distance (no sqrt)
- Verify inner product returns negated dot

**Agent D — Architecture Review:**
- Verify `Metric` cross-target resolution is clean
- Verify `Bundle.module` used (not parameterless `makeDefaultLibrary`)
- Check no Phase 2+ code leaked in (no VectorBuffer, GraphBuffer, etc.)
- Verify Package.swift has correct structure

### Step 5: Check Orchestrator Review Items

Go through R1–R12 in `tasks/todo.md` and check each one based on findings.

### Step 6: Report

If all R1–R12 pass:
- Mark review items as checked in todo.md
- Report to user: "Phase 1 review passed. Ready for Phase 2."

If any item fails:
- Add `> [REVIEW] FAIL: <reason>` comment under the relevant todo item
- Uncheck `[ ]` the failed item
- Report to user which items failed and why
- Optionally dispatch a fix agent with a targeted prompt

---

## Review Agent Prompt Template

When dispatching review subagents, use this prompt structure:

```
You are reviewing Phase 1 of the MetalANNS project.

TASK: [specific review focus]

CHECK these files:
- [exact file paths]

VERIFY:
- [specific assertions]

REPORT: List any violations found. If none, say "PASS".
```
