# Phase 6: Algorithmic Optimizations and Bug Fixes

## Role
You are a Swift 6.0 / Metal systems engineer hardening MetalANNS for production. Your work in this phase is four independent fixes: one `StreamingIndex` bug, one test-suite hygiene pass, one GPU kernel optimization, and one PQ safety guard.

---

## Context

### Pre-Conditions
- Phase 5 (`GPUCPUParityTests.swift`) should be complete before Task 6.3 — the parity tests serve as regression coverage for the kernel change.
- All other tasks (6.1, 6.2, 6.4) are independent and can be done in any order.

### What Exists That You Must Know Before Touching Anything

**`Tests/MetalANNSTests/TestUtilities.swift`** — Does NOT exist. Task 6.2 creates it.

**`Tests/MetalANNSTests/StreamingIndexMemoryTests.swift`** — Does NOT exist. The plan references it, but the rangeSearch test belongs in the already-existing `StreamingIndexSearchTests.swift` instead.

**`SeededGenerator` is defined `private struct` in 9 test files** (after Phase 5):
`SearchBufferPoolTests`, `GPUADCSearchTests`, `IVFPQGPUTests`, `IVFPQIndexTests`,
`IVFPQPersistenceTests`, `PQVectorBufferTests`, `ProductQuantizerTests`,
`IVFPQComprehensiveTests`, `GPUCPUParityTests`.
Task 6.2 consolidates all of these into a single `internal` definition in `TestUtilities.swift`.

**`NNDescent.metal` local_join already has symmetric updates** — The plan says "This also adds the symmetric update (b's list gets a as a neighbor), which the current code omits." **This is incorrect.** Both `try_insert_neighbor(a, b, ...)` and `try_insert_neighbor(b, a, ...)` are ALREADY in the current kernel (lines 331–350). Task 6.3 adds only the early-exit guard — `pair_dist < a_worst` / `pair_dist < b_worst` — to skip the CAS for pairs that cannot improve either neighbor list.

**`GPUADCSearch.swift`**: `tableLengthBytes` is computed at line 61. The guard goes there — before buffer allocation, not at line 136 as the plan suggests (line 136 is inside the command encoder closure, which is too late to surface a useful error).

---

## Task 6.1: Fix rangeSearch Returns Empty on Exact Match

**Root cause:** `StreamingIndex.swift:248` has `guard maxDistance > 0 else { return [] }`. A query for `maxDistance: 0.0` hits this guard and returns no results, even when the query vector exists verbatim in the index.

**Files:**
- Fix: `Sources/MetalANNS/StreamingIndex.swift` (line 248)
- Test: `Tests/MetalANNSTests/StreamingIndexSearchTests.swift` (add to existing suite)

### Step 1: Write the failing test

Open `StreamingIndexSearchTests.swift` and add inside the `StreamingIndexSearchTests` struct:

```swift
@Test("rangeSearch with maxDistance 0 returns exact matches")
func rangeSearchZeroDistanceReturnsExactMatch() async throws {
    let index = StreamingIndex(config: StreamingConfiguration(
        deltaCapacity: 50,
        mergeStrategy: .blocking,
        indexConfiguration: IndexConfiguration(degree: 4, metric: .l2)
    ))

    let dim = 8
    var rng = SeededGenerator(state: 55)
    var inserted: [[Float]] = []
    for i in 0..<10 {
        let v = (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        inserted.append(v)
        try await index.insert(v, id: "v-\(i)")
    }

    // maxDistance == 0 should return the vector itself (L2 distance to itself is 0)
    let results = try await index.rangeSearch(query: inserted[3], maxDistance: 0.0, limit: 10)
    #expect(results.count >= 1, "rangeSearch(maxDistance: 0) must return the exact match, not []")
}
```

Note: this test uses `SeededGenerator`. If Task 6.2 (TestUtilities) is not done yet, define it locally as `private struct SeededGenerator` at the top of this file — it's already defined there if the file has other seeded tests. Check first before adding.

### Step 2: Verify the test fails

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing 'MetalANNSTests/StreamingIndexSearchTests/rangeSearchZeroDistanceReturnsExactMatch' \
  2>&1 | tail -20
```

Expected: **FAIL** — returns 0 results because `guard maxDistance > 0` exits early.

### Step 3: Apply the fix

In `Sources/MetalANNS/StreamingIndex.swift` at line 248, change exactly one character:

```swift
// Before (line 248):
guard maxDistance > 0 else {

// After:
guard maxDistance >= 0 else {
```

### Step 4: Verify pass + no regressions

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/StreamingIndexSearchTests 2>&1 | tail -20
```

Expected: **PASS**, including the new test.

### Step 5: Commit

```bash
git add Sources/MetalANNS/StreamingIndex.swift \
        Tests/MetalANNSTests/StreamingIndexSearchTests.swift
git commit -m "fix: allow maxDistance=0 in StreamingIndex.rangeSearch for exact match queries"
```

---

## Task 6.2: Consolidate SeededGenerator into Shared TestUtilities.swift

**Problem:** `SeededGenerator` is copy-pasted as a `private struct` in 9 test files. If the implementation ever changes (e.g., better PRNG), every file needs updating. Centralise it once.

**Files to create:** `Tests/MetalANNSTests/TestUtilities.swift`

**Files to modify:** All 9 test files that have a `private struct SeededGenerator` definition.

### Step 1: Audit which files need updating

```bash
grep -rn "struct SeededGenerator" Tests/MetalANNSTests/
```

Note every file path. You will remove the `private struct SeededGenerator` block from each.

### Step 2: Create TestUtilities.swift

```swift
// Tests/MetalANNSTests/TestUtilities.swift
import Foundation

/// Deterministic XOR-shift PRNG for reproducible test data.
/// Defined once here; all test files in this module use this shared definition.
/// Do NOT declare `private` — it must be `internal` (module-visible) to be shared.
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    init(state: UInt64) {
        // Disallow state=0 (XOR-shift degenerates to all-zeros)
        self.state = state == 0 ? 1 : state
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
```

### Step 3: Remove private definitions from every file

For each file found in Step 1, delete the `private struct SeededGenerator { ... }` block (the full block including the closing brace). The module-level definition in TestUtilities.swift replaces it.

**Do this carefully:** each file has a slightly different private block. Delete only the struct definition — do not touch anything else in the file.

### Step 4: Verify the full suite compiles and passes

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30
```

Expected: **All PASS**. If you see `error: invalid redeclaration of 'SeededGenerator'`, you missed removing a `private struct` definition from one of the files — search again.

### Step 5: Commit

```bash
git add Tests/MetalANNSTests/TestUtilities.swift
# Also stage every file where you removed the private definition
git add Tests/MetalANNSTests/
git commit -m "test: consolidate SeededGenerator into shared TestUtilities.swift"
```

---

## Task 6.3: Early-Exit Guard in local_join Kernel

**Prerequisite:** Phase 5 parity tests must be passing before this task — they are your regression net for kernel changes.

**What already exists:** `NNDescent.metal` lines 317–350 already compute the full symmetric update:
```metal
float pair_dist = compute_metric_distance(vectors, a, b, dim, metric_type);
try_insert_neighbor(adj_ids, adj_dists_bits, a, b, ...);  // forward
try_insert_neighbor(adj_ids, adj_dists_bits, b, a, ...);  // symmetric — ALREADY THERE
```

**What is missing:** The code calls `try_insert_neighbor` (which performs atomic CAS loops) for every `(a, b)` pair even when `pair_dist` cannot improve either list. The fix reads each node's current worst neighbor distance first and skips the CAS when it would certainly fail.

**Files:**
- `Sources/MetalANNSCore/Shaders/NNDescent.metal` (lines 317–350 in the `local_join` kernel)
- `Sources/MetalANNSCore/Shaders/NNDescentFloat16.metal` (mirror the same change)

### Step 1: Read the current local_join inner loop

Read lines 295–360 of `NNDescent.metal` to understand the full context before editing.

### Step 2: Replace the inner loop in NNDescent.metal

Find the `for (uint fi = 0; fi < fwd_count; fi++)` block (around line 317) and replace the body with:

```metal
for (uint fi = 0; fi < fwd_count; fi++) {
    uint a = fwd[fi];
    if (a >= node_count) continue;

    // Read a's current worst neighbor distance before the inner loop.
    // Refreshed after each successful insert to tighten the bound.
    uint a_worst_bits = atomic_load_explicit(
        &adj_dists_bits[a * degree + degree - 1], memory_order_relaxed);
    float a_worst = as_type<float>(a_worst_bits);

    for (uint ri = 0; ri < actual_reverse; ri++) {
        uint b = rev[ri];
        if (b >= node_count || a == b) continue;

        // Read b's current worst distance before computing the pair distance.
        uint b_worst_bits = atomic_load_explicit(
            &adj_dists_bits[b * degree + degree - 1], memory_order_relaxed);
        float b_worst = as_type<float>(b_worst_bits);

        // Skip the expensive distance computation if neither list can benefit.
        if (a_worst <= 0.0f && b_worst <= 0.0f) continue;

        float pair_dist = compute_metric_distance(vectors, a, b, dim, metric_type);

        // Forward: only attempt CAS if pair improves a's list.
        if (pair_dist < a_worst) {
            try_insert_neighbor(adj_ids, adj_dists_bits, a, b,
                                node_count, degree, pair_dist, update_counter);
            // Refresh a_worst so subsequent ri iterations use a tighter bound.
            a_worst_bits = atomic_load_explicit(
                &adj_dists_bits[a * degree + degree - 1], memory_order_relaxed);
            a_worst = as_type<float>(a_worst_bits);
        }

        // Symmetric: only attempt CAS if pair improves b's list.
        if (pair_dist < b_worst) {
            try_insert_neighbor(adj_ids, adj_dists_bits, b, a,
                                node_count, degree, pair_dist, update_counter);
        }
    }
}
```

### Step 3: Mirror the same change in NNDescentFloat16.metal

Find the corresponding `local_join` inner loop in `NNDescentFloat16.metal` and apply the identical guard pattern. The only difference is that `vectors` is `half` typed — distance computation and CAS logic are the same structure.

### Step 4: Run Phase 5 parity tests to verify no regression

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/GPUCPUParityTests 2>&1 | tail -30
```

Expected: **PASS** with average overlap >= 0.60 at all scales. If recall drops below 0.60, the early-exit bound is incorrect — revert and investigate.

### Step 5: Run NNDescent GPU tests to verify recall is maintained or improved

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/NNDescentGPUTests 2>&1 | tail -20
```

Expected: **PASS**. Graph quality should equal or exceed the pre-change baseline.

### Step 6: Commit

```bash
git add Sources/MetalANNSCore/Shaders/NNDescent.metal \
        Sources/MetalANNSCore/Shaders/NNDescentFloat16.metal
git commit -m "perf: add early-exit guards in local_join to skip CAS for non-improving pairs"
```

---

## Task 6.4: PQ Threadgroup Memory Safety Guard

**Root cause:** `GPUADCSearch.swift` calls `scanEncoder.setThreadgroupMemoryLength(tableLengthBytes, index: 0)` without first checking whether `tableLengthBytes` fits within the device's threadgroup memory limit. On devices with a small threadgroup budget (e.g., 16 KB instead of 32 KB), this silently over-allocates and the kernel produces undefined results or crashes.

**Where `tableLengthBytes` is computed:** line 61 of `GPUADCSearch.swift`:
```swift
let tableLengthBytes = m * ks * MemoryLayout<Float>.stride
```

**Files:**
- Fix: `Sources/MetalANNSCore/GPUADCSearch.swift` (add guard after line 61)
- Test: `Tests/MetalANNSTests/GPUADCSearchTests.swift` (add boundary test)

### Step 1: Write a test that validates the guard is checked

Add to `GPUADCSearchTests.swift`:

```swift
@Test("GPUADCSearch rejects table that exceeds device threadgroup memory")
func rejectsDistanceTableExceedingThreadgroupLimit() async throws {
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    guard let context = try? MetalContext() else { return }

    let maxTG = device.maxThreadgroupMemoryLength
    // Construct m/ks values that would require more threadgroup memory than available.
    // m * ks * 4 > maxTG ↔ m * ks > maxTG / 4
    let limit = maxTG / MemoryLayout<Float>.stride
    // Use m=256, ks=256 → 256*256*4 = 262144 bytes (always exceeds any Apple GPU's 32KB limit)
    let m = 256
    let ks = 256
    let tableLengthBytes = m * ks * MemoryLayout<Float>.stride

    guard tableLengthBytes > maxTG else {
        // This device has unusually large threadgroup memory; test is not meaningful.
        print("Skipping: device maxThreadgroupMemoryLength \(maxTG) exceeds table \(tableLengthBytes)")
        return
    }

    // Build a minimal dummy PQ to trigger the code path
    let dummyCodes = Array(repeating: Array(repeating: UInt8(0), count: m), count: 4)
    let dummyCodebook = Array(repeating: Array(repeating: Float(0), count: 1), count: m * ks)
    let dummyQuery = Array(repeating: Float(0), count: m)

    do {
        _ = try await GPUADCSearch.scan(
            context: context,
            query: dummyQuery,
            candidateCodes: dummyCodes,
            codebook: dummyCodebook,
            m: m,
            ks: ks
        )
        Issue.record("Expected GPUADCSearch to throw for oversized table, but it succeeded")
    } catch let error as ANNSError {
        // Expected path — verify the error message is informative
        let msg = error.localizedDescription
        #expect(msg.contains("threadgroup") || msg.contains("table"),
            "Error '\(msg)' should mention threadgroup or table size")
    }
}
```

If `GPUADCSearch.scan(context:query:candidateCodes:codebook:m:ks:)` is not the correct method signature, read `GPUADCSearch.swift` lines 50–65 to find the actual public method name and adjust accordingly.

### Step 2: Add the guard in GPUADCSearch.swift

After line 61 (immediately after `let tableLengthBytes = m * ks * MemoryLayout<Float>.stride`):

```swift
guard tableLengthBytes <= context.device.maxThreadgroupMemoryLength else {
    throw ANNSError.searchFailed(
        "PQ distance table (\(tableLengthBytes) bytes) exceeds device threadgroup memory limit "
        + "(\(context.device.maxThreadgroupMemoryLength) bytes). "
        + "Reduce M (current: \(m)) or Ks (current: \(ks))."
    )
}
```

### Step 3: Run the new test

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' \
  -only-testing MetalANNSTests/GPUADCSearchTests 2>&1 | tail -20
```

Expected: **PASS**.

### Step 4: Run the full suite

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30
```

Expected: All green.

### Step 5: Commit

```bash
git add Sources/MetalANNSCore/GPUADCSearch.swift \
        Tests/MetalANNSTests/GPUADCSearchTests.swift
git commit -m "fix: guard against PQ distance table exceeding device threadgroup memory limit"
```

---

## Definition of Done

All of the following must be true before declaring Phase 6 complete:

- [ ] **6.1**: `rangeSearch(maxDistance: 0.0)` returns the exact-match vector; StreamingIndex.swift line 248 reads `>= 0`
- [ ] **6.2**: `TestUtilities.swift` exists with `internal struct SeededGenerator`; all `private struct SeededGenerator` definitions removed from the 9 individual test files; full suite compiles and passes
- [ ] **6.3**: `NNDescent.metal` and `NNDescentFloat16.metal` `local_join` inner loops have `pair_dist < a_worst` / `pair_dist < b_worst` guards before `try_insert_neighbor`; Phase 5 parity tests still pass at >= 0.60 recall
- [ ] **6.4**: `GPUADCSearch.swift` throws a descriptive `ANNSError.searchFailed` before `setThreadgroupMemoryLength` when the table exceeds `maxThreadgroupMemoryLength`
- [ ] Full `xcodebuild test` suite green with no regressions

---

## Anti-Patterns to Avoid

- **Do not add the PQ guard at line 136** (inside the encoder closure). The guard belongs immediately after `tableLengthBytes` is computed at line 61, so it fails before any buffer allocations or encoder setup.
- **Do not describe 6.3 as "adding symmetric updates."** Symmetric updates (`try_insert_neighbor(b, a, ...)`) are ALREADY in the kernel. Only the early-exit guards are new.
- **Do not put the rangeSearch test in a new `StreamingIndexMemoryTests.swift`.** That file doesn't exist and `StreamingIndexSearchTests.swift` is the correct home for a search correctness test.
- **Do not declare `struct SeededGenerator` as `private` in TestUtilities.swift.** It must be `internal` (the default, no modifier) so every test file in the module can see it.
- **Do not skip the parity-test regression check after 6.3.** A kernel change that lowers GPU-CPU overlap below 0.60 must be investigated, not papered over with a lower threshold.
- **Do not apply 6.3 without reading the current kernel first.** The plan's description of the current code is partially wrong. Always read `NNDescent.metal` lines 295–360 before editing.
