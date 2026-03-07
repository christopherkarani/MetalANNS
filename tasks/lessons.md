# Lessons Learned

- 2026-02-28: In NN-Descent GPU `local_join`, preserve cross-node pairwise refinement semantics (`a <- b` and `b <- a` with `dist(a,b)`). Refactoring to update only `tid` can silently degrade graph quality and recall.
- 2026-02-28: Protect GPU kernel semantic invariants with deterministic kernel-level regression tests (small crafted graph, single-pass assertion), not only end-to-end recall tests.
- 2026-03-02: Legacy fallback chains must never let one corrupt sidecar block a lower-priority valid source; catch read/decode failures per layer and continue fallback. Also purge deprecated sidecars on successful migrations to prevent stale-state resurrection.
- 2026-03-02: Before adding/refactoring tests, verify called APIs against concrete type definitions (`rg` + source check). In this repo `VectorBuffer` write API is `insert(vector:at:)`; assuming nonexistent helpers like `setVector` causes avoidable compile breaks.
- 2026-03-03: In concurrent GPU kernels, never cache a precheck threshold across inner-loop candidates when lock states can change. For NN-Descent `local_join` early-exit guards, recompute worst-distance bounds per `(a,b)` candidate to avoid skipping valid inserts.
- 2026-03-06: When replacing an ordered frontier with an indexed walk, preserve the original priority-queue semantics. For graph search, new inserts can land ahead of the current cursor; use a heap or true pop-min structure, not a monotonically increasing index into a mutating sorted array.
- 2026-03-06: Reused dedupe/mark arrays in iterative refinement algorithms must be iteration-scoped. If the stamp key omits the outer iteration, later passes can silently suppress valid candidates and stall convergence.
- 2026-03-06: When introducing new shared helper types, verify the helper file is actually included in the patch/worktree diff. A local untracked file can mask missing-file failures that will break CI or code review application.
