# Lessons Learned

- 2026-02-28: In NN-Descent GPU `local_join`, preserve cross-node pairwise refinement semantics (`a <- b` and `b <- a` with `dist(a,b)`). Refactoring to update only `tid` can silently degrade graph quality and recall.
- 2026-02-28: Protect GPU kernel semantic invariants with deterministic kernel-level regression tests (small crafted graph, single-pass assertion), not only end-to-end recall tests.
- 2026-03-02: Legacy fallback chains must never let one corrupt sidecar block a lower-priority valid source; catch read/decode failures per layer and continue fallback. Also purge deprecated sidecars on successful migrations to prevent stale-state resurrection.
- 2026-03-02: Before adding/refactoring tests, verify called APIs against concrete type definitions (`rg` + source check). In this repo `VectorBuffer` write API is `insert(vector:at:)`; assuming nonexistent helpers like `setVector` causes avoidable compile breaks.
- 2026-03-03: In concurrent GPU kernels, never cache a precheck threshold across inner-loop candidates when lock states can change. For NN-Descent `local_join` early-exit guards, recompute worst-distance bounds per `(a,b)` candidate to avoid skipping valid inserts.
