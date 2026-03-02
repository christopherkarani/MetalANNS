# Lessons Learned

- 2026-02-28: In NN-Descent GPU `local_join`, preserve cross-node pairwise refinement semantics (`a <- b` and `b <- a` with `dist(a,b)`). Refactoring to update only `tid` can silently degrade graph quality and recall.
- 2026-02-28: Protect GPU kernel semantic invariants with deterministic kernel-level regression tests (small crafted graph, single-pass assertion), not only end-to-end recall tests.
- 2026-03-02: Legacy fallback chains must never let one corrupt sidecar block a lower-priority valid source; catch read/decode failures per layer and continue fallback. Also purge deprecated sidecars on successful migrations to prevent stale-state resurrection.
