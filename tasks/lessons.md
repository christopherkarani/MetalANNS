# Lessons Learned

- 2026-02-27: Quantized sidecars must be invalidated explicitly on every save path. Persist a sidecar signature in metadata, bind sidecar payloads to that signature, and delete stale `.qhnsw.json` when quantization is unavailable.
