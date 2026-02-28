# Phase 19: Streaming Inserts

> **For Claude:** This is an **implementation prompt** for Phase 19 of MetalANNS v4.
> Execute via TDD (RED→GREEN→commit). Dispatch to subagent; orchestrator reviews using
> the R1-R13 checklist in `tasks/todo.md`.

**Goal:** Enable unbounded continuous ingest without capacity errors. A `StreamingIndex`
actor wraps two `ANNSIndex` shards — a frozen *base* and a small *delta* — and merges
them asynchronously when the delta fills up. Search always covers both. HNSW lives only
on the base so it is never invalidated by live inserts.

---

## Current Limitation (Verified from Codebase)

### ANNSIndex.swift — lines 154-156 and line 64

```swift
// Fixed capacity set at build time:
let capacity = max(2, inputVectors.count * 2)       // line 64

// insert() — hard capacity check with no resize path:
let nextInternalID = idMap.count
guard nextInternalID < vectors.capacity, nextInternalID < graph.capacity else {
    throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
}                                                   // lines 169-171

// batchInsert() — same hard check:
guard startSlot + vectors.count <= vectorStorage.capacity,
      startSlot + vectors.count <= graph.capacity else {
    throw ANNSError.constructionFailed("Index capacity exceeded; rebuild with larger capacity")
}                                                   // lines 237-240

// insert() — HNSW invalidated on every insert:
hnsw = nil                                          // line 195
```

There is no auto-resize path. Any caller inserting past the initial capacity ceiling gets
a hard error. The HNSW index is also set to `nil` on every insert, making it unusable for
real-time ingest workloads.

---

## Architecture: Two-Level Merge

```
StreamingIndex actor
  ├── base: ANNSIndex?    — frozen, fully-built, has HNSW (CPU) or GPU graph
  │                          immutable once frozen; safe to read concurrently
  └── delta: ANNSIndex    — small, insert-only, no HNSW (flat search)
       ↓ when delta.count >= deltaCapacity
  MergeTask (background async Task)
       1. Collect base.allVectors + delta.allVectors + all IDs
       2. Build new ANNSIndex from combined set (rebuilds HNSW on completion)
       3. Atomic swap: base = newBase, delta = fresh empty ANNSIndex(config)
```

**Why this design:**
- Search always probes base + delta — no blind spot during merge
- HNSW lives only on base — never invalidated by inserts to delta
- No capacity overflow — delta fills up → merge → delta resets to fresh capacity
- During background merge the old (frozen) base is still queryable
- `flush()` forces a synchronous merge for controlled teardown / testing

---

## System Context

### Key types already in the codebase

```swift
// ANNSIndex (Sources/MetalANNS/ANNSIndex.swift) — do NOT modify
public actor ANNSIndex {
    public func build(vectors: [[Float]], ids: [String]) async throws
    public func insert(_ vector: [Float], id: String) async throws
    public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws
    public func search(query: [Float], k: Int, filter: SearchFilter?, metric: Metric?) async throws -> [SearchResult]
    public func batchSearch(queries: [[Float]], k: Int, filter: SearchFilter?, metric: Metric?) async throws -> [[SearchResult]]
    public func rangeSearch(query: [Float], maxDistance: Float, limit: Int, filter: SearchFilter?, metric: Metric?) async throws -> [SearchResult]
    public func setMetadata(_ column: String, value: String, for id: String) throws
    public func setMetadata(_ column: String, value: Float, for id: String) throws
    public func setMetadata(_ column: String, value: Int64, for id: String) throws
    public func delete(id: String) throws
    public func save(to url: URL) async throws
    public static func load(from url: URL) async throws -> ANNSIndex
    public var count: Int { get }
}

// IndexConfiguration (Sources/MetalANNS/IndexConfiguration.swift) — do NOT modify
public struct IndexConfiguration: Sendable, Codable {
    public var degree: Int          // default 32
    public var metric: Metric       // default .cosine
    public var efConstruction: Int  // default 100
    public var efSearch: Int        // default 64
    public var maxIterations: Int   // default 20
    public var useFloat16: Bool     // default false
    public var convergenceThreshold: Float  // default 0.001
    public var hnswConfiguration: HNSWConfiguration
    public var repairConfiguration: RepairConfiguration
    public static let `default`: IndexConfiguration
}

// ANNSError (Sources/MetalANNS/Errors.swift) — do NOT modify
public enum ANNSError: Error, Sendable {
    case deviceNotSupported
    case dimensionMismatch(expected: Int, got: Int)
    case idAlreadyExists(String)
    case idNotFound(String)
    case corruptFile(String)
    case constructionFailed(String)
    case searchFailed(String)
    case indexEmpty
}
```

### Delta index initialisation pattern

The delta ANNSIndex cannot be `build()`-ed with zero vectors. It needs at least one
vector to be "built". Instead, we use a **lazy-init** pattern:
- Delta starts as `nil`; it becomes an `ANNSIndex` after the first insert calls `build()`.
- Or we use a two-phase approach: accumulate vectors until `build()` threshold, then
  switch the delta to a live insert target.

**Recommended approach** — lazy build:
```
deltaVectors: [[Float]]    — accumulation buffer before first build
deltaIDs: [String]         — parallel ID list
delta: ANNSIndex?          — nil until first batch of >= 1 vector is built
```

When `delta == nil`:
- Append to `deltaVectors` / `deltaIDs`
- When count reaches 1, call `delta = ANNSIndex(config); await delta.build(deltaVectors, deltaIDs)`
- Subsequent inserts call `delta.insert()` until `delta.count >= deltaCapacity`

When `delta` hits capacity → trigger merge.

---

## Tasks

### Task 1: StreamingConfiguration + Tests

**Acceptance**: `StreamingConfigurationTests` passes (3 tests). First git commit.

**Checklist:**

- [ ] 1.1 — Create `Tests/MetalANNSTests/StreamingConfigurationTests.swift` with tests:
  - `defaultValues` — `StreamingConfiguration()` has `deltaCapacity == 10_000`,
    `mergeStrategy == .background`
  - `customValues` — init with `deltaCapacity: 500, mergeStrategy: .blocking`,
    verify stored correctly
  - `codableRoundTrip` — encode to JSON, decode back, verify equality

- [ ] 1.2 — **RED**: Tests fail (`StreamingConfiguration` not defined)

- [ ] 1.3 — Create `Sources/MetalANNS/StreamingConfiguration.swift`:
  ```swift
  import MetalANNSCore

  /// Controls how StreamingIndex merges its delta into the base index.
  public struct StreamingConfiguration: Sendable, Codable, Equatable {
      /// Maximum number of vectors in the delta index before a merge is triggered.
      /// Choose based on acceptable latency spike at merge time; default 10K is safe for
      /// most real-time workloads (≈50ms merge time on Apple Silicon).
      public var deltaCapacity: Int

      /// Strategy for merging delta into the base index.
      public var mergeStrategy: MergeStrategy

      /// Configuration used for both base and delta ANNSIndex instances.
      public var indexConfiguration: IndexConfiguration

      public static let `default` = StreamingConfiguration(
          deltaCapacity: 10_000,
          mergeStrategy: .background,
          indexConfiguration: .default
      )

      public init(
          deltaCapacity: Int = 10_000,
          mergeStrategy: MergeStrategy = .background,
          indexConfiguration: IndexConfiguration = .default
      ) {
          self.deltaCapacity = max(1, deltaCapacity)
          self.mergeStrategy = mergeStrategy
          self.indexConfiguration = indexConfiguration
      }

      /// Determines when and how merges happen.
      public enum MergeStrategy: Sendable, Codable, Equatable {
          /// Merge runs as a detached background Task.
          /// Old base stays queryable during the merge. Search probes both.
          case background

          /// Merge runs inline — `insert()` / `batchInsert()` blocks until merge completes.
          /// Useful for testing and predictable teardown.
          case blocking
      }
  }
  ```

- [ ] 1.4 — **GREEN**: All 3 tests pass

- [ ] 1.5 — **GIT**: `git commit -m "feat: add StreamingConfiguration with MergeStrategy enum"`

---

### Task 2: StreamingIndex Actor Skeleton + Insert + Delta Tracking

**Acceptance**: `StreamingIndexInsertTests` passes (4 tests). Second git commit.

**Checklist:**

- [ ] 2.1 — Create `Tests/MetalANNSTests/StreamingIndexInsertTests.swift` with tests:
  - `insertSingleVector` — create `StreamingIndex`, insert 1 vector, verify `count == 1`
  - `insertBeyondSingleCapacity` — insert 25 vectors into `StreamingIndex` with
    `deltaCapacity: 10` (blocking strategy), verify `count == 25`, no error
  - `batchInsert` — `batchInsert` 50 vectors in one call with `deltaCapacity: 20`
    (blocking), verify `count == 50`
  - `duplicateIDThrows` — insert `id: "a"`, insert `id: "a"` again, verify
    `ANNSError.idAlreadyExists` is thrown

  > Use `dim: 4` vectors for speed. Use `mergeStrategy: .blocking` to keep tests
  > deterministic — no background tasks.

- [ ] 2.2 — **RED**: Tests fail (`StreamingIndex` not defined)

- [ ] 2.3 — Create `Sources/MetalANNS/StreamingIndex.swift`:
  ```swift
  import Foundation
  import MetalANNSCore

  /// A continuous-ingest index that uses a two-level merge architecture.
  ///
  /// New vectors land in a small ``delta`` index. When the delta reaches
  /// ``StreamingConfiguration/deltaCapacity`` it is merged into the frozen
  /// ``base`` index asynchronously (or synchronously for ``.blocking`` strategy).
  /// Search always probes both, so there is no blind spot during a merge.
  public actor StreamingIndex {
      // MARK: - State
      private var base: ANNSIndex?
      private var mergeTask: Task<Void, Error>?
      private var _isMerging: Bool = false

      // Pre-build accumulation buffer (used before delta is first built)
      private var pendingVectors: [[Float]] = []
      private var pendingIDs: [String] = []
      private var delta: ANNSIndex?

      // Track all IDs for duplicate detection across base + delta + pending
      private var allIDs: Set<String> = []

      private let config: StreamingConfiguration

      // MARK: - Init

      public init(config: StreamingConfiguration = .default) {
          self.config = config
      }

      // MARK: - Public API (see tasks below for implementation)

      public var count: Int { get }                             // Task 2
      public var isMerging: Bool { get }                       // Task 6

      public func insert(_ vector: [Float], id: String) async throws  // Task 2
      public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws  // Task 2
  }
  ```

  Implement `count`, `insert`, `batchInsert`:

  **`count`:**
  ```swift
  public var count: Int {
      (base?.count ?? 0) + (delta?.count ?? 0) + pendingVectors.count
  }
  ```

  **`insert(_:id:)` algorithm:**
  1. Check `allIDs.contains(id)` → throw `.idAlreadyExists(id)` if true
  2. Append to `pendingVectors` / `pendingIDs`, insert into `allIDs`
  3. If `delta == nil` and `pendingVectors.count >= 1`:
     - Build delta: `let d = ANNSIndex(config.indexConfiguration); try await d.build(pendingVectors, pendingIDs)`
     - Set `delta = d`, clear `pendingVectors` / `pendingIDs`
  4. Else if `delta != nil`:
     - Move pending → delta via `delta.batchInsert` (if any pending accumulated),
       then `delta.insert(vector, id: id)`
     - Remove from `pendingVectors`/`pendingIDs` since they went into delta
  5. After delta is updated, check `shouldMerge()` → trigger merge if true (see Task 3)

  **`batchInsert(_:ids:)` algorithm:**
  1. Validate `vectors.count == ids.count`, validate no internal duplicates, check
     none of the ids are in `allIDs`
  2. Insert all into `allIDs`
  3. Append to `pendingVectors`/`pendingIDs`
  4. Flush pending → delta (build if delta is nil, batch-insert otherwise)
  5. Check `shouldMerge()` → trigger merge if needed

  **`shouldMerge()`:**
  ```swift
  private func shouldMerge() -> Bool {
      guard let delta else { return false }
      return delta.count >= config.deltaCapacity
  }
  ```

  > **Implementation note:** Building the delta requires `async`, which means
  > `insert()` must be `async throws`. The pending → delta flush is performed
  > inline in `insert()` / `batchInsert()` by `await`-ing the build/insert on
  > the delta `ANNSIndex` actor.

- [ ] 2.4 — **GREEN**: All 4 insert tests pass

- [ ] 2.5 — **GIT**: `git commit -m "feat: StreamingIndex actor skeleton with insert and delta capacity tracking"`

---

### Task 3: Background Merge Task + Atomic Swap

**Acceptance**: `StreamingIndexMergeTests` passes (3 tests). Third git commit.

**Checklist:**

- [ ] 3.1 — Add to `Tests/MetalANNSTests/StreamingIndexInsertTests.swift` (or create
  `StreamingIndexMergeTests.swift`):
  - `mergePreservesAllVectors` — insert `deltaCapacity + 5` vectors (blocking strategy),
    verify `count == deltaCapacity + 5`, then search with all inserted vectors as queries
    and verify each is found in top-1
  - `backgroundMergeTriggered` — insert beyond capacity with `.background` strategy,
    call `flush()`, await completion, verify `count` and `isMerging == false`
  - `mergeClearsIsMerging` — check `isMerging` is true during merge (use `.blocking`,
    capture state just before flush completes), false after

- [ ] 3.2 — **RED**: Tests fail (merge not implemented)

- [ ] 3.3 — Implement merge logic in `StreamingIndex.swift`:

  ```swift
  // MARK: - Merge

  /// Collects all vectors and IDs from base + delta + pending.
  private func collectAll() async throws -> (vectors: [[Float]], ids: [String]) {
      var vectors: [[Float]] = []
      var ids: [String] = []

      if let base {
          // Extract base vectors using its count
          let baseCount = base.count + base_deletedCount  // use idMap.count approach
          // Simpler: rely on search to probe both — but for merge we need raw data.
          // See note below.
      }
      ...
  }
  ```

  > **⚠️ Key Implementation Challenge — extracting vectors from ANNSIndex:**
  > `ANNSIndex` has no public `allVectors` or `allIDs` accessor. The merge
  > must rebuild from scratch. The recommended approach is to **keep a parallel
  > shadow copy of all vectors and IDs in `StreamingIndex` itself**, since it
  > sits above `ANNSIndex`.

  **Revised state model (replace the approach above):**
  ```swift
  private var allVectorsList: [[Float]] = []   // grows with every insert; never shrinks
  private var allIDsList: [String] = []        // parallel to allVectorsList; never shrinks
  private var allIDs: Set<String> = []         // fast duplicate check
  private var deletedIDs: Set<String> = []     // soft-deleted IDs to exclude from merge
  ```

  On `insert(vector, id)`: append to `allVectorsList`, `allIDsList`, insert into `allIDs`.
  On `delete(id)`: insert into `deletedIDs` (handled in Task 5).

  **Merge algorithm:**
  ```swift
  private func triggerMerge() async throws {
      guard let currentDelta = delta, !_isMerging else { return }
      _isMerging = true

      // Collect active (non-deleted) vectors
      var mergeVectors: [[Float]] = []
      var mergeIDs: [String] = []
      for (vector, id) in zip(allVectorsList, allIDsList) {
          guard !deletedIDs.contains(id) else { continue }
          mergeVectors.append(vector)
          mergeIDs.append(id)
      }

      let cfg = config.indexConfiguration
      let newBase = ANNSIndex(configuration: cfg)
      try await newBase.build(vectors: mergeVectors, ids: mergeIDs)

      // Atomic swap
      self.base = newBase
      self.delta = nil        // reset; next insert will rebuild delta lazily
      // pendingVectors / pendingIDs already empty at this point
      _isMerging = false
  }
  ```

  **Calling merge (inside `insert`/`batchInsert` after each update):**
  ```swift
  if shouldMerge() {
      switch config.mergeStrategy {
      case .blocking:
          try await triggerMerge()
      case .background:
          if mergeTask == nil || mergeTask?.isCancelled == true {
              mergeTask = Task { [self] in try await self.triggerMerge() }
              // Note: actor isolation — this Task captures self and runs on actor
          }
      }
  }
  ```

  > **Actor isolation note:** A `Task { [self] in ... }` launched from inside an actor
  > method runs on the actor's executor, so it is safe. The outer `insert()` call
  > returns immediately in `.background` mode while the merge task runs concurrently.

- [ ] 3.4 — Implement `flush()`:
  ```swift
  /// Force-merges all pending delta vectors into the base index.
  /// In `.background` mode this awaits any in-progress merge task first.
  /// Safe to call multiple times.
  public func flush() async throws {
      if let task = mergeTask {
          try await task.value
          mergeTask = nil
      }
      // Merge remaining delta (if any vectors)
      if delta != nil || !pendingVectors.isEmpty {
          try await triggerMerge()
      }
  }
  ```

- [ ] 3.5 — **GREEN**: All 3 merge tests pass

- [ ] 3.6 — **REGRESSION**: `StreamingIndexInsertTests` still pass

- [ ] 3.7 — **GIT**: `git commit -m "feat: StreamingIndex background merge Task with atomic base swap"`

---

### Task 4: Search — Base + Delta Probing

**Acceptance**: `StreamingIndexSearchTests` passes (4 tests). Fourth git commit.

**Checklist:**

- [ ] 4.1 — Create `Tests/MetalANNSTests/StreamingIndexSearchTests.swift` with tests:
  - `searchFindsBaseAndDelta` — build base with 100 vectors (via flush), insert 50 more
    into delta (no flush), search for a known delta vector, verify it appears in top-5
  - `recallAfterMerge` — insert 300 vectors in 3 batches of 100 (blocking, `deltaCapacity: 100`),
    flush, verify recall@10 > 0.90 on 20 random queries drawn from inserted vectors
  - `searchWithFilterForwards` — insert 50 vectors, set `setMetadata("tag", value: "hot", for: id)`
    on 10 of them, search with `filter: .equals("tag", "hot")`, verify only tagged results returned
  - `rangeSearchCoversAll` — insert 100 vectors, flush, rangeSearch with large radius,
    verify result count > 0

- [ ] 4.2 — **RED**: Tests fail (search not implemented)

- [ ] 4.3 — Implement search methods in `StreamingIndex.swift`:

  ```swift
  public func search(
      query: [Float],
      k: Int,
      filter: SearchFilter? = nil,
      metric: Metric? = nil
  ) async throws -> [SearchResult] {
      var results: [SearchResult] = []

      // Probe base (frozen, has HNSW)
      if let base {
          let baseResults = try await base.search(query: query, k: k,
                                                   filter: filter, metric: metric)
          results.append(contentsOf: baseResults)
      }

      // Probe delta (small flat search)
      if let delta {
          let deltaResults = try await delta.search(query: query, k: k,
                                                     filter: filter, metric: metric)
          results.append(contentsOf: deltaResults)
      }

      // Merge, deduplicate by id, sort by score, take top-k
      var seen = Set<String>()
      let deduped = results.filter { seen.insert($0.id).inserted }
      return Array(deduped.sorted { $0.score < $1.score }.prefix(k))
  }

  public func batchSearch(
      queries: [[Float]],
      k: Int,
      filter: SearchFilter? = nil,
      metric: Metric? = nil
  ) async throws -> [[SearchResult]] {
      try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
          for (idx, query) in queries.enumerated() {
              group.addTask { [self] in
                  let r = try await self.search(query: query, k: k,
                                                 filter: filter, metric: metric)
                  return (idx, r)
              }
          }
          var ordered = Array<[SearchResult]?>(repeating: nil, count: queries.count)
          for try await (idx, r) in group { ordered[idx] = r }
          return ordered.map { $0! }
      }
  }

  public func rangeSearch(
      query: [Float],
      radius: Float,
      filter: SearchFilter? = nil,
      metric: Metric? = nil
  ) async throws -> [SearchResult] {
      var results: [SearchResult] = []

      if let base {
          let baseResults = try await base.rangeSearch(
              query: query, maxDistance: radius, limit: Int.max,
              filter: filter, metric: metric)
          results.append(contentsOf: baseResults)
      }
      if let delta {
          let deltaResults = try await delta.rangeSearch(
              query: query, maxDistance: radius, limit: Int.max,
              filter: filter, metric: metric)
          results.append(contentsOf: deltaResults)
      }

      var seen = Set<String>()
      let deduped = results.filter { seen.insert($0.id).inserted }
      return deduped.sorted { $0.score < $1.score }
  }
  ```

- [ ] 4.4 — **GREEN**: All 4 search tests pass

- [ ] 4.5 — **REGRESSION**: Insert tests and merge tests still pass

- [ ] 4.6 — **GIT**: `git commit -m "feat: StreamingIndex search probes base and delta with result merge"`

---

### Task 5: Metadata Forwarding + Delete

**Acceptance**: `StreamingIndexMetadataTests` passes (3 tests). Fifth git commit.

**Checklist:**

- [ ] 5.1 — Add to `Tests/MetalANNSTests/StreamingIndexSearchTests.swift` (or create
  `StreamingIndexMetadataTests.swift`):
  - `setAndGetStringMetadata` — insert vector, `setMetadata("tag", value: "hot", for: id)`,
    call `search` with `filter: .equals("tag", "hot")`, verify result found
  - `metadataPreservedAfterMerge` — insert + set metadata, flush (merge), search with
    filter, verify metadata-tagged result still returned after merge
  - `deleteRemovesFromResults` — insert 10 vectors, delete id `"v5"`, search, verify
    `"v5"` does not appear in any result

- [ ] 5.2 — **RED**: Tests fail (metadata forwarding not implemented)

- [ ] 5.3 — Implement in `StreamingIndex.swift`:

  ```swift
  // MARK: - Metadata

  public func setMetadata(_ column: String, value: String, for id: String) async throws {
      if let internalID = try? await base?.containsID(id), internalID {
          try await base?.setMetadata(column, value: value, for: id)
      } else if let internalID = try? await delta?.containsID(id), internalID {
          try await delta?.setMetadata(column, value: value, for: id)
      } else {
          throw ANNSError.idNotFound(id)
      }
  }
  ```

  > **⚠️ ANNSIndex has no `containsID` method.** Since `StreamingIndex` already
  > tracks `allIDs` and `allIDsList`, it knows which ids exist. To route metadata to
  > the correct shard, track a parallel dictionary:
  > ```swift
  > // Shard membership for routing
  > private var idInBase: Set<String> = []    // ids known to be in base
  > private var idInDelta: Set<String> = []   // ids known to be in delta
  > ```
  > On merge completion, update `idInBase = Set(mergeIDs)`, clear `idInDelta`.
  > After a delta insert, add the id to `idInDelta`.

  Full metadata implementation:
  ```swift
  public func setMetadata(_ column: String, value: String, for id: String) async throws {
      guard allIDs.contains(id) else { throw ANNSError.idNotFound(id) }
      if idInBase.contains(id) { try await base!.setMetadata(column, value: value, for: id) }
      else if idInDelta.contains(id) { try await delta!.setMetadata(column, value: value, for: id) }
      else { throw ANNSError.idNotFound(id) }
  }

  public func setMetadata(_ column: String, value: Float, for id: String) async throws {
      guard allIDs.contains(id) else { throw ANNSError.idNotFound(id) }
      if idInBase.contains(id) { try await base!.setMetadata(column, value: value, for: id) }
      else if idInDelta.contains(id) { try await delta!.setMetadata(column, value: value, for: id) }
      else { throw ANNSError.idNotFound(id) }
  }

  public func setMetadata(_ column: String, value: Int64, for id: String) async throws {
      guard allIDs.contains(id) else { throw ANNSError.idNotFound(id) }
      if idInBase.contains(id) { try await base!.setMetadata(column, value: value, for: id) }
      else if idInDelta.contains(id) { try await delta!.setMetadata(column, value: value, for: id) }
      else { throw ANNSError.idNotFound(id) }
  }
  ```

  **`delete(id:)` implementation:**
  ```swift
  public func delete(id: String) async throws {
      guard allIDs.contains(id) else { throw ANNSError.idNotFound(id) }
      deletedIDs.insert(id)
      // Soft-delete from the relevant shard
      if idInBase.contains(id) { try await base?.delete(id: id) }
      if idInDelta.contains(id) { try await delta?.delete(id: id) }
  }
  ```

- [ ] 5.4 — **GREEN**: All 3 metadata + delete tests pass

- [ ] 5.5 — **REGRESSION**: All previous streaming tests pass

- [ ] 5.6 — **GIT**: `git commit -m "feat: StreamingIndex metadata forwarding and delete routing"`

---

### Task 6: flush() + isMerging State + Concurrent Safety

**Acceptance**: `StreamingIndexFlushTests` passes (3 tests). Sixth git commit.

**Checklist:**

- [ ] 6.1 — Create `Tests/MetalANNSTests/StreamingIndexFlushTests.swift` with tests:
  - `flushMergesAllPending` — insert 50 vectors (deltaCapacity=30, blocking), call
    `flush()`, verify `count == 50` and `isMerging == false`
  - `flushIdempotent` — call `flush()` twice in a row, no crash, `isMerging == false`
  - `concurrentInsertAndSearch` — from 4 concurrent tasks: 2 insert batches of 20
    vectors each, 2 search for queries. Verify no panics and final `count >= 30`
    (some inserts may race with merge but no data loss)

- [ ] 6.2 — **RED**: Tests fail or crash under concurrency

- [ ] 6.3 — Implement `isMerging`:
  ```swift
  public var isMerging: Bool { _isMerging }
  ```

- [ ] 6.4 — Verify `flush()` from Task 3 handles the idempotent case:
  - If `delta == nil && pendingVectors.isEmpty` → return early without building

- [ ] 6.5 — Verify actor isolation prevents data races:
  - All state mutations happen inside `actor StreamingIndex` — Swift's actor model
    guarantees mutual exclusion. No `@unchecked Sendable` needed.
  - Background `Task { [self] in try await self.triggerMerge() }` awaits actor turns;
    it does not bypass isolation.

- [ ] 6.6 — **GREEN**: All 3 flush/concurrency tests pass

- [ ] 6.7 — **REGRESSION**: Full streaming test suite passes

- [ ] 6.8 — **GIT**: `git commit -m "feat: StreamingIndex flush() and isMerging state with concurrent safety"`

---

### Task 7: Persistence — save(to:) + load(from:)

**Acceptance**: `StreamingIndexPersistenceTests` passes (3 tests). Seventh commit.

**Checklist:**

- [ ] 7.1 — Create `Tests/MetalANNSTests/StreamingIndexPersistenceTests.swift` with tests:
  - `saveAndLoadEmpty` — create StreamingIndex, insert 5 vectors, flush, save, load,
    verify `count == 5`
  - `searchAfterLoad` — save + load, run search for a known vector, verify found in top-3
  - `saveRequiresFlush` — save without flushing when delta is non-nil should either
    auto-flush and save OR throw a clear error (`constructionFailed`) — document your
    choice in Task Notes

- [ ] 7.2 — **RED**: Tests fail (save/load not implemented)

- [ ] 7.3 — Implement persistence:

  Persistence format: a directory containing two files:
  - `base.anns` — the base ANNSIndex serialized via `base.save(to:)`
  - `streaming.meta.json` — StreamingConfiguration + list of all allVectorsList + allIDsList
    + deletedIDs (so reload can reconstruct delta state)

  ```swift
  // MARK: - Persistence

  public func save(to url: URL) async throws {
      // Auto-flush delta so we persist a clean merged base
      try await flush()
      guard let base else {
          throw ANNSError.constructionFailed("Nothing to save — index is empty")
      }

      try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
      let baseURL = url.appendingPathComponent("base.anns")
      try await base.save(to: baseURL)

      // Persist config so load() can reconstruct StreamingIndex with same settings
      let meta = PersistedMeta(config: config)
      let metaData = try JSONEncoder().encode(meta)
      try metaData.write(to: url.appendingPathComponent("streaming.meta.json"),
                         options: .atomic)
  }

  public static func load(from url: URL) async throws -> StreamingIndex {
      let metaURL = url.appendingPathComponent("streaming.meta.json")
      let metaData = try Data(contentsOf: metaURL)
      let meta = try JSONDecoder().decode(PersistedMeta.self, from: metaData)

      let streaming = StreamingIndex(config: meta.config)
      let baseURL = url.appendingPathComponent("base.anns")
      let base = try await ANNSIndex.load(from: baseURL)
      await streaming.applyLoadedBase(base)
      return streaming
  }

  // Internal helper — sets base and syncs id tracking
  private func applyLoadedBase(_ loadedBase: ANNSIndex) {
      self.base = loadedBase
      // Re-sync allIDs from base — we don't persist allVectorsList for brevity;
      // this means after load, merge will rebuild from base.allVectors.
      // For full fidelity, also persist allVectorsList in streaming.meta.json.
  }

  private struct PersistedMeta: Codable {
      let config: StreamingConfiguration
  }
  ```

  > **Design choice note for Task Notes 7:** After `load()`, `allVectorsList` is empty
  > (we don't re-serialize all raw vectors since `base.anns` already has them).
  > This means a post-load merge would only merge delta content. This is acceptable for
  > the v4 use case. Document this in Task Notes.

- [ ] 7.4 — **GREEN**: All 3 persistence tests pass

- [ ] 7.5 — **GIT**: `git commit -m "feat: StreamingIndex save/load with auto-flush on save"`

---

### Task 8: Full Suite + Completion Signal

**Acceptance**: Full suite passes. Final commit.

**Checklist:**

- [ ] 8.1 — Build:
  ```
  xcodebuild build -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation
  ```
  → **BUILD SUCCEEDED**

- [ ] 8.2 — Test:
  ```
  xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -skipPackagePluginValidation
  ```
  Expected passing new test suites:
  - `StreamingConfigurationTests` (3 tests)
  - `StreamingIndexInsertTests` (4 tests)
  - `StreamingIndexMergeTests` (3 tests)
  - `StreamingIndexSearchTests` (4 tests)
  - `StreamingIndexMetadataTests` (3 tests)
  - `StreamingIndexFlushTests` (3 tests)
  - `StreamingIndexPersistenceTests` (3 tests)

  All Phase 1-18 tests unchanged. Known `MmapTests` baseline failure allowed.

- [ ] 8.3 — Verify git log shows exactly 8 commits for this phase (including this one)

- [ ] 8.4 — Run the 50K integration scenario manually:
  ```swift
  let si = StreamingIndex(config: StreamingConfiguration(
      deltaCapacity: 10_000,
      mergeStrategy: .blocking
  ))
  for batch in 0..<5 {
      let vecs = (0..<10_000).map { _ in (0..<128).map { _ in Float.random(in: -1...1) } }
      let ids = (0..<10_000).map { "b\(batch)-v\($0)" }
      try await si.batchInsert(vecs, ids: ids)
  }
  assert(await si.count == 50_000)
  ```

- [ ] 8.5 — **GIT**: `git commit -m "chore: phase 19 complete - streaming inserts"`

---

## Task Notes

Use this section to document decisions and issues as you work:

### Task 2 Notes
_(Document your choice between lazy-build pattern vs pre-allocated delta)_

### Task 3 Notes
_(Document the actor isolation approach for the background merge Task)_

### Task 7 Notes
_(Document your save format and allVectorsList persistence decision)_

---

## Success Criteria

✅ `StreamingConfiguration` — `deltaCapacity`, `MergeStrategy.background/blocking`, Codable
✅ `StreamingIndex` actor — insert, batchInsert, search, batchSearch, rangeSearch
✅ Merge — background Task with atomic base swap; blocking mode for tests
✅ Search always probes base + delta — no blind spot during merge
✅ Metadata forwarding — `setMetadata` routed to correct shard; preserved across merges
✅ Delete — soft-deletes from both shards; excluded from merge
✅ `flush()` — forces merge, awaits background task if running
✅ `isMerging` — reflects background merge state
✅ Persistence — `save(to:)` auto-flushes; `load(from:)` reconstructs base
✅ No modifications to `ANNSIndex`, `IndexConfiguration`, or any Phase 1-18 files
✅ No regressions across all Phase 1-18 tests

---

## Anti-Patterns

❌ **Don't** modify `ANNSIndex.swift` — `StreamingIndex` wraps it; it does not inherit from it
❌ **Don't** build the delta with zero vectors — `ANNSIndex.build()` requires `!inputVectors.isEmpty`
❌ **Don't** re-use an `@unchecked Sendable` workaround — `StreamingIndex` is a proper `actor`
❌ **Don't** hold raw `MTLBuffer` references across merge — let `ANNSIndex` manage its own buffers
❌ **Don't** call `hnsw = nil` inside `StreamingIndex` — that's internal to `ANNSIndex`
❌ **Don't** assert hard timing thresholds in any test — background merge timing varies by hardware
❌ **Don't** forget to reset `idInDelta` and set `idInBase` after each merge swap
❌ **Don't** skip the deduplication step in `search()` — a vector can appear in both base and delta
  if it was inserted to delta and a merge copied it to base but delta wasn't immediately cleared
❌ **Don't** call `triggerMerge()` while `_isMerging == true` — guard at the top of `triggerMerge()`

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `Sources/MetalANNS/StreamingConfiguration.swift` | **New** | Config struct + MergeStrategy enum |
| `Sources/MetalANNS/StreamingIndex.swift` | **New** | Two-level merge actor |
| `Tests/MetalANNSTests/StreamingConfigurationTests.swift` | **New** | Config tests |
| `Tests/MetalANNSTests/StreamingIndexInsertTests.swift` | **New** | Insert + delta tests |
| `Tests/MetalANNSTests/StreamingIndexMergeTests.swift` | **New** | Merge correctness tests |
| `Tests/MetalANNSTests/StreamingIndexSearchTests.swift` | **New** | Base+delta search tests |
| `Tests/MetalANNSTests/StreamingIndexMetadataTests.swift` | **New** | Metadata routing + delete tests |
| `Tests/MetalANNSTests/StreamingIndexFlushTests.swift` | **New** | flush() + concurrency tests |
| `Tests/MetalANNSTests/StreamingIndexPersistenceTests.swift` | **New** | save/load round-trip tests |

**No existing files are modified. Total new code: ~600 lines (including tests).**

---

## Commits Expected

1. `feat: add StreamingConfiguration with MergeStrategy enum`
2. `feat: StreamingIndex actor skeleton with insert and delta capacity tracking`
3. `feat: StreamingIndex background merge Task with atomic base swap`
4. `feat: StreamingIndex search probes base and delta with result merge`
5. `feat: StreamingIndex metadata forwarding and delete routing`
6. `feat: StreamingIndex flush() and isMerging state with concurrent safety`
7. `feat: StreamingIndex save/load with auto-flush on save`
8. `chore: phase 19 complete - streaming inserts`
