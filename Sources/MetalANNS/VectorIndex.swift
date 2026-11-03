import Foundation
import MetalANNSCore

public protocol IndexKey: Hashable, Sendable, LosslessStringConvertible {}
extension String: IndexKey {}
extension UInt64: IndexKey {}

public enum VectorIndexState {
    public enum Unbuilt: Sendable {}
    public enum Ready: Sendable {}
    public enum ReadOnly: Sendable {}
}

public enum ReadOnlyLoadMode: Sendable {
    case mmap
    case diskBacked
}

public struct VectorRecord<Key: IndexKey>: Sendable {
    public let id: Key
    public let vector: [Float]

    public init(id: Key, vector: [Float]) {
        self.id = id
        self.vector = vector
    }
}

/// One search hit.
///
/// Results are ordered nearest first: ascending `distance`, which is the same
/// as descending `similarity`.
///
/// | `metric` | `distance` (lower is closer) | `similarity` (higher is closer) |
/// |---|---|---|
/// | `.cosine` | `1 − cos θ`, in `[0, 2]` | `cos θ`, in `[−1, 1]` |
/// | `.innerProduct` | `−(a · b)` | `a · b` |
/// | `.l2` | squared Euclidean `‖a − b‖²` | `−‖a − b‖²` |
/// | `.hamming` | differing bits | `−(differing bits)` |
public struct VectorNeighbor<Key: IndexKey>: Sendable {
    public let id: Key
    /// Distance under `metric`; lower is closer.
    public let distance: Float
    /// The metric `distance` was computed with (the per-call override, or the
    /// index's configured metric).
    public let metric: Metric
    public let internalID: UInt32

    /// `distance` converted to a score where higher is closer; see the table
    /// on `VectorNeighbor`.
    public var similarity: Float {
        metric.similarity(fromDistance: distance)
    }

    @available(
        *, unavailable,
        message: "score was a distance (lower is closer). Use `distance`, or `similarity` for higher-is-closer."
    )
    public var score: Float { distance }

    public init(id: Key, distance: Float, metric: Metric, internalID: UInt32) {
        self.id = id
        self.distance = distance
        self.metric = metric
        self.internalID = internalID
    }
}

public struct Field<Value>: Sendable {
    public let name: String

    public init(_ name: String) {
        self.name = name
    }
}

public indirect enum QueryFilter: Sendable {
    case any
    case equals(column: String, value: String)
    case greaterThan(column: String, value: Float)
    case lessThan(column: String, value: Float)
    case greaterThanInt(column: String, value: Int64)
    case lessThanInt(column: String, value: Int64)
    case `in`(column: String, values: Set<String>)
    case and([QueryFilter])
    case or([QueryFilter])
    case not(QueryFilter)

    public static func equals(_ field: Field<String>, _ value: String) -> QueryFilter {
        .equals(column: field.name, value: value)
    }

    public static func greaterThan(_ field: Field<Float>, _ value: Float) -> QueryFilter {
        .greaterThan(column: field.name, value: value)
    }

    public static func lessThan(_ field: Field<Float>, _ value: Float) -> QueryFilter {
        .lessThan(column: field.name, value: value)
    }

    public static func greaterThan(_ field: Field<Int64>, _ value: Int64) -> QueryFilter {
        .greaterThanInt(column: field.name, value: value)
    }

    public static func lessThan(_ field: Field<Int64>, _ value: Int64) -> QueryFilter {
        .lessThanInt(column: field.name, value: value)
    }

    public static func oneOf(_ field: Field<String>, _ values: Set<String>) -> QueryFilter {
        .in(column: field.name, values: values)
    }

    fileprivate func toCoreFilter() -> _LegacySearchFilter? {
        switch self {
        case .any:
            return nil
        case .equals(let column, let value):
            return .equals(column: column, value: value)
        case .greaterThan(let column, let value):
            return .greaterThan(column: column, value: value)
        case .lessThan(let column, let value):
            return .lessThan(column: column, value: value)
        case .greaterThanInt(let column, let value):
            return .greaterThanInt(column: column, value: value)
        case .lessThanInt(let column, let value):
            return .lessThanInt(column: column, value: value)
        case .in(let column, let values):
            return .in(column: column, values: values)
        case .and(let filters):
            return .and(filters.compactMap { $0.toCoreFilter() })
        case .or(let filters):
            return .or(filters.compactMap { $0.toCoreFilter() })
        case .not(let filter):
            if let core = filter.toCoreFilter() {
                return .not(core)
            }
            return nil
        }
    }
}

@resultBuilder
public enum QueryFilterBuilder {
    public static func buildBlock(_ components: QueryFilter...) -> QueryFilter {
        collapse(components)
    }

    public static func buildOptional(_ component: QueryFilter?) -> QueryFilter {
        component ?? .any
    }

    public static func buildEither(first component: QueryFilter) -> QueryFilter {
        component
    }

    public static func buildEither(second component: QueryFilter) -> QueryFilter {
        component
    }

    public static func buildArray(_ components: [QueryFilter]) -> QueryFilter {
        collapse(components)
    }

    private static func collapse(_ components: [QueryFilter]) -> QueryFilter {
        let filtered = components.filter {
            if case .any = $0 {
                return false
            }
            return true
        }
        switch filtered.count {
        case 0: return .any
        case 1: return filtered[0]
        default: return .and(filtered)
        }
    }
}

/// Type-safe vector index.
///
/// Concurrency: `VectorIndex` is a `Sendable` value wrapping a `GraphIndex`
/// actor, so copies may be used from any task. Searches run concurrently with
/// each other; mutations (`insert`, `batchInsert`, `compact`) are exclusive and
/// wait for in-flight searches, so a search never observes a half-applied
/// mutation and no mutation is lost to actor re-entrancy.
public struct VectorIndex<Key: IndexKey, State>: Sendable {
    private let rawIndex: GraphIndex
    /// Metric reported on results when a search does not override it.
    private let defaultMetric: Metric

    private init(rawIndex: GraphIndex, defaultMetric: Metric) {
        self.rawIndex = rawIndex
        self.defaultMetric = defaultMetric
    }

    /// Power-user escape hatch that exposes the underlying `GraphIndex` actor.
    ///
    /// > Warning: Accessing `advanced` bypasses the phantom-type state machine.
    /// > The caller is responsible for ensuring operations are valid for the
    /// > current index state (e.g., do not call `search` on an unbuilt index).
    /// > Prefer the typed API (`build`, `search`, `insert`, etc.) whenever possible.
    public var advanced: GraphIndex {
        rawIndex
    }

    fileprivate static func mapNeighbors(_ results: [SearchResult], metric: Metric) -> [VectorNeighbor<Key>] {
        results.compactMap { result in
            let candidate: Key? =
                if !result.id.isEmpty {
                    Key(result.id)
                } else if let numericID = result.numericID {
                    Key(String(numericID))
                } else {
                    nil
                }

            guard let id = candidate else {
                return nil
            }
            return VectorNeighbor(id: id, distance: result.score, metric: metric, internalID: result.internalID)
        }
    }
}

extension VectorIndex where State == VectorIndexState.Unbuilt {
    public init(configuration: IndexConfiguration = .default) {
        self.init(rawIndex: GraphIndex(configuration: configuration), defaultMetric: configuration.metric)
    }

    /// Builds the index. Any non-zero count works: with `count <= degree` the
    /// exact neighbour graph is built, and it fills in as vectors are inserted.
    ///
    /// Throws `ANNSError.nonFiniteVector` for NaN/±inf input and
    /// `CancellationError` when the calling task is cancelled (checked between
    /// construction iterations).
    public consuming func build(records: [VectorRecord<Key>]) async throws -> VectorIndex<Key, VectorIndexState.Ready> {
        guard !records.isEmpty else {
            throw ANNSError.constructionFailed("Cannot build index with empty records")
        }
        let vectors = records.map(\.vector)
        let ids = records.map { String($0.id) }
        try await rawIndex.build(vectors: vectors, ids: ids)
        return VectorIndex<Key, VectorIndexState.Ready>(rawIndex: rawIndex, defaultMetric: defaultMetric)
    }

    public consuming func build(vectors: [[Float]], ids: [Key]) async throws -> VectorIndex<Key, VectorIndexState.Ready>
    {
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        try await rawIndex.build(vectors: vectors, ids: ids.map { String($0) })
        return VectorIndex<Key, VectorIndexState.Ready>(rawIndex: rawIndex, defaultMetric: defaultMetric)
    }

    /// Builds from one row-major buffer of `ids.count × dimension` floats,
    /// without materializing `[[Float]]`. The buffer is copied once, into the
    /// index's vector storage; pass the array you already hold (copy-on-write
    /// means no copy is made on the way in).
    public consuming func build(
        flatVectors: [Float],
        dimension: Int,
        ids: [Key]
    ) async throws -> VectorIndex<Key, VectorIndexState.Ready> {
        try await rawIndex.build(flatVectors: flatVectors, dimension: dimension, ids: ids.map { String($0) })
        return VectorIndex<Key, VectorIndexState.Ready>(rawIndex: rawIndex, defaultMetric: defaultMetric)
    }

    public static func load(from url: URL) async throws -> VectorIndex<Key, VectorIndexState.Ready> {
        let loaded = try await GraphIndex.load(from: url)
        return VectorIndex<Key, VectorIndexState.Ready>(
            rawIndex: loaded,
            defaultMetric: await loaded.configuration.metric
        )
    }

    public static func loadReadOnly(
        from url: URL,
        mode: ReadOnlyLoadMode = .mmap
    ) async throws -> VectorIndex<Key, VectorIndexState.ReadOnly> {
        let loaded: GraphIndex
        switch mode {
        case .mmap:
            loaded = try await GraphIndex.loadMmap(from: url)
        case .diskBacked:
            loaded = try await GraphIndex.loadDiskBacked(from: url)
        }
        return VectorIndex<Key, VectorIndexState.ReadOnly>(
            rawIndex: loaded,
            defaultMetric: await loaded.configuration.metric
        )
    }
}

extension VectorIndex where State == VectorIndexState.Ready {
    public var count: Int {
        get async {
            await rawIndex.count
        }
    }

    /// Inserts a vector. Storage grows as needed (no fixed capacity). An id
    /// that was deleted can be inserted again; it takes its old slot back.
    /// Throws `ANNSError.idAlreadyExists` for a live id.
    public func insert(_ record: VectorRecord<Key>) async throws {
        try await rawIndex.insert(record.vector, id: String(record.id))
    }

    public func batchInsert(_ records: [VectorRecord<Key>]) async throws {
        let vectors = records.map(\.vector)
        let ids = records.map { String($0.id) }
        try await rawIndex.batchInsert(vectors, ids: ids)
    }

    /// Soft-deletes `id`: it stops appearing in results immediately and can be
    /// re-inserted. `compact()` reclaims deleted slots.
    public func delete(id: Key) async throws {
        try await rawIndex.delete(id: String(id))
    }

    /// Rebuilds the graph without deleted vectors and reclaims their slots.
    public func compact() async throws {
        try await rawIndex.compact()
    }

    public func save(to url: URL) async throws {
        try await rawIndex.save(to: url)
    }

    public func saveMmapCompatible(to url: URL) async throws {
        try await rawIndex.saveMmapCompatible(to: url)
    }

    public func setMetadata(_ field: Field<String>, value: String, for id: Key) async throws {
        try await rawIndex.setMetadata(field.name, value: value, for: String(id))
    }

    public func setMetadata(_ field: Field<Float>, value: Float, for id: Key) async throws {
        try await rawIndex.setMetadata(field.name, value: value, for: String(id))
    }

    public func setMetadata(_ field: Field<Int64>, value: Int64, for id: Key) async throws {
        try await rawIndex.setMetadata(field.name, value: value, for: String(id))
    }

    public func search(
        query: [Float],
        topK: Int,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [VectorNeighbor<Key>] {
        let results = try await rawIndex.search(
            query: query,
            k: topK,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return Self.mapNeighbors(results, metric: metric ?? defaultMetric)
    }

    public func batchSearch(
        queries: [[Float]],
        topK: Int,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [[VectorNeighbor<Key>]] {
        let results = try await rawIndex.batchSearch(
            queries: queries,
            k: topK,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return results.map { Self.mapNeighbors($0, metric: metric ?? defaultMetric) }
    }

    /// Returns every vector whose `distance` is at most `maxDistance`
    /// (distance units of the search metric; see `VectorNeighbor`).
    public func rangeSearch(
        query: [Float],
        maxDistance: Float,
        limit: Int = 1000,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [VectorNeighbor<Key>] {
        let results = try await rawIndex.rangeSearch(
            query: query,
            maxDistance: maxDistance,
            limit: limit,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return Self.mapNeighbors(results, metric: metric ?? defaultMetric)
    }
}

extension VectorIndex where State == VectorIndexState.ReadOnly {
    public var count: Int {
        get async {
            await rawIndex.count
        }
    }

    public func save(to url: URL) async throws {
        try await rawIndex.save(to: url)
    }

    public func search(
        query: [Float],
        topK: Int,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [VectorNeighbor<Key>] {
        let results = try await rawIndex.search(
            query: query,
            k: topK,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return Self.mapNeighbors(results, metric: metric ?? defaultMetric)
    }

    public func batchSearch(
        queries: [[Float]],
        topK: Int,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [[VectorNeighbor<Key>]] {
        let results = try await rawIndex.batchSearch(
            queries: queries,
            k: topK,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return results.map { Self.mapNeighbors($0, metric: metric ?? defaultMetric) }
    }

    /// Returns every vector whose `distance` is at most `maxDistance`
    /// (distance units of the search metric; see `VectorNeighbor`).
    public func rangeSearch(
        query: [Float],
        maxDistance: Float,
        limit: Int = 1000,
        metric: Metric? = nil,
        @QueryFilterBuilder filter: () -> QueryFilter = { .any }
    ) async throws -> [VectorNeighbor<Key>] {
        let results = try await rawIndex.rangeSearch(
            query: query,
            maxDistance: maxDistance,
            limit: limit,
            filter: filter().toCoreFilter(),
            metric: metric
        )
        return Self.mapNeighbors(results, metric: metric ?? defaultMetric)
    }
}
