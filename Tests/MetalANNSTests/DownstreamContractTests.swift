import Foundation
import Testing

@testable import MetalANNS
@testable import MetalANNSCore

/// Contracts the main downstream user (Wax) relies on. Each section started as
/// a failing reproduction against 0.3.0.
@Suite("Downstream contracts")
struct DownstreamContractTests {
    private typealias Unbuilt = VectorIndex<UInt64, VectorIndexState.Unbuilt>
    private typealias Ready = VectorIndex<UInt64, VectorIndexState.Ready>

    private static func ids(_ range: Range<Int>) -> [UInt64] { range.map(UInt64.init) }

    private static func build(
        _ vectors: [[Float]],
        configuration: IndexConfiguration = IndexConfiguration(degree: 8)
    ) async throws -> Ready {
        try await Unbuilt(configuration: configuration).build(vectors: vectors, ids: ids(0..<vectors.count))
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Double {
        zip(a, b).reduce(0) { $0 + Double($1.0) * Double($1.1) }
    }

    /// Brute-force similarity (higher is closer) in float64.
    private static func referenceSimilarity(_ a: [Float], _ b: [Float], metric: Metric) -> Double {
        switch metric {
        case .cosine: dot(a, b) / (dot(a, a).squareRoot() * dot(b, b).squareRoot())
        case .innerProduct: dot(a, b)
        case .l2: -zip(a, b).reduce(0) { $0 + Double($1.0 - $1.1) * Double($1.0 - $1.1) }
        case .hamming: 0
        }
    }

    private static func expectANNSError(
        _ matches: (ANNSError) -> Bool,
        _ body: () async throws -> Void,
        sourceLocation: SourceLocation = #_sourceLocation
    ) async {
        do {
            try await body()
            Issue.record("expected an ANNSError, none thrown", sourceLocation: sourceLocation)
        } catch let error as ANNSError {
            #expect(matches(error), "unexpected \(error)", sourceLocation: sourceLocation)
        } catch {
            Issue.record("expected ANNSError, got \(error)", sourceLocation: sourceLocation)
        }
    }

    private static func isIDAlreadyExists(_ error: ANNSError) -> Bool {
        if case .idAlreadyExists = error { return true }
        return false
    }

    private static func isConstructionFailed(_ error: ANNSError) -> Bool {
        if case .constructionFailed = error { return true }
        return false
    }

    private static func isNonFinite(at index: Int) -> (ANNSError) -> Bool {
        { error in
            if case .nonFiniteVector(let reported) = error { return reported == index }
            return false
        }
    }

    // MARK: 1. Score semantics

    @Test("distance and similarity follow the documented convention", arguments: [Metric.cosine, .innerProduct, .l2])
    func scoreSemantics(metric: MetalANNSCore.Metric) async throws {
        var rng = SeededGenerator(state: 5)
        let vectors: [[Float]] = (0..<200).map { _ in
            let scale = Float.random(in: 0.5...2, using: &rng)
            return (0..<16).map { _ in Float.random(in: -1...1, using: &rng) * scale }
        }
        for exactLimit in [IndexConfiguration.defaultExactSearchMaxVectorCount, 0] {
            var configuration = IndexConfiguration(degree: 8, metric: metric)
            configuration.exactSearchMaxVectorCount = exactLimit
            let index = try await Self.build(vectors, configuration: configuration)
            let query = vectors[17]
            let hits = try await index.search(query: query, topK: 10)
            #expect(hits.count == 10)
            #expect(hits.allSatisfy { $0.metric == metric })
            #expect(zip(hits, hits.dropFirst()).allSatisfy { $0.distance <= $1.distance })
            #expect(zip(hits, hits.dropFirst()).allSatisfy { $0.similarity >= $1.similarity })
            for hit in hits {
                let reference = Self.referenceSimilarity(query, vectors[Int(hit.id)], metric: metric)
                #expect(abs(Double(hit.similarity) - reference) < 1e-3 * max(1, abs(reference)))
            }
            if metric == .cosine {
                #expect(hits[0].id == 17)
                #expect(abs(hits[0].similarity - 1) < 1e-5)
                #expect(abs(hits[0].distance) < 1e-5)
            }
        }
    }

    @Test("a per-call metric override is reported on the hits")
    func metricOverrideIsReported() async throws {
        let vectors = seededVectors(count: 50, dim: 8)
        let index = try await Self.build(vectors)
        let hits = try await index.search(query: vectors[0], topK: 3, metric: .l2)
        #expect(hits.allSatisfy { $0.metric == .l2 })
        #expect(hits[0].similarity == -hits[0].distance)
    }

    // MARK: 2. Re-insert after delete

    @Test("a deleted id can be inserted again and is found by its new vector")
    func reinsertAfterDelete() async throws {
        let vectors = seededVectors(count: 64, dim: 16)
        let index = try await Self.build(vectors)
        let replacement = seededVector(dim: 16, seed: 777)

        try await index.delete(id: 5)
        #expect(try await index.search(query: vectors[5], topK: 64).allSatisfy { $0.id != 5 })
        try await index.insert(VectorRecord(id: 5, vector: replacement))

        #expect(await index.count == 64)
        let hits = try await index.search(query: replacement, topK: 1)
        #expect(hits.first?.id == 5)
        #expect(abs(hits.first?.distance ?? 1) < 1e-5)
        await Self.expectANNSError(Self.isIDAlreadyExists) {
            try await index.insert(VectorRecord(id: 5, vector: replacement))
        }
    }

    @Test("re-insert works on the graph path, in batches, and survives save/load")
    func reinsertGraphPathBatchAndPersistence() async throws {
        let n = 300
        let vectors = seededVectors(count: n, dim: 16)
        var configuration = IndexConfiguration(degree: 8)
        configuration.exactSearchMaxVectorCount = 0
        let index = try await Self.build(vectors, configuration: configuration)
        let deleted: [UInt64] = Array(0..<UInt64(40))
        for id in deleted { try await index.delete(id: id) }

        let replacements = seededVectors(count: 40, dim: 16, seed: 99)
        let fresh = seededVectors(count: 10, dim: 16, seed: 123)
        try await index.batchInsert(
            zip(deleted, replacements).map { VectorRecord(id: $0, vector: $1) }
                + fresh.enumerated().map { VectorRecord(id: UInt64(10_000 + $0), vector: $1) }
        )
        #expect(await index.count == n + 10)

        func verify(_ index: Ready) async throws {
            var found = 0
            for (id, vector) in zip(deleted, replacements) {
                if try await index.search(query: vector, topK: 1).first?.id == id { found += 1 }
            }
            #expect(found >= 38, "graph search found \(found)/40 re-inserted ids")
            for (offset, vector) in fresh.enumerated() {
                #expect(try await index.search(query: vector, topK: 1).first?.id == UInt64(10_000 + offset))
            }
        }
        try await verify(index)

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("reinsert.anns")
        try await index.save(to: url)
        let loaded = try await Unbuilt.load(from: url)
        #expect(await loaded.count == n + 10)
        try await verify(loaded)
    }

    @Test("numeric-id re-insert after delete")
    func numericReinsert() async throws {
        let vectors = seededVectors(count: 40, dim: 8)
        let index = GraphIndex(configuration: IndexConfiguration(degree: 4))
        try await index.build(vectors: Array(vectors.prefix(20)), ids: (0..<20).map { "s\($0)" })
        try await index.insert(vectors[20], numericID: 42)
        try await index.delete(numericID: 42)
        try await index.insert(vectors[21], numericID: 42)
        let hits = try await index.search(query: vectors[21], k: 1)
        #expect(hits.first?.numericID == 42)
    }

    // MARK: 3. Growable capacity

    @Test("inserts grow past the build size", arguments: [false, true])
    func capacityGrows(useFloat16: Bool) async throws {
        let base = seededVectors(count: 40, dim: 16)
        var configuration = IndexConfiguration(degree: 8)
        configuration.useFloat16 = useFloat16
        let index = try await Self.build(base, configuration: configuration)
        let extra = seededVectors(count: 700, dim: 16, seed: 7)
        for (offset, vector) in extra.prefix(200).enumerated() {
            try await index.insert(VectorRecord(id: UInt64(1_000 + offset), vector: vector))
        }
        try await index.batchInsert(
            extra.dropFirst(200).enumerated().map { VectorRecord(id: UInt64(1_200 + $0), vector: $1) }
        )
        #expect(await index.count == 740)
        for probe in [0, 199, 200, 699] {
            let hits = try await index.search(query: extra[probe], topK: 1)
            #expect(hits.first?.id == UInt64(1_000 + probe))
        }
    }

    @Test("graph search keeps recall after growth")
    func graphRecallAfterGrowth() async throws {
        let all = seededVectors(count: 2_000, dim: 16, seed: 3)
        var configuration = IndexConfiguration(degree: 16)
        configuration.exactSearchMaxVectorCount = 0
        let index = try await Self.build(Array(all.prefix(100)), configuration: configuration)
        let rest = all.dropFirst(100).enumerated().map { VectorRecord(id: UInt64(100 + $0), vector: $1) }
        try await index.batchInsert(rest)
        var hits = 0
        for q in stride(from: 0, to: 2_000, by: 40) {
            if try await index.search(query: all[q], topK: 1).first?.id == UInt64(q) { hits += 1 }
        }
        #expect(hits >= 48, "self-recall@1 \(hits)/50")
    }

    // MARK: 4. Small builds

    @Test("builds with count <= degree", arguments: [1, 2, 8, 32])
    func smallBuilds(count: Int) async throws {
        let vectors = seededVectors(count: count, dim: 16)
        for exactLimit in [IndexConfiguration.defaultExactSearchMaxVectorCount, 0] {
            var configuration = IndexConfiguration.default
            configuration.exactSearchMaxVectorCount = exactLimit
            let index = try await Self.build(vectors, configuration: configuration)
            #expect(await index.count == count)
            let hits = try await index.search(query: vectors[0], topK: 5)
            #expect(hits.count == min(5, count))
            #expect(hits.first?.id == 0)

            // Grow through the degree threshold and keep working.
            let extra = seededVectors(count: 100, dim: 16, seed: 31)
            for (offset, vector) in extra.enumerated() {
                try await index.insert(VectorRecord(id: UInt64(500 + offset), vector: vector))
            }
            #expect(try await index.search(query: extra[99], topK: 1).first?.id == 599)
        }
    }

    @Test("compaction down to fewer survivors than the degree")
    func compactToSmall() async throws {
        let vectors = seededVectors(count: 100, dim: 16)
        let index = try await Self.build(vectors, configuration: .default)
        for id in 0..<UInt64(95) { try await index.delete(id: id) }
        try await index.compact()
        #expect(await index.count == 5)
        #expect(try await index.search(query: vectors[97], topK: 1).first?.id == 97)
    }

    // MARK: 5. Cancellation

    @Test("a build cancelled before it starts throws CancellationError")
    func cancelledBeforeStart() async throws {
        let vectors = seededVectors(count: 20_000, dim: 64)
        let task = Task {
            try await Unbuilt(configuration: .default).build(vectors: vectors, ids: Self.ids(0..<20_000))
        }
        task.cancel()
        let result = await task.result
        guard case .failure(let error) = result else {
            Issue.record("cancelled build ran to completion")
            return
        }
        #expect(error is CancellationError)
    }

    @Test("a build cancelled midway stops early", arguments: [true, false])
    func cancelledMidway(gpuSized: Bool) async throws {
        // GPU path just above `minGPUConstructionNodeCount`; CPU path below it.
        // Kept small because tests run unoptimized and CI's GPU is virtual.
        let count = gpuSized ? 8_192 : 600
        let dim = gpuSized ? 32 : 16
        let vectors = seededVectors(count: count, dim: dim)
        var configuration = IndexConfiguration(degree: 16, maxIterations: gpuSized ? 20 : 8)
        configuration.convergenceThreshold = 0
        let ids = (0..<count).map(String.init)
        let fullStart = ContinuousClock.now
        try await GraphIndex(configuration: configuration).build(vectors: vectors, ids: ids)
        let fullDuration = ContinuousClock.now - fullStart

        let index = GraphIndex(configuration: configuration)
        let task = Task { try await index.build(vectors: vectors, ids: ids) }

        // Cancel once the build is observably running (it holds the write gate),
        // rather than after a sleep: sleeps oversleep badly on loaded CI runners.
        let deadline = ContinuousClock.now + .seconds(120)
        while await !index.writerActive, ContinuousClock.now < deadline {
            await Task.yield()
        }
        let cancelledAt = ContinuousClock.now
        task.cancel()
        let result = await task.result
        let stopLatency = ContinuousClock.now - cancelledAt

        guard case .failure(let error) = result else {
            Issue.record("build finished despite cancellation")
            return
        }
        #expect(error is CancellationError)
        // Cancelled near the start, so almost all the work was still ahead:
        // stopping in under half a full build means the loops checked.
        #expect(stopLatency < fullDuration / 2, "stopped \(stopLatency) after cancel; full build \(fullDuration)")
        #expect(await index.isBuilt == false)
        // The gate was released and the index is still usable.
        try await index.build(vectors: Array(vectors.prefix(2)), ids: ["a", "b"])
        #expect(await index.count == 2)
    }

    // MARK: 6. Flat build

    @Test("flat build matches nested build and allocates exactly the build size")
    func flatBuild() async throws {
        let n = 500
        let rows = seededVectors(count: n, dim: 24)
        let flat = rows.flatMap { $0 }
        let nested = try await Self.build(rows)
        let fromFlat = try await Unbuilt(configuration: IndexConfiguration(degree: 8))
            .build(flatVectors: flat, dimension: 24, ids: Self.ids(0..<n))
        #expect(await fromFlat.count == n)
        #expect(await fromFlat.advanced.vectors?.capacity == n)
        for q in [0, 250, 499] {
            let a = try await nested.search(query: rows[q], topK: 10)
            let b = try await fromFlat.search(query: rows[q], topK: 10)
            #expect(a.map(\.id) == b.map(\.id))
        }
        await Self.expectANNSError(Self.isConstructionFailed) {
            _ = try await Unbuilt().build(flatVectors: Array(flat.dropLast()), dimension: 24, ids: Self.ids(0..<n))
        }
        await Self.expectANNSError(Self.isConstructionFailed) {
            _ = try await Unbuilt().build(flatVectors: flat, dimension: 24, ids: Self.ids(0..<(n - 1)))
        }
    }

    // MARK: 7. Input validation

    @Test("non-finite vectors are rejected with their position", arguments: [Float.nan, .infinity, -.infinity])
    func nonFiniteRejected(value: Float) async throws {
        var rows = seededVectors(count: 64, dim: 16)
        rows[10][3] = value
        await Self.expectANNSError(Self.isNonFinite(at: 10)) { _ = try await Self.build(rows) }
        await Self.expectANNSError(Self.isNonFinite(at: 10)) {
            _ = try await Unbuilt(configuration: IndexConfiguration(degree: 8))
                .build(flatVectors: rows.flatMap { $0 }, dimension: 16, ids: Self.ids(0..<64))
        }

        let clean = seededVectors(count: 64, dim: 16)
        let index = try await Self.build(clean)
        var bad = clean[0]
        bad[0] = value
        await Self.expectANNSError(Self.isNonFinite(at: 0)) {
            try await index.insert(VectorRecord(id: 999, vector: bad))
        }
        await Self.expectANNSError(Self.isNonFinite(at: 2)) {
            try await index.batchInsert([
                VectorRecord(id: 1_000, vector: clean[1]),
                VectorRecord(id: 1_001, vector: clean[2]),
                VectorRecord(id: 1_002, vector: bad),
            ])
        }
        #expect(await index.count == 64, "a rejected batch inserts nothing")
        await Self.expectANNSError(Self.isNonFinite(at: 0)) { _ = try await index.search(query: bad, topK: 3) }
        await Self.expectANNSError(Self.isNonFinite(at: 1)) {
            _ = try await index.batchSearch(queries: [clean[0], bad], topK: 3)
        }
        await Self.expectANNSError(Self.isNonFinite(at: 0)) {
            _ = try await index.rangeSearch(query: bad, maxDistance: 1)
        }
    }

    // MARK: 8. Exact vs graph parity

    @Test("exact and graph paths report the same cosine distances for non-unit vectors", arguments: [5_000])
    func exactGraphParity(count: Int) async throws {
        var rng = SeededGenerator(state: 99)
        let dim = 384
        let vectors: [[Float]] = (0..<count).map { _ in
            let scale = Float.random(in: 0.5...3, using: &rng)
            return (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) * scale }
        }
        var graphConfiguration = IndexConfiguration.default
        graphConfiguration.exactSearchMaxVectorCount = 0
        let exact = try await Self.build(vectors, configuration: .default)
        let graph = try await Self.build(vectors, configuration: graphConfiguration)

        var maxPathDelta: Float = 0
        var maxReferenceDelta: Double = 0
        for q in 0..<5 {
            let query = vectors[(q * 97) % count].map { $0 * 1.7 + 0.01 }
            let exactHits = try await exact.search(query: query, topK: 10)
            let graphHits = Dictionary(
                uniqueKeysWithValues: try await graph.search(query: query, topK: 10).map { ($0.id, $0.distance) }
            )
            for hit in exactHits {
                let reference = 1 - Self.referenceSimilarity(query, vectors[Int(hit.id)], metric: .cosine)
                maxReferenceDelta = max(maxReferenceDelta, abs(Double(hit.distance) - reference))
                if let graphDistance = graphHits[hit.id] {
                    maxPathDelta = max(maxPathDelta, abs(graphDistance - hit.distance))
                }
            }
        }
        #expect(maxPathDelta < 1e-5)
        #expect(maxReferenceDelta < 1e-5)
    }

    // MARK: 9. Concurrency

    @Test("an insert that lands during compact() is kept")
    func insertDuringCompact() async throws {
        let n = 8_192
        let vectors = seededVectors(count: n, dim: 64)
        let index = try await Self.build(vectors, configuration: .default)
        for id in 0..<UInt64(100) { try await index.delete(id: id) }
        let newVector = seededVector(dim: 64, seed: 4242)
        async let compaction: Void = index.compact()
        try await Task.sleep(for: .milliseconds(5))
        try await index.insert(VectorRecord(id: 999_999, vector: newVector))
        try await compaction
        #expect(await index.count == n - 100 + 1)
        #expect(try await index.search(query: newVector, topK: 1).first?.id == 999_999)
    }

    @Test("a delete that lands during compact() is kept")
    func deleteDuringCompact() async throws {
        let n = 8_192
        let vectors = seededVectors(count: n, dim: 64)
        let index = try await Self.build(vectors, configuration: .default)
        for id in 0..<UInt64(100) { try await index.delete(id: id) }
        async let compaction: Void = index.compact()
        try await Task.sleep(for: .milliseconds(5))
        try await index.delete(id: 500)
        try await compaction
        #expect(await index.count == n - 101)
        #expect(try await index.search(query: vectors[500], topK: 5).contains { $0.id == 500 } == false)
    }

    @Test("re-insert refreshes every stored edge distance into the reused slot")
    func reinsertRefreshesInEdges() async throws {
        let vectors = seededVectors(count: 500, dim: 16)
        let index = try await Self.build(vectors, configuration: IndexConfiguration(degree: 8, metric: .cosine))
        let replacement = [Float](repeating: 3, count: 16)
        try await index.delete(id: 7)
        try await index.insert(VectorRecord(id: 7, vector: replacement))

        let graphIndex = index.advanced
        let graph = try #require(await graphIndex.graph)
        let storage = try #require(await graphIndex.vectors)
        let metric = await graphIndex.configuration.metric
        var inEdges = 0
        for node in 0..<graph.nodeCount where node != 7 {
            let ids = graph.neighborIDs(of: node)
            let distances = graph.neighborDistances(of: node)
            for (slot, id) in ids.enumerated() where id == 7 {
                inEdges += 1
                let expected = SIMDDistance.distance(storage.vector(at: node), replacement, metric: metric)
                #expect(abs(distances[slot] - expected) < 1e-4, "node \(node)")
            }
        }
        #expect(inEdges > 0)
    }

    @Test("concurrent searches, inserts, deletes and re-inserts stay consistent")
    func concurrentMutationAndSearch() async throws {
        let base = seededVectors(count: 2_000, dim: 32)
        let index = try await Self.build(base, configuration: .default)
        let extra = seededVectors(count: 120, dim: 32, seed: 8)

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                for (offset, vector) in extra.enumerated() {
                    try await index.insert(VectorRecord(id: UInt64(10_000 + offset), vector: vector))
                }
            }
            group.addTask {
                for id in 0..<UInt64(60) {
                    try await index.delete(id: id)
                    try await index.insert(VectorRecord(id: id, vector: base[Int(id) + 1]))
                }
            }
            for worker in 0..<4 {
                group.addTask {
                    for q in 0..<50 {
                        let hits = try await index.search(query: base[(worker * 50 + q) % base.count], topK: 10)
                        #expect(hits.count == 10)
                        #expect(Set(hits.map(\.id)).count == hits.count)
                    }
                }
            }
            try await group.waitForAll()
        }

        #expect(await index.count == 2_120)
        for (offset, vector) in extra.enumerated() where offset % 20 == 0 {
            #expect(try await index.search(query: vector, topK: 1).first?.id == UInt64(10_000 + offset))
        }
    }

    @Test("a second concurrent build on the same GraphIndex is rejected")
    func concurrentBuildRejected() async throws {
        let vectors = seededVectors(count: 5_000, dim: 32)
        let ids = (0..<5_000).map(String.init)
        let index = GraphIndex(configuration: .default)
        let first = Task { try await index.build(vectors: vectors, ids: ids) }
        let second = Task { try await index.build(vectors: vectors, ids: ids) }
        let results = [await first.result, await second.result]
        #expect(results.filter { if case .success = $0 { true } else { false } }.count == 1)
    }
}
