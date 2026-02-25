import Foundation
import MetalANNS

struct BenchmarkRunner {
    struct Config {
        var vectorCount: Int = 1000
        var dim: Int = 128
        var degree: Int = 32
        var queryCount: Int = 100
        var k: Int = 10
        var efSearch: Int = 64
        var metric: Metric = .cosine
    }

    struct Results {
        var buildTimeMs: Double
        var queryLatencyP50Ms: Double
        var queryLatencyP95Ms: Double
        var queryLatencyP99Ms: Double
        var recallAt1: Double
        var recallAt10: Double
        var recallAt100: Double
    }

    static func run(config: Config) async throws -> Results {
        let vectors = makeVectors(count: config.vectorCount, dim: config.dim, seedOffset: 0)
        let ids = (0..<config.vectorCount).map { "v_\($0)" }

        let index = ANNSIndex(
            configuration: IndexConfiguration(
                degree: config.degree,
                metric: config.metric,
                efSearch: config.efSearch
            )
        )

        let buildStart = DispatchTime.now().uptimeNanoseconds
        try await index.build(vectors: vectors, ids: ids)
        let buildEnd = DispatchTime.now().uptimeNanoseconds
        let buildTimeMs = Double(buildEnd - buildStart) / 1_000_000.0

        let queries = makeVectors(count: config.queryCount, dim: config.dim, seedOffset: 1_000_000)
        var latenciesMs: [Double] = []
        latenciesMs.reserveCapacity(config.queryCount)

        let top1Count = min(1, config.vectorCount)
        let top10Count = min(10, config.vectorCount)
        let top100Count = min(100, config.vectorCount)

        var recallAt1Total: Double = 0
        var recallAt10Total: Double = 0
        var recallAt100Total: Double = 0

        for query in queries {
            let latencyStart = DispatchTime.now().uptimeNanoseconds
            _ = try await index.search(query: query, k: config.k)
            let latencyEnd = DispatchTime.now().uptimeNanoseconds
            latenciesMs.append(Double(latencyEnd - latencyStart) / 1_000_000.0)

            let approx = try await index.search(query: query, k: top100Count)
            let exact = bruteForceTopK(
                query: query,
                vectors: vectors,
                ids: ids,
                k: top100Count,
                metric: config.metric
            )

            let approxTop1 = Set(approx.prefix(top1Count).map(\.id))
            let exactTop1 = Set(exact.prefix(top1Count))
            recallAt1Total += Double(approxTop1.intersection(exactTop1).count) / Double(max(1, top1Count))

            let approxTop10 = Set(approx.prefix(top10Count).map(\.id))
            let exactTop10 = Set(exact.prefix(top10Count))
            recallAt10Total += Double(approxTop10.intersection(exactTop10).count) / Double(max(1, top10Count))

            let approxTop100 = Set(approx.prefix(top100Count).map(\.id))
            let exactTop100 = Set(exact.prefix(top100Count))
            recallAt100Total += Double(approxTop100.intersection(exactTop100).count) / Double(max(1, top100Count))
        }

        let queryCount = Double(max(1, queries.count))
        return Results(
            buildTimeMs: buildTimeMs,
            queryLatencyP50Ms: percentile(0.50, in: latenciesMs),
            queryLatencyP95Ms: percentile(0.95, in: latenciesMs),
            queryLatencyP99Ms: percentile(0.99, in: latenciesMs),
            recallAt1: recallAt1Total / queryCount,
            recallAt10: recallAt10Total / queryCount,
            recallAt100: recallAt100Total / queryCount
        )
    }

    private static func percentile(_ p: Double, in values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        let sorted = values.sorted()
        let rank = Int(ceil(p * Double(sorted.count))) - 1
        let index = min(max(rank, 0), sorted.count - 1)
        return sorted[index]
    }

    private static func makeVectors(count: Int, dim: Int, seedOffset: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let i = Float((row + seedOffset) * dim + col)
                return sin(i * 0.173) + cos(i * 0.071)
            }
        }
    }

    private static func bruteForceTopK(
        query: [Float],
        vectors: [[Float]],
        ids: [String],
        k: Int,
        metric: Metric
    ) -> [String] {
        let scored = vectors.enumerated().map { idx, vector -> (id: String, distance: Float) in
            (ids[idx], distance(query: query, vector: vector, metric: metric))
        }
        return scored.sorted { $0.distance < $1.distance }.prefix(k).map(\.id)
    }

    private static func distance(query: [Float], vector: [Float], metric: Metric) -> Float {
        switch metric {
        case .cosine:
            var dot: Float = 0
            var normQ: Float = 0
            var normV: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
                normQ += query[d] * query[d]
                normV += vector[d] * vector[d]
            }
            let denom = sqrt(normQ) * sqrt(normV)
            return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
        case .l2:
            var sum: Float = 0
            for d in 0..<query.count {
                let diff = query[d] - vector[d]
                sum += diff * diff
            }
            return sum
        case .innerProduct:
            var dot: Float = 0
            for d in 0..<query.count {
                dot += query[d] * vector[d]
            }
            return -dot
        }
    }
}
