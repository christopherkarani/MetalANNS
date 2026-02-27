import Foundation
import MetalANNS
import MetalANNSCore

public struct IVFPQBenchmark: Sendable {
    public struct ComparisonResults: Sendable {
        public var annsResults: BenchmarkReport.Row
        public var ivfpqResults: BenchmarkReport.Row

        public init(annsResults: BenchmarkReport.Row, ivfpqResults: BenchmarkReport.Row) {
            self.annsResults = annsResults
            self.ivfpqResults = ivfpqResults
        }
    }

    static func run(
        dataset: BenchmarkDataset,
        annsConfig: BenchmarkRunner.Config,
        ivfpqConfig: IVFPQConfiguration
    ) async throws -> ComparisonResults {
        let anns = try await BenchmarkRunner.run(config: annsConfig, dataset: dataset)
        let annsRow = BenchmarkReport.Row(
            label: "ANNSIndex",
            recallAt10: anns.recallAt10,
            qps: anns.qps,
            buildTimeMs: anns.buildTimeMs,
            p50Ms: anns.queryLatencyP50Ms,
            p95Ms: anns.queryLatencyP95Ms,
            p99Ms: anns.queryLatencyP99Ms
        )

        let capacity = max(dataset.trainVectors.count * 2, dataset.trainVectors.count + 1)
        let ivfpqIndex = try IVFPQIndex(
            capacity: capacity,
            dimension: dataset.dimension,
            config: ivfpqConfig
        )

        let trainStart = DispatchTime.now().uptimeNanoseconds
        try await ivfpqIndex.train(vectors: dataset.trainVectors)
        let ids = (0..<dataset.trainVectors.count).map { "v_\($0)" }
        try await ivfpqIndex.add(vectors: dataset.trainVectors, ids: ids)
        let trainEnd = DispatchTime.now().uptimeNanoseconds
        let buildTimeMs = Double(trainEnd - trainStart) / 1_000_000.0

        let top1Count = min(1, dataset.neighborsCount)
        let top10Count = min(10, dataset.neighborsCount)
        let top100Count = min(100, dataset.neighborsCount)
        let queryK = max(10, top100Count)

        var latenciesMs: [Double] = []
        latenciesMs.reserveCapacity(dataset.testVectors.count)

        var recallAt1Total: Double = 0
        var recallAt10Total: Double = 0
        var recallAt100Total: Double = 0

        let batchStart = DispatchTime.now().uptimeNanoseconds
        for (index, query) in dataset.testVectors.enumerated() {
            let latencyStart = DispatchTime.now().uptimeNanoseconds
            let results = await ivfpqIndex.search(query: query, k: queryK)
            let latencyEnd = DispatchTime.now().uptimeNanoseconds
            latenciesMs.append(Double(latencyEnd - latencyStart) / 1_000_000.0)

            let expected = dataset.groundTruth[index]
            let resultIDs = results.compactMap(parseID)

            let approxTop1 = Set(resultIDs.prefix(top1Count))
            let exactTop1 = Set(expected.prefix(top1Count))
            recallAt1Total += Double(approxTop1.intersection(exactTop1).count) / Double(max(1, top1Count))

            let approxTop10 = Set(resultIDs.prefix(top10Count))
            let exactTop10 = Set(expected.prefix(top10Count))
            recallAt10Total += Double(approxTop10.intersection(exactTop10).count) / Double(max(1, top10Count))

            let approxTop100 = Set(resultIDs.prefix(top100Count))
            let exactTop100 = Set(expected.prefix(top100Count))
            recallAt100Total += Double(approxTop100.intersection(exactTop100).count) / Double(max(1, top100Count))
        }
        let batchEnd = DispatchTime.now().uptimeNanoseconds

        let queryCount = max(1, dataset.testVectors.count)
        let totalBatchSeconds = Double(batchEnd - batchStart) / 1_000_000_000.0
        let qps = totalBatchSeconds > 0 ? Double(dataset.testVectors.count) / totalBatchSeconds : 0

        let ivfpqRow = BenchmarkReport.Row(
            label: "IVFPQIndex",
            recallAt10: recallAt10Total / Double(queryCount),
            qps: qps,
            buildTimeMs: buildTimeMs,
            p50Ms: percentile(0.50, in: latenciesMs),
            p95Ms: percentile(0.95, in: latenciesMs),
            p99Ms: percentile(0.99, in: latenciesMs)
        )

        return ComparisonResults(annsResults: annsRow, ivfpqResults: ivfpqRow)
    }

    static func renderComparison(_ results: ComparisonResults) -> String {
        BenchmarkReport(
            rows: [results.annsResults, results.ivfpqResults],
            datasetLabel: "comparison"
        ).renderTable()
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

    private static func parseID(_ result: SearchResult) -> UInt32? {
        if let direct = UInt32(result.id) {
            return direct
        }

        if let underscore = result.id.firstIndex(of: "_") {
            let suffixIndex = result.id.index(after: underscore)
            return UInt32(result.id[suffixIndex...])
        }
        return nil
    }
}
