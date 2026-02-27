import Testing
@testable import MetalANNSBenchmarks
import MetalANNSCore

@Suite("BenchmarkRunner Sweep Tests")
struct BenchmarkRunnerSweepTests {
    @Test("sweepReturnsOneRowPerConfig")
    func sweepReturnsOneRowPerConfig() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 200,
            testCount: 50,
            dimension: 32,
            k: 100,
            metric: .cosine,
            seed: 42
        )

        let configs: [(label: String, config: BenchmarkRunner.Config)] = [
            ("efSearch=16", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 16, metric: .cosine)),
            ("efSearch=64", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 64, metric: .cosine)),
            ("efSearch=128", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 128, metric: .cosine))
        ]

        let report = try await BenchmarkRunner.sweep(configs: configs, dataset: dataset)
        #expect(report.rows.count == 3)
    }

    @Test("qpsIsPositive")
    func qpsIsPositive() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 200,
            testCount: 50,
            dimension: 32,
            k: 100,
            metric: .cosine,
            seed: 123
        )

        let configs: [(label: String, config: BenchmarkRunner.Config)] = [
            ("efSearch=32", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 32, metric: .cosine)),
            ("efSearch=64", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 64, metric: .cosine)),
            ("efSearch=128", BenchmarkRunner.Config(vectorCount: 200, dim: 32, degree: 32, queryCount: 50, k: 10, efSearch: 128, metric: .cosine))
        ]

        let report = try await BenchmarkRunner.sweep(configs: configs, dataset: dataset)
        #expect(report.rows.allSatisfy { $0.qps > 0 })
    }

    @Test("recallFromDataset")
    func recallFromDataset() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 200,
            testCount: 50,
            dimension: 32,
            k: 100,
            metric: .cosine,
            seed: 9
        )

        let config = BenchmarkRunner.Config(
            vectorCount: 200,
            dim: 32,
            degree: 32,
            queryCount: 50,
            k: 10,
            efSearch: 128,
            metric: .cosine
        )

        let results = try await BenchmarkRunner.run(config: config, dataset: dataset)
        #expect(results.recallAt10 > 0.5)
    }
}
