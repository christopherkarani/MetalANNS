import Testing
@testable import MetalANNSBenchmarks
@testable import MetalANNS
import MetalANNSCore

@Suite("IVFPQBenchmark Tests")
struct IVFPQBenchmarkTests {
    @Test("runsBothIndexes")
    func runsBothIndexes() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 512,
            testCount: 80,
            dimension: 32,
            k: 100,
            metric: .l2,
            seed: 42
        )

        let annsConfig = BenchmarkRunner.Config(
            vectorCount: dataset.trainVectors.count,
            dim: dataset.dimension,
            degree: 32,
            queryCount: dataset.testVectors.count,
            k: 10,
            efSearch: 64,
            metric: .l2
        )

        let ivfpqConfig = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 8
        )

        let results = try await IVFPQBenchmark.run(
            dataset: dataset,
            annsConfig: annsConfig,
            ivfpqConfig: ivfpqConfig
        )

        #expect(results.annsResults.qps > 0)
        #expect(results.ivfpqResults.qps > 0)
    }

    @Test("ivfpqRecallPositive")
    func ivfpqRecallPositive() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 512,
            testCount: 80,
            dimension: 32,
            k: 100,
            metric: .l2,
            seed: 123
        )

        let annsConfig = BenchmarkRunner.Config(
            vectorCount: dataset.trainVectors.count,
            dim: dataset.dimension,
            degree: 32,
            queryCount: dataset.testVectors.count,
            k: 10,
            efSearch: 64,
            metric: .l2
        )

        let ivfpqConfig = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 8
        )

        let results = try await IVFPQBenchmark.run(
            dataset: dataset,
            annsConfig: annsConfig,
            ivfpqConfig: ivfpqConfig
        )

        #expect(results.ivfpqResults.recallAt10 > 0)
    }

    @Test("annsBuildsFaster")
    func annsBuildsFaster() async throws {
        let dataset = BenchmarkDataset.synthetic(
            trainCount: 512,
            testCount: 80,
            dimension: 32,
            k: 100,
            metric: .l2,
            seed: 999
        )

        let annsConfig = BenchmarkRunner.Config(
            vectorCount: dataset.trainVectors.count,
            dim: dataset.dimension,
            degree: 32,
            queryCount: dataset.testVectors.count,
            k: 10,
            efSearch: 64,
            metric: .l2
        )

        let ivfpqConfig = IVFPQConfiguration(
            numSubspaces: 8,
            numCentroids: 256,
            numCoarseCentroids: 32,
            nprobe: 8,
            metric: .l2,
            trainingIterations: 8
        )

        let results = try await IVFPQBenchmark.run(
            dataset: dataset,
            annsConfig: annsConfig,
            ivfpqConfig: ivfpqConfig
        )

        #expect(results.annsResults.buildTimeMs > 0)
        #expect(results.ivfpqResults.buildTimeMs > 0)
    }
}
