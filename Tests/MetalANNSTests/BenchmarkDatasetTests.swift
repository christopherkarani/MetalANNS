import Foundation
import Testing
@testable import MetalANNSBenchmarks
import MetalANNSCore

@Suite("BenchmarkDataset Tests")
struct BenchmarkDatasetTests {
    @Test("writeAndReadRoundTrip")
    func writeAndReadRoundTrip() throws {
        let dataset = sampleDataset(metric: .cosine)
        let path = temporaryPath(name: "roundtrip")
        defer { try? FileManager.default.removeItem(atPath: path) }

        try dataset.save(to: path)
        let loaded = try BenchmarkDataset.load(from: path)

        #expect(loaded.dimension == dataset.dimension)
        #expect(loaded.neighborsCount == dataset.neighborsCount)
        #expect(loaded.metric == dataset.metric)
        #expect(loaded.trainVectors == dataset.trainVectors)
        #expect(loaded.testVectors == dataset.testVectors)
        #expect(loaded.groundTruth == dataset.groundTruth)
    }

    @Test("trainVectorsPreserved")
    func trainVectorsPreserved() throws {
        let dataset = sampleDataset(metric: .l2)
        let path = temporaryPath(name: "train")
        defer { try? FileManager.default.removeItem(atPath: path) }

        try dataset.save(to: path)
        let loaded = try BenchmarkDataset.load(from: path)

        #expect(loaded.trainVectors == dataset.trainVectors)
    }

    @Test("testVectorsPreserved")
    func testVectorsPreserved() throws {
        let dataset = sampleDataset(metric: .innerProduct)
        let path = temporaryPath(name: "test")
        defer { try? FileManager.default.removeItem(atPath: path) }

        try dataset.save(to: path)
        let loaded = try BenchmarkDataset.load(from: path)

        #expect(loaded.testVectors == dataset.testVectors)
    }

    @Test("groundTruthPreserved")
    func groundTruthPreserved() throws {
        let dataset = sampleDataset(metric: .cosine)
        let path = temporaryPath(name: "gt")
        defer { try? FileManager.default.removeItem(atPath: path) }

        try dataset.save(to: path)
        let loaded = try BenchmarkDataset.load(from: path)

        #expect(loaded.groundTruth == dataset.groundTruth)
    }

    @Test("metricRoundTrip")
    func metricRoundTrip() throws {
        for metric in [Metric.cosine, .l2, .innerProduct, .hamming] {
            let dataset = sampleDataset(metric: metric)
            let path = temporaryPath(name: "metric-\(metric.rawValue)")
            defer { try? FileManager.default.removeItem(atPath: path) }

            try dataset.save(to: path)
            let loaded = try BenchmarkDataset.load(from: path)

            #expect(loaded.metric == metric)
        }
    }
}

private func sampleDataset(metric: Metric) -> BenchmarkDataset {
    BenchmarkDataset(
        trainVectors: [
            [0.1, 0.2, 0.3, 0.4],
            [1.5, -2.0, 3.25, -4.75],
            [9.0, 8.0, 7.0, 6.0]
        ],
        testVectors: [
            [0.25, 0.5, 0.75, 1.0],
            [-1.0, -2.0, -3.0, -4.0]
        ],
        groundTruth: [
            [1, 0, 2],
            [2, 1, 0]
        ],
        dimension: 4,
        metric: metric,
        neighborsCount: 3
    )
}

private func temporaryPath(name: String) -> String {
    FileManager.default.temporaryDirectory
        .appendingPathComponent("bench-dataset-\(name)-\(UUID().uuidString).annbin")
        .path
}
