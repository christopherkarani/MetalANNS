import Foundation
import MetalANNS
import MetalANNSCore

let efSearchSweep = [16, 32, 64, 128, 256]

print("MetalANNS Benchmark Suite")
print("========================")

do {
    let args = Array(CommandLine.arguments.dropFirst())
    if args.contains("--help") {
        printUsage()
        exit(0)
    }

    let datasetPath = try value(for: "--dataset", in: args)
    let csvOutPath = try value(for: "--csv-out", in: args)
    let sweepMode = args.contains("--sweep")
    let ivfpqMode = args.contains("--ivfpq")

    if ivfpqMode {
        let dataset: BenchmarkDataset
        if let datasetPath {
            dataset = try BenchmarkDataset.load(from: datasetPath)
        } else {
            dataset = BenchmarkDataset.synthetic(
                trainCount: 1_200,
                testCount: 200,
                dimension: 128,
                k: 100,
                metric: .l2,
                seed: 42
            )
        }

        let config = BenchmarkRunner.Config(
            vectorCount: dataset.trainVectors.count,
            dim: dataset.dimension,
            degree: 32,
            queryCount: dataset.testVectors.count,
            k: 10,
            efSearch: 64,
            metric: dataset.metric
        )

        let results = try await BenchmarkRunner.run(config: config, dataset: dataset)
        let report = BenchmarkReport(
            rows: [
                BenchmarkReport.Row(
                    label: "ANNSIndex",
                    recallAt10: results.recallAt10,
                    qps: results.qps,
                    buildTimeMs: results.buildTimeMs,
                    p50Ms: results.queryLatencyP50Ms,
                    p95Ms: results.queryLatencyP95Ms,
                    p99Ms: results.queryLatencyP99Ms
                )
            ],
            datasetLabel: datasetPath ?? "synthetic"
        )

        print(report.renderTable())
        if let csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        exit(0)
    }

    if sweepMode {
        let dataset: BenchmarkDataset
        if let datasetPath {
            dataset = try BenchmarkDataset.load(from: datasetPath)
        } else {
            let base = BenchmarkRunner.Config()
            dataset = BenchmarkDataset.synthetic(
                trainCount: base.vectorCount,
                testCount: base.queryCount,
                dimension: base.dim,
                k: 100,
                metric: base.metric,
                seed: 42
            )
        }

        let baseConfig = BenchmarkRunner.Config(
            vectorCount: dataset.trainVectors.count,
            dim: dataset.dimension,
            degree: 32,
            queryCount: dataset.testVectors.count,
            k: 10,
            efSearch: 64,
            metric: dataset.metric
        )

        let configs = efSearchSweep.map { efSearch in
            (
                label: "efSearch=\(efSearch)",
                config: BenchmarkRunner.Config(
                    vectorCount: baseConfig.vectorCount,
                    dim: baseConfig.dim,
                    degree: baseConfig.degree,
                    queryCount: baseConfig.queryCount,
                    k: baseConfig.k,
                    efSearch: efSearch,
                    metric: baseConfig.metric
                )
            )
        }

        let report = try await BenchmarkRunner.sweep(configs: configs, dataset: dataset)
        print(report.renderTable())
        print("Pareto frontier points: \(report.paretoFrontier().count)")

        if let csvOutPath {
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
        exit(0)
    }

    if let datasetPath {
        let dataset = try BenchmarkDataset.load(from: datasetPath)
        let config = BenchmarkRunner.Config(
            vectorCount: dataset.trainVectors.count,
            dim: dataset.dimension,
            degree: 32,
            queryCount: dataset.testVectors.count,
            k: 10,
            efSearch: 64,
            metric: dataset.metric
        )
        let results = try await BenchmarkRunner.run(config: config, dataset: dataset)
        printResults(results)

        if let csvOutPath {
            let report = BenchmarkReport(
                rows: [
                    BenchmarkReport.Row(
                        label: "single",
                        recallAt10: results.recallAt10,
                        qps: results.qps,
                        buildTimeMs: results.buildTimeMs,
                        p50Ms: results.queryLatencyP50Ms,
                        p95Ms: results.queryLatencyP95Ms,
                        p99Ms: results.queryLatencyP99Ms
                    )
                ],
                datasetLabel: datasetPath
            )
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
    } else {
        let config = BenchmarkRunner.Config()
        let results = try await BenchmarkRunner.run(config: config)
        printResults(results)

        if let csvOutPath {
            let report = BenchmarkReport(
                rows: [
                    BenchmarkReport.Row(
                        label: "single",
                        recallAt10: results.recallAt10,
                        qps: results.qps,
                        buildTimeMs: results.buildTimeMs,
                        p50Ms: results.queryLatencyP50Ms,
                        p95Ms: results.queryLatencyP95Ms,
                        p99Ms: results.queryLatencyP99Ms
                    )
                ],
                datasetLabel: "synthetic"
            )
            try report.saveCSV(to: csvOutPath)
            print("Saved CSV: \(csvOutPath)")
        }
    }
} catch {
    fputs("Benchmark failed: \(error)\n", stderr)
    printUsage()
    exit(1)
}

func printResults(_ results: BenchmarkRunner.Results) {
    print("Build time:      \(String(format: "%.1f", results.buildTimeMs)) ms")
    print("Query p50:       \(String(format: "%.2f", results.queryLatencyP50Ms)) ms")
    print("Query p95:       \(String(format: "%.2f", results.queryLatencyP95Ms)) ms")
    print("Query p99:       \(String(format: "%.2f", results.queryLatencyP99Ms)) ms")
    print("Recall@1:        \(String(format: "%.3f", results.recallAt1))")
    print("Recall@10:       \(String(format: "%.3f", results.recallAt10))")
    print("Recall@100:      \(String(format: "%.3f", results.recallAt100))")
}

func printUsage() {
    print("USAGE:")
    print("  MetalANNSBenchmarks                             # synthetic single run (default)")
    print("  MetalANNSBenchmarks --sweep                     # efSearch sweep on synthetic data")
    print("  MetalANNSBenchmarks --dataset <path.annbin>     # load dataset, single run")
    print("  MetalANNSBenchmarks --dataset <path.annbin> --sweep    # load dataset, efSearch sweep")
    print("  MetalANNSBenchmarks --dataset <path.annbin> --csv-out <path.csv>  # save CSV")
    print("  MetalANNSBenchmarks --ivfpq                     # compare ANNSIndex vs IVFPQIndex (synthetic)")
}

func value(for flag: String, in args: [String]) throws -> String? {
    guard let index = args.firstIndex(of: flag) else {
        return nil
    }

    let valueIndex = args.index(after: index)
    guard valueIndex < args.count else {
        throw BenchmarkDatasetError.invalidDataset("Missing value for \(flag)")
    }

    return args[valueIndex]
}
