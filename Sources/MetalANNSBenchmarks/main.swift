import Foundation
import MetalANNS
import MetalANNSCore

print("MetalANNS Benchmark Suite")
print("========================")

let semaphore = DispatchSemaphore(value: 0)

Task {
    defer { semaphore.signal() }

    do {
        let config = BenchmarkRunner.Config()
        let results = try await BenchmarkRunner.run(config: config)

        print("Build time:      \(String(format: "%.1f", results.buildTimeMs)) ms")
        print("Query p50:       \(String(format: "%.2f", results.queryLatencyP50Ms)) ms")
        print("Query p95:       \(String(format: "%.2f", results.queryLatencyP95Ms)) ms")
        print("Query p99:       \(String(format: "%.2f", results.queryLatencyP99Ms)) ms")
        print("Recall@1:        \(String(format: "%.3f", results.recallAt1))")
        print("Recall@10:       \(String(format: "%.3f", results.recallAt10))")
        print("Recall@100:      \(String(format: "%.3f", results.recallAt100))")
    } catch {
        fputs("Benchmark failed: \(error)\n", stderr)
        exit(1)
    }
}

semaphore.wait()
