import Foundation
import Testing
@testable import MetalANNSBenchmarks

@Suite("BenchmarkReport Tests")
struct BenchmarkReportTests {
    @Test("tableOutput")
    func tableOutput() {
        let report = sampleReport()
        let table = report.renderTable()

        #expect(table.contains("label"))
        #expect(table.contains("recall@10"))
        #expect(table.contains("QPS"))
        #expect(table.contains("efSearch=16"))
        #expect(table.contains("efSearch=128"))
    }

    @Test("csvOutput")
    func csvOutput() {
        let report = sampleReport()
        let csv = report.renderCSV()
        let lines = csv.split(separator: "\n", omittingEmptySubsequences: true)

        #expect(lines.count == 4)
        #expect(String(lines[0]) == "label,recall@10,qps,buildTimeMs,p50ms,p95ms,p99ms")
    }

    @Test("paretoFrontier")
    func paretoFrontier() {
        let rows: [BenchmarkReport.Row] = [
            .init(label: "a", recallAt10: 0.90, qps: 4000, buildTimeMs: 100, p50Ms: 1, p95Ms: 2, p99Ms: 3),
            .init(label: "b", recallAt10: 0.92, qps: 3500, buildTimeMs: 110, p50Ms: 1.1, p95Ms: 2.1, p99Ms: 3.1),
            .init(label: "c", recallAt10: 0.91, qps: 3000, buildTimeMs: 120, p50Ms: 1.2, p95Ms: 2.2, p99Ms: 3.2), // dominated by b
            .init(label: "d", recallAt10: 0.95, qps: 2500, buildTimeMs: 130, p50Ms: 1.3, p95Ms: 2.3, p99Ms: 3.3),
            .init(label: "e", recallAt10: 0.88, qps: 4500, buildTimeMs: 90, p50Ms: 0.9, p95Ms: 1.9, p99Ms: 2.9)
        ]

        let frontier = BenchmarkReport(rows: rows, datasetLabel: "synthetic").paretoFrontier()
        let labels = Set(frontier.map(\.label))

        #expect(frontier.count == 4)
        #expect(labels.contains("a"))
        #expect(labels.contains("b"))
        #expect(labels.contains("d"))
        #expect(labels.contains("e"))
        #expect(!labels.contains("c"))
    }
}

private func sampleReport() -> BenchmarkReport {
    BenchmarkReport(
        rows: [
            .init(label: "efSearch=16", recallAt10: 0.841, qps: 6800, buildTimeMs: 141.8, p50Ms: 0.8, p95Ms: 1.4, p99Ms: 2.2),
            .init(label: "efSearch=64", recallAt10: 0.953, qps: 4231, buildTimeMs: 142.0, p50Ms: 1.2, p95Ms: 2.1, p99Ms: 3.4),
            .init(label: "efSearch=128", recallAt10: 0.971, qps: 2108, buildTimeMs: 142.0, p50Ms: 2.4, p95Ms: 4.2, p99Ms: 6.8)
        ],
        datasetLabel: "synthetic"
    )
}
