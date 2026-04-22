import Foundation

public struct LatencyDistribution: Sendable {
    public let countSamples: Int
    public let meanMs: Double
    public let stdDevMs: Double
    public let minMs: Double
    public let maxMs: Double
    public let p50Ms: Double
    public let p90Ms: Double
    public let p95Ms: Double
    public let p99Ms: Double
    public let p999Ms: Double
    public let histogramBuckets: [HistogramBucket]

    public struct HistogramBucket: Sendable {
        public let lowerMs: Double
        public let upperMs: Double
        public let count: Int

        public init(lowerMs: Double, upperMs: Double, count: Int) {
            self.lowerMs = lowerMs
            self.upperMs = upperMs
            self.count = count
        }
    }

    public init(
        countSamples: Int,
        meanMs: Double,
        stdDevMs: Double,
        minMs: Double,
        maxMs: Double,
        p50Ms: Double,
        p90Ms: Double,
        p95Ms: Double,
        p99Ms: Double,
        p999Ms: Double,
        histogramBuckets: [HistogramBucket]
    ) {
        self.countSamples = countSamples
        self.meanMs = meanMs
        self.stdDevMs = stdDevMs
        self.minMs = minMs
        self.maxMs = maxMs
        self.p50Ms = p50Ms
        self.p90Ms = p90Ms
        self.p95Ms = p95Ms
        self.p99Ms = p99Ms
        self.p999Ms = p999Ms
        self.histogramBuckets = histogramBuckets
    }

    public static func empty() -> LatencyDistribution {
        LatencyDistribution(
            countSamples: 0,
            meanMs: 0,
            stdDevMs: 0,
            minMs: 0,
            maxMs: 0,
            p50Ms: 0,
            p90Ms: 0,
            p95Ms: 0,
            p99Ms: 0,
            p999Ms: 0,
            histogramBuckets: []
        )
    }

    public static func compute(fromLatenciesMs latencies: [Double], bucketCount: Int = 32) -> LatencyDistribution {
        guard !latencies.isEmpty else {
            return .empty()
        }

        let sorted = latencies.sorted()
        let count = sorted.count
        let minValue = sorted.first ?? 0
        let maxValue = sorted.last ?? 0
        let avg = mean(in: sorted)
        let stddev = standardDeviation(in: sorted, mean: avg)

        let buckets = makeHistogramBuckets(
            sortedLatencies: sorted,
            bucketCount: max(1, bucketCount)
        )

        return LatencyDistribution(
            countSamples: count,
            meanMs: avg,
            stdDevMs: stddev,
            minMs: minValue,
            maxMs: maxValue,
            p50Ms: percentile(0.50, in: sorted),
            p90Ms: percentile(0.90, in: sorted),
            p95Ms: percentile(0.95, in: sorted),
            p99Ms: percentile(0.99, in: sorted),
            p999Ms: percentile(0.999, in: sorted),
            histogramBuckets: buckets
        )
    }

    public func renderASCIIHistogram(width: Int = 40) -> String {
        guard !histogramBuckets.isEmpty else {
            return "(no samples)"
        }

        let maxCount = histogramBuckets.reduce(0) { Swift.max($0, $1.count) }
        guard maxCount > 0 else {
            return "(no samples)"
        }

        let safeWidth = max(1, width)
        var lines: [String] = []
        let labelWidth = 18
        for bucket in histogramBuckets {
            let label = String(format: "[%7.3f,%7.3f)", bucket.lowerMs, bucket.upperMs)
            let barLength = Int((Double(bucket.count) / Double(maxCount)) * Double(safeWidth))
            let bar = String(repeating: "#", count: barLength)
            let padded = label.count >= labelWidth
                ? label
                : label + String(repeating: " ", count: labelWidth - label.count)
            lines.append("\(padded) | \(bar) \(bucket.count)")
        }
        return lines.joined(separator: "\n")
    }

    public func cdfCSV() -> String {
        guard countSamples > 0 else {
            return "latencyMs,cumulativeFraction\n"
        }

        var lines: [String] = ["latencyMs,cumulativeFraction"]
        var cumulative = 0
        let total = histogramBuckets.reduce(0) { $0 + $1.count }
        let denominator = max(1, total)
        for bucket in histogramBuckets {
            cumulative += bucket.count
            let fraction = Double(cumulative) / Double(denominator)
            lines.append(String(format: "%.6f,%.6f", bucket.upperMs, fraction))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    public func histogramCSV() -> String {
        var lines: [String] = ["lowerMs,upperMs,count"]
        for bucket in histogramBuckets {
            lines.append(String(format: "%.6f,%.6f,%d", bucket.lowerMs, bucket.upperMs, bucket.count))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func percentile(_ p: Double, in sorted: [Double]) -> Double {
        guard !sorted.isEmpty else {
            return 0
        }
        let rank = Int(ceil(p * Double(sorted.count))) - 1
        let index = min(max(rank, 0), sorted.count - 1)
        return sorted[index]
    }

    private static func mean(in values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func standardDeviation(in values: [Double], mean avg: Double) -> Double {
        guard values.count > 1 else {
            return 0
        }
        let squaredSum = values.reduce(0.0) { running, value in
            let difference = value - avg
            return running + (difference * difference)
        }
        return sqrt(squaredSum / Double(values.count))
    }

    private static func makeHistogramBuckets(
        sortedLatencies: [Double],
        bucketCount: Int
    ) -> [HistogramBucket] {
        guard let minValue = sortedLatencies.first,
              let maxValue = sortedLatencies.last else {
            return []
        }

        if minValue == maxValue || bucketCount <= 1 {
            return [
                HistogramBucket(
                    lowerMs: minValue,
                    upperMs: maxValue,
                    count: sortedLatencies.count
                )
            ]
        }

        // Use log-spaced buckets when min > 0; otherwise use linear spacing.
        let edges: [Double]
        if minValue > 0 {
            edges = logSpacedEdges(min: minValue, max: maxValue, bucketCount: bucketCount)
        } else {
            // Shift to a small epsilon to avoid log(0).
            let epsilon = max(maxValue * 1e-9, 1e-9)
            let lower = epsilon
            let logEdges = logSpacedEdges(min: lower, max: max(maxValue, lower * 2), bucketCount: bucketCount - 1)
            // Prepend an initial edge at 0 so the first bucket captures zeros.
            var combined = [0.0]
            combined.append(contentsOf: logEdges)
            edges = combined
        }

        var buckets: [HistogramBucket] = []
        buckets.reserveCapacity(edges.count - 1)
        var cursor = 0
        let total = sortedLatencies.count

        for i in 0..<(edges.count - 1) {
            let lower = edges[i]
            let upper = edges[i + 1]
            let isLast = (i == edges.count - 2)
            var count = 0
            while cursor < total {
                let value = sortedLatencies[cursor]
                let inRange = isLast ? (value <= upper) : (value < upper)
                if value >= lower && inRange {
                    count += 1
                    cursor += 1
                } else if value < lower {
                    // Should not happen with sorted input, but guard anyway.
                    cursor += 1
                } else {
                    break
                }
            }
            buckets.append(HistogramBucket(lowerMs: lower, upperMs: upper, count: count))
        }

        // Any remaining samples (numerical edge cases) go into the last bucket.
        if cursor < total, var last = buckets.last {
            let remaining = total - cursor
            last = HistogramBucket(
                lowerMs: last.lowerMs,
                upperMs: last.upperMs,
                count: last.count + remaining
            )
            buckets[buckets.count - 1] = last
        }

        return buckets
    }

    private static func logSpacedEdges(min lo: Double, max hi: Double, bucketCount: Int) -> [Double] {
        guard bucketCount >= 1, lo > 0, hi > lo else {
            return [lo, max(hi, lo)]
        }
        let logLo = log(lo)
        let logHi = log(hi)
        let step = (logHi - logLo) / Double(bucketCount)
        var edges: [Double] = []
        edges.reserveCapacity(bucketCount + 1)
        for i in 0...bucketCount {
            edges.append(exp(logLo + Double(i) * step))
        }
        // Ensure first/last match exactly (avoid floating drift).
        edges[0] = lo
        edges[edges.count - 1] = hi
        return edges
    }
}
