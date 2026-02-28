import Foundation

public struct MetricsSnapshot: Sendable, Codable {
    public let insertCount: Int
    public let searchCount: Int
    public let batchSearchCount: Int
    public let mergeCount: Int
    public let searchP50LatencyMs: Double
    public let searchP99LatencyMs: Double
    public let timestamp: Date

    public init(
        insertCount: Int,
        searchCount: Int,
        batchSearchCount: Int,
        mergeCount: Int,
        searchP50LatencyMs: Double,
        searchP99LatencyMs: Double,
        timestamp: Date
    ) {
        self.insertCount = insertCount
        self.searchCount = searchCount
        self.batchSearchCount = batchSearchCount
        self.mergeCount = mergeCount
        self.searchP50LatencyMs = searchP50LatencyMs
        self.searchP99LatencyMs = searchP99LatencyMs
        self.timestamp = timestamp
    }
}
