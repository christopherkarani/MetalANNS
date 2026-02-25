import Foundation

public struct SoftDeletion: Sendable, Codable {
    private var deletedIDs: Set<UInt32> = []

    public init() {}

    public mutating func markDeleted(_ internalID: UInt32) {
        deletedIDs.insert(internalID)
    }

    public mutating func undelete(_ internalID: UInt32) {
        deletedIDs.remove(internalID)
    }

    public func isDeleted(_ internalID: UInt32) -> Bool {
        deletedIDs.contains(internalID)
    }

    public var deletedCount: Int {
        deletedIDs.count
    }

    /// All internal IDs currently marked as deleted.
    public var allDeletedIDs: Set<UInt32> {
        deletedIDs
    }

    public func filterResults(_ results: [SearchResult]) -> [SearchResult] {
        results.filter { !isDeleted($0.internalID) }
    }
}
