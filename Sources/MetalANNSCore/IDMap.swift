import Foundation

/// Bidirectional mapping between external String IDs and internal UInt32 IDs.
public struct IDMap: Sendable, Codable {
    private var externalToInternal: [String: UInt32] = [:]
    private var internalToExternal: [UInt32: String] = [:]
    private var nextID: UInt32 = 0

    public init() {}

    /// The next internal ID that will be assigned.
    public var nextInternalID: UInt32 {
        nextID
    }

    /// Returns true if at least `count` additional IDs can be assigned.
    public func canAllocate(_ count: Int) -> Bool {
        guard count >= 0 else {
            return false
        }
        // Reserve UInt32.max as invalid/sentinel in graph structures.
        let remaining = Int(UInt32.max &- nextID)
        return count <= remaining
    }

    /// Assigns a new internal ID. Returns nil if the external ID already exists.
    public mutating func assign(externalID: String) -> UInt32? {
        guard externalToInternal[externalID] == nil else {
            return nil
        }
        guard nextID < UInt32.max else {
            return nil
        }

        let internalID = nextID
        externalToInternal[externalID] = internalID
        internalToExternal[internalID] = externalID
        nextID += 1
        return internalID
    }

    public func internalID(for externalID: String) -> UInt32? {
        externalToInternal[externalID]
    }

    public func externalID(for internalID: UInt32) -> String? {
        internalToExternal[internalID]
    }

    public var count: Int {
        externalToInternal.count
    }
}
