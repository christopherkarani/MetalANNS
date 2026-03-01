import Foundation

/// Bidirectional mapping between external String IDs and internal UInt32 IDs.
public struct IDMap: Sendable, Codable {
    private var externalToInternal: [String: UInt32] = [:]
    private var internalToExternal: [UInt32: String] = [:]
    private var nextID: UInt32 = 0

    public init() {}

    /// Assigns a new internal ID. Returns nil if the external ID already exists.
    public mutating func assign(externalID: String) -> UInt32? {
        guard externalToInternal[externalID] == nil else {
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

    public var nextInternalID: UInt32 {
        nextID
    }

    public var count: Int {
        externalToInternal.count
    }
}

//
// MARK: - Persistence Reconstruction

extension IDMap {
    package static func makeForPersistence(rows: [(String, UInt32)], nextID: UInt32) -> IDMap {
        var map = IDMap()
        map.externalToInternal.removeAll()
        map.internalToExternal.removeAll()
        for (externalID, internalID) in rows {
            map.externalToInternal[externalID] = internalID
            map.internalToExternal[internalID] = externalID
        }
        map.nextID = nextID
        return map
    }
}
