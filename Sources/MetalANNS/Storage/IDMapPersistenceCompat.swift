import Foundation
import MetalANNSCore

extension IDMap {
    static func makeForPersistence(rows: [(String, UInt32)], nextID: UInt32) -> IDMap {
        struct IDMapPayload: Codable {
            let externalToInternal: [String: UInt32]
            let internalToExternal: [UInt32: String]
            let nextID: UInt32
        }

        let externalToInternal = Dictionary(uniqueKeysWithValues: rows.map { ($0.0, $0.1) })
        let internalToExternal = Dictionary(uniqueKeysWithValues: rows.map { ($0.1, $0.0) })
        let payload = IDMapPayload(
            externalToInternal: externalToInternal,
            internalToExternal: internalToExternal,
            nextID: nextID
        )

        if
            let data = try? JSONEncoder().encode(payload),
            let decoded = try? JSONDecoder().decode(IDMap.self, from: data)
        {
            return decoded
        }

        var fallback = IDMap()
        for (externalID, _) in rows.sorted(by: { $0.1 < $1.1 }) {
            _ = fallback.assign(externalID: externalID)
        }
        return fallback
    }
}
