import Foundation
import MetalANNS

enum BenchmarkIDParser {
    static func uint32(from id: String) -> UInt32? {
        if let value = UInt32(id) {
            return value
        }

        if let underscore = id.firstIndex(of: "_") {
            let suffix = id[id.index(after: underscore)...]
            return UInt32(suffix)
        }

        return nil
    }

    static func uint32Set(
        from results: [SearchResult],
        limit: Int
    ) -> Set<UInt32> {
        let capacity = max(0, min(limit, results.count))
        if capacity == 0 {
            return []
        }

        var ids: Set<UInt32> = []
        ids.reserveCapacity(capacity)

        for result in results.prefix(capacity) {
            if let parsed = uint32(from: result.id) {
                ids.insert(parsed)
            }
        }

        return ids
    }
}
