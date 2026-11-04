import Foundation

/// Rejects NaN and ±infinity at API boundaries. A single non-finite component
/// makes every distance to that vector non-finite, which silently corrupts
/// graph construction, pruning, and top-k ordering.
package enum VectorValidation {
    /// True when every element is finite. Branch-free so the loop vectorizes;
    /// a value is non-finite exactly when all exponent bits are set.
    @inline(__always)
    package static func allFinite(_ values: UnsafeBufferPointer<Float>) -> Bool {
        var nonFinite: UInt32 = 0
        for value in values {
            nonFinite |= (value.bitPattern & 0x7F80_0000) == 0x7F80_0000 ? 1 : 0
        }
        return nonFinite == 0
    }

    package static func requireFinite(_ vector: [Float], index: Int = 0) throws(ANNSError) {
        guard vector.withUnsafeBufferPointer(allFinite) else {
            throw .nonFiniteVector(index: index)
        }
    }

    package static func requireFinite(rows: [[Float]]) throws(ANNSError) {
        for (index, row) in rows.enumerated() {
            try requireFinite(row, index: index)
        }
    }

    /// Validates a row-major `rowCount × dimension` buffer, reporting the
    /// first offending row.
    package static func requireFinite(flat: [Float], dimension: Int) throws(ANNSError) {
        let firstBadRow: Int? = flat.withUnsafeBufferPointer { buffer in
            guard !allFinite(buffer) else { return nil }
            var row = 0
            while row * dimension < buffer.count {
                let slice = UnsafeBufferPointer(rebasing: buffer[(row * dimension)..<((row + 1) * dimension)])
                if !allFinite(slice) { return row }
                row += 1
            }
            return nil
        }
        if let firstBadRow {
            throw .nonFiniteVector(index: firstBadRow)
        }
    }
}
