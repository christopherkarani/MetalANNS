import Foundation

public final class PQVectorBuffer: QuantizedStorage, @unchecked Sendable {
    public let originalDimension: Int
    public private(set) var count: Int
    public let capacity: Int

    private let pq: ProductQuantizer
    private let codeLength: Int
    private var codes: [UInt8]

    public init(capacity: Int, dim: Int, pq: ProductQuantizer) throws {
        guard capacity > 0 else {
            throw ANNSError.constructionFailed("PQVectorBuffer capacity must be greater than zero")
        }
        let expectedDimension = pq.numSubspaces * pq.subspaceDimension
        guard dim == expectedDimension else {
            throw ANNSError.dimensionMismatch(expected: expectedDimension, got: dim)
        }

        self.originalDimension = dim
        self.count = 0
        self.capacity = capacity
        self.pq = pq
        self.codeLength = pq.numSubspaces
        self.codes = [UInt8](repeating: 0, count: capacity * pq.numSubspaces)
    }

    public func insert(vector: [Float], at index: Int) throws {
        let encoded = try pq.encode(vector: vector)
        try insertEncoded(code: encoded, at: index)
    }

    public func insertEncoded(code: [UInt8], at index: Int) throws {
        guard index >= 0, index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) out of bounds for capacity \(capacity)")
        }
        guard code.count == codeLength else {
            throw ANNSError.dimensionMismatch(expected: codeLength, got: code.count)
        }

        let offset = index * codeLength
        for i in 0..<codeLength {
            codes[offset + i] = code[i]
        }
        if count < index + 1 {
            count = index + 1
        }
    }

    public func approximateDistance(query: [Float], to index: UInt32, metric: Metric) -> Float {
        guard query.count == originalDimension else {
            return Float.greatestFiniteMagnitude
        }
        let indexInt = Int(index)
        guard indexInt >= 0, indexInt < count else {
            return Float.greatestFiniteMagnitude
        }
        return pq.approximateDistance(query: query, codes: code(at: indexInt), metric: metric)
    }

    public func reconstruct(at index: UInt32) -> [Float] {
        let indexInt = Int(index)
        guard indexInt >= 0, indexInt < count else {
            return []
        }
        do {
            return try pq.reconstruct(codes: code(at: indexInt))
        } catch {
            return []
        }
    }

    public func code(at index: Int) -> [UInt8] {
        guard index >= 0, index < count else {
            return []
        }
        let offset = index * codeLength
        return Array(codes[offset..<(offset + codeLength)])
    }

    public var compressedCodeBytes: Int {
        count * codeLength
    }
}
