import Foundation
import Metal

/// GPU-resident flat buffer storing `capacity` vectors of `dim` dimensions.
/// Layout: vector[i] starts at offset `i * dim` in the underlying Float32 buffer.
public final class VectorBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let dim: Int
    public let capacity: Int
    public private(set) var count: Int = 0

    private let rawPointer: UnsafeMutablePointer<Float>

    public init(capacity: Int, dim: Int, device: MTLDevice? = nil) throws(ANNSError) {
        guard capacity >= 0, dim > 0 else {
            throw ANNSError.constructionFailed("VectorBuffer requires capacity >= 0 and dim > 0")
        }

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        let elementCount = capacity * dim
        let byteLength = elementCount * MemoryLayout<Float>.stride

        guard let buffer = metalDevice.makeBuffer(length: max(byteLength, 4), options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate VectorBuffer")
        }

        self.buffer = buffer
        self.dim = dim
        self.capacity = capacity
        self.rawPointer = buffer.contents().bindMemory(to: Float.self, capacity: max(elementCount, 1))
    }

    public func setCount(_ newCount: Int) {
        count = newCount
    }

    public func insert(vector: [Float], at index: Int) throws(ANNSError) {
        guard vector.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
        guard index >= 0, index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) is out of bounds for capacity \(capacity)")
        }

        let offset = index * dim
        vector.withUnsafeBufferPointer { source in
            guard let baseAddress = source.baseAddress else { return }
            rawPointer.advanced(by: offset).update(from: baseAddress, count: dim)
        }
    }

    public func batchInsert(vectors: [[Float]], startingAt start: Int) throws(ANNSError) {
        for (offset, vector) in vectors.enumerated() {
            try insert(vector: vector, at: start + offset)
        }
    }

    public func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < capacity, "Index out of bounds")
        let offset = index * dim
        let pointer = rawPointer.advanced(by: offset)
        return Array(UnsafeBufferPointer(start: pointer, count: dim))
    }

    public var floatPointer: UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(start: rawPointer, count: capacity * dim)
    }
}

extension VectorBuffer: VectorStorage {
    public var isFloat16: Bool { false }
}
