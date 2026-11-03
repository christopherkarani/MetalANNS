import Foundation
import Metal

/// Reallocation helpers for growable mutable indexes. Every mutable storage
/// type is a flat row-major buffer with a fixed per-row stride, so growing is a
/// single byte copy of the old buffer into a larger one.
package enum StorageGrowth {
    /// Capacity to allocate when `required` rows must fit: at least double the
    /// current capacity so repeated single inserts are amortized O(1).
    package static func grownCapacity(current: Int, required: Int) -> Int {
        max(required, current * 2, 16)
    }

    /// Returns a copy of `storage` with room for `capacity` rows, or nil when
    /// the storage kind cannot grow (mmap / disk-backed / quantized).
    package static func grow(
        _ storage: any VectorStorage,
        to capacity: Int,
        device: MTLDevice?
    ) throws -> (any VectorStorage)? {
        precondition(capacity >= storage.capacity)
        let device = device ?? storage.buffer.device
        let dim = storage.dim
        let grown: any VectorStorage
        switch storage {
        case is VectorBuffer:
            grown = try VectorBuffer(capacity: capacity, dim: dim, device: device)
        case is Float16VectorBuffer:
            grown = try Float16VectorBuffer(capacity: capacity, dim: dim, device: device)
        case is BinaryVectorBuffer:
            grown = try BinaryVectorBuffer(capacity: capacity, dim: dim, device: device)
        default:
            return nil
        }
        copyBytes(from: storage.buffer, to: grown.buffer)
        grown.setCount(storage.count)
        return grown
    }

    /// Drops the process-wide search caches keyed by `buffer` (host norms,
    /// int8 codes, IVF partitions). Call when replacing a vector buffer.
    package static func releaseSearchCaches(for buffer: MTLBuffer) {
        FlatGPUSearch.invalidateHostNormCache(buffer: buffer)
        Int8CodeCache.invalidate(buffer: buffer)
        IVFFlatSearch.invalidate(buffer: buffer)
    }

    package static func grow(_ graph: GraphBuffer, to capacity: Int, device: MTLDevice?) throws -> GraphBuffer {
        precondition(capacity >= graph.capacity)
        let grown = try GraphBuffer(
            capacity: capacity,
            degree: graph.degree,
            device: device ?? graph.adjacencyBuffer.device
        )
        copyBytes(from: graph.adjacencyBuffer, to: grown.adjacencyBuffer)
        copyBytes(from: graph.distanceBuffer, to: grown.distanceBuffer)
        grown.setCount(graph.nodeCount)
        return grown
    }

    private static func copyBytes(from source: MTLBuffer, to destination: MTLBuffer) {
        precondition(destination.length >= source.length)
        destination.contents().copyMemory(from: source.contents(), byteCount: source.length)
    }
}
