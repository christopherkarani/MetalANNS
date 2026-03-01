import Foundation
import Metal

/// Thread-safe pool of reusable MTLBuffer triplets for GPU search operations.
/// Eliminates per-search allocation overhead in FullGPUSearch.
public final class SearchBufferPool: @unchecked Sendable {
    public struct Buffers: @unchecked Sendable {
        public let queryBuffer: MTLBuffer
        public let outputDistanceBuffer: MTLBuffer
        public let outputIDBuffer: MTLBuffer
        public let queryDim: Int
        public let maxK: Int
    }

    private let device: MTLDevice
    private var available: [Buffers] = []
    private let lock = NSLock()

    public init(device: MTLDevice) {
        self.device = device
    }

    /// Returns a buffer set with capacity >= requested dimensions.
    /// If no pooled entry fits, allocates new buffers.
    public func acquire(queryDim: Int, maxK: Int) throws -> Buffers {
        lock.lock()
        defer { lock.unlock() }

        if let index = available.firstIndex(where: {
            $0.queryDim >= queryDim && $0.maxK >= maxK
        }) {
            return available.remove(at: index)
        }

        return try allocate(queryDim: queryDim, maxK: maxK)
    }

    /// Returns buffers to the pool for future reuse.
    public func release(_ buffers: Buffers) {
        lock.lock()
        defer { lock.unlock() }
        available.append(buffers)
    }

    private func allocate(queryDim: Int, maxK: Int) throws -> Buffers {
        let floatSize = MemoryLayout<Float>.stride
        let uintSize = MemoryLayout<UInt32>.stride

        guard
            let qBuf = device.makeBuffer(
                length: queryDim * floatSize,
                options: .storageModeShared
            ),
            let dBuf = device.makeBuffer(
                length: max(maxK * floatSize, floatSize),
                options: .storageModeShared
            ),
            let iBuf = device.makeBuffer(
                length: max(maxK * uintSize, uintSize),
                options: .storageModeShared
            )
        else {
            throw ANNSError.searchFailed("Failed to allocate search buffer pool entry")
        }

        return Buffers(
            queryBuffer: qBuf,
            outputDistanceBuffer: dBuf,
            outputIDBuffer: iBuf,
            queryDim: queryDim,
            maxK: maxK
        )
    }
}
