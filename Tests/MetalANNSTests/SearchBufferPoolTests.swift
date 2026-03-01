import Metal
import Testing
@testable import MetalANNSCore

@Suite("SearchBufferPool Tests")
struct SearchBufferPoolTests {
    @Test func acquireAndReleaseReturnsSameBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let pool = SearchBufferPool(device: device)

        let b1 = try pool.acquire(queryDim: 128, maxK: 10)
        let ptr1 = b1.queryBuffer.gpuAddress
        pool.release(b1)

        let b2 = try pool.acquire(queryDim: 128, maxK: 10)
        #expect(b2.queryBuffer.gpuAddress == ptr1)
        pool.release(b2)
    }

    @Test func acquireLargerDimAllocatesNew() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let pool = SearchBufferPool(device: device)

        let small = try pool.acquire(queryDim: 64, maxK: 10)
        pool.release(small)

        let large = try pool.acquire(queryDim: 512, maxK: 10)
        #expect(large.queryBuffer.length >= 512 * MemoryLayout<Float>.stride)
        pool.release(large)
    }

    @Test func concurrentAcquireReturnsDistinctBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let pool = SearchBufferPool(device: device)

        let b1 = try pool.acquire(queryDim: 128, maxK: 10)
        let b2 = try pool.acquire(queryDim: 128, maxK: 10)
        #expect(b1.queryBuffer.gpuAddress != b2.queryBuffer.gpuAddress)
        pool.release(b1)
        pool.release(b2)
    }
}
