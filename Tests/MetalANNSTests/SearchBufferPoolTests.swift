import Metal
import Testing
@testable import MetalANNSCore

private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

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

    @Test func fullGPUSearchCorrectAfterPoolRefactor() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping: no Metal device")
            return
        }
        let context = try MetalContext()

        let dim = 16
        let nodeCount = 32
        let degree = 8
        var rng = SeededGenerator(state: 42)
        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1, using: &rng) }
        }

        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
        for (i, vector) in vectors.enumerated() {
            try vectorBuffer.insert(vector: vector, at: i)
        }
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)
        try await NNDescentGPU.build(
            context: context,
            vectors: vectorBuffer,
            graph: graph,
            nodeCount: nodeCount,
            metric: .cosine
        )

        let r1 = try await FullGPUSearch.search(
            context: context,
            query: vectors[0],
            vectors: vectorBuffer,
            graph: graph,
            entryPoint: 0,
            k: 5,
            ef: 16,
            metric: .cosine
        )
        let r2 = try await FullGPUSearch.search(
            context: context,
            query: vectors[1],
            vectors: vectorBuffer,
            graph: graph,
            entryPoint: 0,
            k: 5,
            ef: 16,
            metric: .cosine
        )

        #expect(r1.count > 0, "First search returned no results")
        #expect(r2.count > 0, "Second search returned no results")
        #expect(r1[0].score < 0.1, "First result should be near-exact match (query is in index)")
    }
}
