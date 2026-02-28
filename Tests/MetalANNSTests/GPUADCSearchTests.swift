import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("GPU ADC Search Tests")
struct GPUADCSearchTests {
    @Test("GPU ADC distances match CPU ADC distances (tolerance 1e-3)")
    func gpuDistancesMatchCPU() async throws {
        #if targetEnvironment(simulator)
        return
        #else
        guard let context = try? MetalContext() else { return }

        let pq = try trainPQ(dim: 64, M: 8)
        let corpus = makeRandomVectors(count: 200, dim: 64, seed: 302)
        let codes = try corpus.map { try pq.encode(vector: $0) }
        let query = makeRandomVectors(count: 1, dim: 64, seed: 303)[0]

        let gpu = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: codes
        )

        #expect(gpu.count == codes.count)
        for i in 0..<codes.count {
            let cpu = pq.approximateDistance(query: query, codes: codes[i], metric: .l2)
            #expect(abs(gpu[i] - cpu) < 1e-3, "Difference at \(i): cpu=\(cpu) gpu=\(gpu[i])")
        }
        #endif
    }

    @Test("Empty code input returns empty distance output")
    func emptyCodesReturnsEmpty() async throws {
        #if targetEnvironment(simulator)
        return
        #else
        guard let context = try? MetalContext() else { return }

        let pq = try trainPQ(dim: 64, M: 8)
        let query = makeRandomVectors(count: 1, dim: 64, seed: 304)[0]

        let distances = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: []
        )

        #expect(distances.isEmpty)
        #endif
    }

    @Test("Flatten codebooks uses M x Ks x subspaceDim layout")
    func flattenCodebooksCorrectLayout() throws {
        let pq = try trainPQ(dim: 16, M: 4)
        let flattened = GPUADCSearch.flattenCodebooks(from: pq)
        let expectedCount = pq.numSubspaces * pq.centroidsPerSubspace * pq.subspaceDimension

        #expect(flattened.count == expectedCount)
        #expect(Array(flattened.prefix(pq.subspaceDimension)) == pq.codebooks[0][0])
    }

    @Test("Providing cached flat codebooks matches nil flatCodebooks path")
    func cachedFlatCodebooksSkipsRecomputation() async throws {
        #if targetEnvironment(simulator)
        return
        #else
        guard let context = try? MetalContext() else { return }

        let pq = try trainPQ(dim: 64, M: 8)
        let corpus = makeRandomVectors(count: 200, dim: 64, seed: 305)
        let codes = try corpus.map { try pq.encode(vector: $0) }
        let query = makeRandomVectors(count: 1, dim: 64, seed: 306)[0]
        let cachedFlat = GPUADCSearch.flattenCodebooks(from: pq)

        let cached = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: codes,
            flatCodebooks: cachedFlat
        )

        let recomputed = try await GPUADCSearch.computeDistances(
            context: context,
            query: query,
            pq: pq,
            codes: codes,
            flatCodebooks: nil
        )

        #expect(cached.count == recomputed.count)
        for i in 0..<cached.count {
            #expect(abs(cached[i] - recomputed[i]) < 1e-6)
        }
        #endif
    }
}

private func trainPQ(dim: Int = 64, M: Int = 8) throws -> ProductQuantizer {
    let vectors = makeRandomVectors(count: 500, dim: dim, seed: 301)
    return try ProductQuantizer.train(
        vectors: vectors,
        numSubspaces: M,
        centroidsPerSubspace: 256,
        maxIterations: 6
    )
}

private func makeRandomVectors(count: Int, dim: Int, seed: UInt64) -> [[Float]] {
    var rng = SeededGenerator(state: seed == 0 ? 1 : seed)
    return (0..<count).map { _ in
        (0..<dim).map { _ in Float.random(in: -1.0...1.0, using: &rng) }
    }
}

private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
