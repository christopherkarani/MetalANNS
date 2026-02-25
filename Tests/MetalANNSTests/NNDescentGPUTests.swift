import Metal
import Testing
@testable import MetalANNSCore

@Suite("GPU NN-Descent Tests")
struct NNDescentGPUTests {
    private func withVectorBuffer<T>(
        _ values: [Float],
        _ body: (UnsafeBufferPointer<Float>) async throws -> T
    ) async throws -> T {
        let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: values.count)
        buffer.initialize(from: values)
        defer {
            buffer.deinitialize()
            buffer.deallocate()
        }
        return try await body(UnsafeBufferPointer(buffer))
    }

    @Test("Random init produces valid graph")
    func randomInitValid() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let context = try MetalContext()
        let nodeCount = 100
        let degree = 8
        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: context.device)

        try await NNDescentGPU.randomInit(
            context: context,
            graph: graph,
            nodeCount: nodeCount,
            seed: 42
        )

        for node in 0..<nodeCount {
            let neighborIDs = graph.neighborIDs(of: node)
            #expect(neighborIDs.count == degree)
            #expect(Set(neighborIDs).count == degree, "Duplicate neighbors at node \(node)")

            for neighborID in neighborIDs {
                #expect(neighborID != UInt32(node), "Self-loop at node \(node)")
                #expect(neighborID < UInt32(nodeCount), "Out-of-range neighbor \(neighborID) at node \(node)")
            }
        }
    }

    @Test("Full GPU NN-Descent construction recall > 0.80")
    func fullGPUConstruction() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let nodeCount = 200
        let dim = 16
        let degree = 8
        let maxIterations = 15

        let vectors = (0..<nodeCount).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1) }
        }

        let context = try MetalContext()
        let vectorBuffer = try VectorBuffer(capacity: nodeCount, dim: dim, device: context.device)
        try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
        vectorBuffer.setCount(nodeCount)

        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: context.device)

        try await NNDescentGPU.build(
            context: context,
            vectors: vectorBuffer,
            graph: graph,
            nodeCount: nodeCount,
            metric: .cosine,
            maxIterations: maxIterations
        )

        let backend = AccelerateBackend()
        let flat = vectors.flatMap { $0 }
        var totalRecall: Float = 0

        for node in 0..<nodeCount {
            let distances = try await withVectorBuffer(flat) { pointer in
                try await backend.computeDistances(
                    query: vectors[node],
                    vectors: pointer,
                    vectorCount: nodeCount,
                    dim: dim,
                    metric: .cosine
                )
            }

            let exactTopK = Set(
                distances.enumerated()
                    .filter { $0.offset != node }
                    .sorted { $0.element < $1.element }
                    .prefix(degree)
                    .map { UInt32($0.offset) }
            )

            let graphTopK = Set(
                graph.neighborIDs(of: node).filter { $0 != UInt32.max && $0 != UInt32(node) && $0 < UInt32(nodeCount) }
            )

            let overlap = exactTopK.intersection(graphTopK).count
            totalRecall += Float(overlap) / Float(degree)
        }

        let averageRecall = totalRecall / Float(nodeCount)
        #expect(averageRecall > 0.80, "Average recall \\(averageRecall) below 0.80")
    }
}
