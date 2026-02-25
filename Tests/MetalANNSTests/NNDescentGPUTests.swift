import Metal
import Testing
@testable import MetalANNSCore

@Suite("GPU NN-Descent Tests")
struct NNDescentGPUTests {
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
}
