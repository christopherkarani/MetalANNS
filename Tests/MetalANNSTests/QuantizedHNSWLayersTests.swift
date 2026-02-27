import Foundation
import Testing
@testable import MetalANNSCore

@Suite("QuantizedHNSWLayers Tests")
struct QuantizedHNSWLayersTests {
    @Test("Construct skip layer")
    func constructSkipLayer() throws {
        let base = SkipLayer(
            nodeToLayerIndex: [0: 0, 1: 1],
            layerIndexToNode: [0, 1],
            adjacency: [[1], [0]]
        )
        let pq = try ProductQuantizer.train(
            vectors: makeVectors(count: 256, dim: 128),
            numSubspaces: 4,
            centroidsPerSubspace: 256,
            maxIterations: 2
        )
        let codes = [try pq.encode(vector: makeVectors(count: 2, dim: 128)[0]),
                     try pq.encode(vector: makeVectors(count: 2, dim: 128)[1])]
        let layer = QuantizedSkipLayer(base: base, pq: pq, codes: codes)

        #expect(layer.base.adjacency.count == 2)
        #expect(layer.codes.count == 2)
    }

    @Test("Nil-PQ skip layer")
    func nilPQSkipLayer() {
        let base = SkipLayer(
            nodeToLayerIndex: [0: 0],
            layerIndexToNode: [0],
            adjacency: [[]]
        )
        let layer = QuantizedSkipLayer(base: base, pq: nil, codes: [])
        #expect(layer.pq == nil)
        #expect(layer.codes.isEmpty)
    }

    @Test("Construct quantized layers")
    func constructQuantizedLayers() {
        let baseA = SkipLayer(
            nodeToLayerIndex: [0: 0],
            layerIndexToNode: [0],
            adjacency: [[]]
        )
        let baseB = SkipLayer(
            nodeToLayerIndex: [1: 0],
            layerIndexToNode: [1],
            adjacency: [[]]
        )

        let layer1 = QuantizedSkipLayer(base: baseA, pq: nil, codes: [])
        let layer2 = QuantizedSkipLayer(base: baseB, pq: nil, codes: [])

        let q = QuantizedHNSWLayers(
            quantizedLayers: [layer1, layer2],
            maxLayer: 2,
            entryPoint: 1
        )

        #expect(q.maxLayer == 2)
        #expect(q.entryPoint == 1)
        #expect(q.quantizedLayer(at: 1) != nil)
        #expect(q.quantizedLayer(at: 2) != nil)
    }

    @Test("Layer lookup handles maxLayer count mismatch safely")
    func layerLookupHandlesMismatchSafely() {
        let base = SkipLayer(
            nodeToLayerIndex: [0: 0],
            layerIndexToNode: [0],
            adjacency: [[]]
        )
        let q = QuantizedHNSWLayers(
            quantizedLayers: [QuantizedSkipLayer(base: base, pq: nil, codes: [])],
            maxLayer: 2,
            entryPoint: 0
        )

        #expect(q.quantizedLayer(at: 1) != nil)
        #expect(q.quantizedLayer(at: 2) == nil)
    }

    @Test("Codable round trip")
    func codableRoundTrip() throws {
        let base = SkipLayer(
            nodeToLayerIndex: [7: 0],
            layerIndexToNode: [7],
            adjacency: [[7]]
        )
        let layers = QuantizedHNSWLayers(
            quantizedLayers: [QuantizedSkipLayer(base: base, pq: nil, codes: [])],
            maxLayer: 1,
            entryPoint: 7
        )

        let encoded = try JSONEncoder().encode(layers)
        let decoded = try JSONDecoder().decode(QuantizedHNSWLayers.self, from: encoded)

        #expect(decoded.maxLayer == 1)
        #expect(decoded.entryPoint == 7)
    }

    private func makeVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let x = Float((row + 1) * (col + 3))
                return sin(x * 0.013) + cos(x * 0.007)
            }
        }
    }
}
