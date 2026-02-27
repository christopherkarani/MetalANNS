import Foundation
import Testing
@testable import MetalANNSCore

@Suite("QuantizedHNSWBuilder Tests")
struct QuantizedHNSWBuilderTests {
    @Test("Builds layers from graph")
    func buildsLayersFromGraph() throws {
        let vectors = makeVectors(count: 500, dim: 128)
        let skip = makeSkipLayer(nodeIDs: Array(0..<260))
        let hnsw = HNSWLayers(
            layers: [skip],
            maxLayer: 1,
            entryPoint: 0
        )

        let q = try QuantizedHNSWBuilder.build(
            from: hnsw,
            vectors: vectors,
            config: QuantizedHNSWConfiguration(pqSubspaces: 4),
            metric: .cosine
        )

        #expect(q.maxLayer >= 1)
    }

    @Test("PQ trained per layer")
    func pqTrainedPerLayer() throws {
        let vectors = makeVectors(count: 500, dim: 128)
        let skip = makeSkipLayer(nodeIDs: Array(0..<260))
        let hnsw = HNSWLayers(layers: [skip], maxLayer: 1, entryPoint: 0)

        let q = try QuantizedHNSWBuilder.build(
            from: hnsw,
            vectors: vectors,
            config: .default,
            metric: .cosine
        )

        #expect(q.quantizedLayers.count == 1)
        #expect(q.quantizedLayers[0].pq != nil)
    }

    @Test("Fallback for small layer")
    func fallbackForSmallLayer() throws {
        let vectors = makeVectors(count: 300, dim: 128)
        let skip = makeSkipLayer(nodeIDs: Array(0..<128))
        let hnsw = HNSWLayers(layers: [skip], maxLayer: 1, entryPoint: 0)

        let q = try QuantizedHNSWBuilder.build(
            from: hnsw,
            vectors: vectors,
            config: .default,
            metric: .cosine
        )

        #expect(q.quantizedLayers[0].pq == nil)
        #expect(q.quantizedLayers[0].codes.isEmpty)
    }

    @Test("Codes count matches layer nodes when PQ exists")
    func codesCountMatchesLayerNodes() throws {
        let vectors = makeVectors(count: 500, dim: 128)
        let largeSkip = makeSkipLayer(nodeIDs: Array(0..<260))
        let smallSkip = makeSkipLayer(nodeIDs: Array(260..<300))
        let hnsw = HNSWLayers(layers: [largeSkip, smallSkip], maxLayer: 2, entryPoint: 0)

        let q = try QuantizedHNSWBuilder.build(
            from: hnsw,
            vectors: vectors,
            config: .default,
            metric: .cosine
        )

        for layer in q.quantizedLayers {
            if layer.pq != nil {
                #expect(layer.codes.count == layer.base.layerIndexToNode.count)
            } else {
                #expect(layer.codes.isEmpty)
            }
        }
    }

    @Test("Dimension mismatch adjusted")
    func dimensionMismatchAdjusted() throws {
        let vectors = makeVectors(count: 300, dim: 130)
        let skip = makeSkipLayer(nodeIDs: Array(0..<260))
        let hnsw = HNSWLayers(layers: [skip], maxLayer: 1, entryPoint: 0)

        let q = try QuantizedHNSWBuilder.build(
            from: hnsw,
            vectors: vectors,
            config: QuantizedHNSWConfiguration(pqSubspaces: 8),
            metric: .cosine
        )

        #expect(q.quantizedLayers[0].pq?.numSubspaces == 5)
    }

    @Test("Rejects malformed skip-layer mappings")
    func rejectsMalformedSkipLayerMappings() {
        let vectors = makeVectors(count: 300, dim: 128)
        let malformed = SkipLayer(
            nodeToLayerIndex: [0: 9],
            layerIndexToNode: [0],
            adjacency: [[0]]
        )
        let hnsw = HNSWLayers(layers: [malformed], maxLayer: 1, entryPoint: 0)

        do {
            _ = try QuantizedHNSWBuilder.build(
                from: hnsw,
                vectors: vectors,
                config: .default,
                metric: .cosine
            )
            #expect(Bool(false), "Expected malformed skip-layer mapping to throw")
        } catch let error as ANNSError {
            if case .constructionFailed = error { }
            else {
                #expect(Bool(false), "Expected constructionFailed, got \(error)")
            }
        } catch {
            #expect(Bool(false), "Expected ANNSError.constructionFailed, got \(error)")
        }
    }

    private func makeVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                Float((row + col) % 64) / 64.0
            }
        }
    }

    private func makeSkipLayer(nodeIDs: [UInt32]) -> SkipLayer {
        var nodeToLayerIndex: [UInt32: UInt32] = [:]
        var layerIndexToNode: [UInt32] = []
        var adjacency: [[UInt32]] = []
        layerIndexToNode.reserveCapacity(nodeIDs.count)
        adjacency.reserveCapacity(nodeIDs.count)

        for (index, nodeID) in nodeIDs.enumerated() {
            let local = UInt32(index)
            nodeToLayerIndex[nodeID] = local
            layerIndexToNode.append(nodeID)
        }

        for i in 0..<nodeIDs.count {
            let current = nodeIDs[i]
            let next = nodeIDs[(i + 1) % nodeIDs.count]
            if current == next {
                adjacency.append([])
            } else {
                adjacency.append([next])
            }
        }

        return SkipLayer(
            nodeToLayerIndex: nodeToLayerIndex,
            layerIndexToNode: layerIndexToNode,
            adjacency: adjacency
        )
    }
}
