import Foundation
import Testing
import Metal
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("HNSWInserter Unit Tests")
struct HNSWInserterTests {
    @Test("Level assignment")
    func levelAssignment() {
        let maxLayers = 6
        let trials = 1_000
        var seenPositive = false

        for _ in 0..<trials {
            let level = HNSWBuilder.assignLevel(
                mL: 1.4426950408889634,
                maxLayers: maxLayers
            )
            #expect((0...maxLayers).contains(level))
            if level > 0 {
                seenPositive = true
            }
        }

        #expect(seenPositive)
    }

    @Test("Insert single node into existing layers")
    func insertSingleNodeIntoExistingLayers() throws {
        var seed = SeededGenerator(state: 0x1234_5678)
        let baseVectors = randomVectors(count: 5, dim: 16, seed: &seed)
        let insertVector = randomVectors(count: 1, dim: 16, seed: &seed)[0]

        let allVectors = baseVectors + [insertVector]
        let storage = try TestVectorStorage(vectors: allVectors)

        let graph: [[(UInt32, Float)]] = Array(repeating: [], count: 5)
        let config = HNSWConfiguration(enabled: true, M: 4, maxLayers: 6)
        var attempts = 0
        var attemptLayers: HNSWLayers?

        while attempts < 2000 {
            var built = try HNSWBuilder.buildLayers(
                vectors: storage,
                graph: graph,
                nodeCount: 5,
                metric: .l2,
                config: HNSWConfiguration(enabled: true, M: 4, maxLayers: 2)
            )

            try HNSWInserter.insert(
                vector: insertVector,
                nodeID: 5,
                into: &built,
                vectorStorage: storage,
                config: config,
                metric: .l2
            )

            let levelsWithNode = (1...built.maxLayer).filter { level in
                built.layers[level - 1].nodeToLayerIndex[5] != nil
            }

            if !levelsWithNode.isEmpty {
                attemptLayers = built
                break
            }
            attempts += 1
        }

        guard let resultLayers = attemptLayers else {
            throw ANNSError.constructionFailed("Unable to insert into any higher layer in 2000 attempts")
        }
        let nodeLevels = (1...resultLayers.maxLayer).filter {
            resultLayers.layers[$0 - 1].nodeToLayerIndex[5] != nil
        }
        let nodeLevel = nodeLevels.max() ?? 0

        #expect(!nodeLevels.isEmpty)
        for level in 1...nodeLevel {
            #expect(resultLayers.layers[level - 1].nodeToLayerIndex[5] != nil)

            let newNodeLocalIndex = Int(resultLayers.layers[level - 1].nodeToLayerIndex[5]!)
            let neighbors = resultLayers.layers[level - 1].adjacency[newNodeLocalIndex]
            for neighbor in neighbors {
                guard let neighborLocalIndex = resultLayers.layers[level - 1].nodeToLayerIndex[neighbor] else {
                    Issue.record("Missing neighbor mapping at level \(level): \(neighbor)")
                    continue
                }
                let reverseAdj = resultLayers.layers[level - 1].adjacency[Int(neighborLocalIndex)]
                #expect(reverseAdj.contains(5))
            }
        }

        _ = seed
        #expect(nodeLevel > 0)
    }

    @Test("Entry point updated")
    func entryPointUpdated() throws {
        var seed = SeededGenerator(state: 0x2A2A_2A2A)
        let baseVectors = randomVectors(count: 40, dim: 16, seed: &seed)
        let insertVector = randomVectors(count: 1, dim: 16, seed: &seed)[0]

        let storage = try TestVectorStorage(vectors: baseVectors + [insertVector])
        let graph: [[(UInt32, Float)]] = Array(repeating: [], count: baseVectors.count)

        let baseConfig = HNSWConfiguration(enabled: true, M: 4, maxLayers: 1)
        var promoted: (layers: HNSWLayers, nodeID: UInt32)?

        for _ in 0..<2000 {
            var candidate = try HNSWBuilder.buildLayers(
                vectors: storage,
                graph: graph,
                nodeCount: baseVectors.count,
                metric: .l2,
                config: baseConfig
            )

            let targetID: UInt32 = 40
            try HNSWInserter.insert(
                vector: insertVector,
                nodeID: targetID,
                into: &candidate,
                vectorStorage: storage,
                config: HNSWConfiguration(enabled: true, M: 4, maxLayers: 6),
                metric: .l2
            )

            let nodeLevels = (1...candidate.maxLayer).filter { level in
                candidate.layers[level - 1].nodeToLayerIndex[targetID] != nil
            }
            if nodeLevels.max() == 2 {
                promoted = (candidate, targetID)
                break
            }
        }

        guard let result = promoted else {
            throw ANNSError.constructionFailed("Failed to observe nodeLevel == 2 in 2000 attempts")
        }

        #expect(result.layers.maxLayer == 2)
        #expect(result.layers.entryPoint == result.nodeID)
        #expect(result.layers.layers[1].nodeToLayerIndex[result.nodeID] != nil)
    }

    @Test("Reject non-append node ID")
    func rejectNonAppendNodeID() throws {
        var seed = SeededGenerator(state: 0xABCDEF01)
        let vectors = randomVectors(count: 7, dim: 8, seed: &seed)
        let storage = try TestVectorStorage(vectors: vectors)
        let graph: [[(UInt32, Float)]] = Array(repeating: [], count: 5)

        var layers = try HNSWBuilder.buildLayers(
            vectors: storage,
            graph: graph,
            nodeCount: 5,
            metric: .l2,
            config: HNSWConfiguration(enabled: true, M: 4, maxLayers: 6)
        )

        #expect(throws: ANNSError.self) {
            try HNSWInserter.insert(
                vector: vectors[5],
                nodeID: 5,
                into: &layers,
                vectorStorage: storage,
                config: HNSWConfiguration(enabled: true, M: 4, maxLayers: 6),
                metric: .l2
            )
        }
    }

    @Test("Insert and search recall")
    func insertAndSearchRecall() async throws {
        var seed = SeededGenerator(state: 0x1111_1111)

        let buildCount = 200
        let insertCount = 50
        let dim = 32

        var quantizedConfig = QuantizedHNSWConfiguration.default
        quantizedConfig.useQuantizedEdges = false
        let config = IndexConfiguration(
            metric: .l2,
            hnswConfiguration: HNSWConfiguration(enabled: true, M: 8, maxLayers: 6),
            quantizedHNSWConfiguration: quantizedConfig
        )
        let index = ANNSIndex(configuration: config, context: nil)

        let baseVectors = randomVectors(count: buildCount, dim: dim, seed: &seed)
        let insertVectors = randomVectors(count: insertCount, dim: dim, seed: &seed)
        let baseIDs = (0..<buildCount).map { "base_\($0)" }
        let insertIDs = (0..<insertCount).map { "insert_\($0)" }

        try await index.build(vectors: baseVectors, ids: baseIDs)
        for (offset, vector) in insertVectors.enumerated() {
            try await index.insert(vector, id: insertIDs[offset])
        }

        #expect(await index.isHNSWBuilt)

        let corpus = baseVectors + insertVectors
        let corpusIDs = baseIDs + insertIDs

        var recallTotal = 0.0
        for query in corpus.prefix(20) {
            let results = try await index.search(query: query, k: 10, metric: .l2)
            let expected = bruteForceTopK(query: query, vectors: corpus, k: 10, metric: .l2)
            recallTotal += recall(results: results, groundTruth: expected, ids: corpusIDs, k: 10)
        }

        #expect(recallTotal / 20.0 > 0.75)
    }

    @Test("Recall vs batch build")
    func recallVsBatchBuild() async throws {
        var seed = SeededGenerator(state: 0x7777_7777)

        let baseCount = 150
        let insertCount = 50
        let dim = 32

        let allVectors = randomVectors(count: baseCount + insertCount, dim: dim, seed: &seed)
        let baseVectors = Array(allVectors.prefix(baseCount))
        let insertVectors = Array(allVectors.suffix(insertCount))

        let baseIDs = (0..<baseCount).map { "base_\($0)" }
        let insertIDs = (0..<insertCount).map { "insert_\($0)" }
        let allIDs = baseIDs + insertIDs

        let config = IndexConfiguration(
            metric: .l2,
            hnswConfiguration: HNSWConfiguration(enabled: true, M: 8, maxLayers: 6)
        )

        let batchIndex = ANNSIndex(configuration: config, context: nil)
        let sequentialIndex = ANNSIndex(configuration: config, context: nil)

        try await batchIndex.build(vectors: allVectors, ids: allIDs)

        try await sequentialIndex.build(vectors: baseVectors, ids: baseIDs)
        for (offset, vector) in insertVectors.enumerated() {
            try await sequentialIndex.insert(vector, id: insertIDs[offset])
        }

        #expect(await sequentialIndex.isHNSWBuilt)

        let queries = Array(allVectors.prefix(20))

        let batchRecall = try await recallFrom(
            index: batchIndex,
            queries: queries,
            vectors: allVectors,
            ids: allIDs
        )
        let incrementalRecall = try await recallFrom(
            index: sequentialIndex,
            queries: queries,
            vectors: allVectors,
            ids: allIDs
        )

        #expect(incrementalRecall >= batchRecall - 0.10)
    }

    private func recallFrom(
        index: ANNSIndex,
        queries: [[Float]],
        vectors: [[Float]],
        ids: [String],
        k: Int = 10
    ) async throws -> Double {
        var sum = 0.0
        for query in queries {
            let results = try await index.search(query: query, k: k, metric: .l2)
            let expected = bruteForceTopK(query: query, vectors: vectors, k: k, metric: .l2)
            sum += recall(results: results, groundTruth: expected, ids: ids, k: k)
        }
        return sum / Double(queries.count)
    }

    private func randomVectors(count: Int, dim: Int, seed: inout SeededGenerator) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dim).map { _ in
                let u1 = max(seed.nextFloat(), .leastNonzeroMagnitude)
                let u2 = max(seed.nextFloat(), .leastNonzeroMagnitude)
                let z = sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
                return z
            }
        }
    }

    private func bruteForceTopK(query: [Float], vectors: [[Float]], k: Int, metric: Metric) -> [Int] {
        let scored = vectors.enumerated().map { (index, vector) in
            (index, SIMDDistance.distance(query, vector, metric: metric))
        }
        return scored
            .sorted { $0.1 < $1.1 }
            .prefix(min(k, scored.count))
            .map(\.0)
    }

    private func recall(results: [SearchResult], groundTruth: [Int], ids: [String], k: Int) -> Double {
        let expectedIDs = Set<String>(groundTruth.prefix(k).compactMap { index in
            guard ids.indices.contains(index) else { return nil }
            return ids[index]
        })
        let found = Set(results.prefix(k).map(\.id))
        let overlap = found.intersection(expectedIDs).count
        return Double(overlap) / Double(max(1, k))
    }

    private final class TestVectorStorage: VectorStorage, @unchecked Sendable {
        private var values: [[Float]]
        private var storedCount: Int

        let buffer: MTLBuffer
        let dim: Int
        let capacity: Int
        let isFloat16 = false

        var count: Int { storedCount }

        init(vectors: [[Float]]) throws {
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw ANNSError.constructionFailed("No Metal device available")
            }
            guard let first = vectors.first else {
                throw ANNSError.constructionFailed("VectorStorage requires at least one vector")
            }

            let dim = first.count
            let bytes = max(1, vectors.count * dim * 4)
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw ANNSError.constructionFailed("Unable to allocate storage buffer")
            }

            self.values = vectors
            self.storedCount = vectors.count
            self.buffer = buffer
            self.dim = dim
            self.capacity = vectors.count
        }

        func setCount(_ newCount: Int) {
            storedCount = max(0, min(newCount, capacity))
        }

        func insert(vector: [Float], at index: Int) throws {
            guard vector.count == dim else {
                throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
            }
            guard index >= 0, index < capacity else {
                throw ANNSError.constructionFailed("Index \(index) is out of bounds")
            }

            if index >= values.count {
                values.append(vector)
            } else {
                values[index] = vector
            }
        }

        func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
            for (offset, vector) in vectors.enumerated() {
                try insert(vector: vector, at: start + offset)
            }
        }

        func vector(at index: Int) -> [Float] {
            values[index]
        }
    }

    private struct SeededGenerator {
        var state: UInt64

        init(state: UInt64) {
            self.state = state
        }

        mutating func nextFloat() -> Float {
            state = 2862933555777941757 &* state &+ 3037000493
            let value = state >> 11
            return Float(value) / Float(UInt64.max >> 11)
        }
    }
}
