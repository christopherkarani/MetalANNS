import Foundation
import Testing
@testable import MetalANNS
@testable import MetalANNSCore

@Suite("QuantizedHNSW Persistence Tests")
struct QuantizedHNSWPersistenceTests {
    @Test("Save and load quantized index")
    func saveAndLoadQuantizedIndex() async throws {
        let vectors = makeVectors(count: 600, dim: 128)
        let ids = (0..<vectors.count).map { "id-\($0)" }
        let index = ANNSIndex(configuration: .default, context: nil)
        try await index.build(vectors: vectors, ids: ids)

        let url = tempIndexURL()
        defer { cleanup(url) }

        try await index.save(to: url)
        let loaded = try await ANNSIndex.load(from: url)

        for query in vectors.prefix(10) {
            let results = try await loaded.search(query: query, k: 10)
            #expect(!results.isEmpty)
        }
    }

    @Test("Quantized layers survive round trip")
    func quantizedLayersSurviveRoundTrip() async throws {
        let vectors = makeVectors(count: 600, dim: 128)
        let ids = (0..<vectors.count).map { "id-\($0)" }
        let index = ANNSIndex(configuration: .default, context: nil)
        try await index.build(vectors: vectors, ids: ids)

        let url = tempIndexURL()
        defer { cleanup(url) }

        try await index.save(to: url)
        let loaded = try await ANNSIndex.load(from: url)

        #expect(await loaded.hasQuantizedHNSWForTesting())
    }

    @Test("Backward compatible metadata")
    func backwardCompatible() async throws {
        let vectors = makeVectors(count: 300, dim: 128)
        let ids = (0..<vectors.count).map { "id-\($0)" }
        let index = ANNSIndex(configuration: .default, context: nil)
        try await index.build(vectors: vectors, ids: ids)

        let url = tempIndexURL()
        defer { cleanup(url) }

        try await index.save(to: url)

        let metaURL = URL(fileURLWithPath: url.path + ".meta.json")
        let data = try Data(contentsOf: metaURL)
        var json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
        if var config = json?["configuration"] as? [String: Any] {
            config.removeValue(forKey: "quantizedHNSWConfiguration")
            json?["configuration"] = config
        }
        let updated = try JSONSerialization.data(withJSONObject: json ?? [:], options: [.prettyPrinted])
        try updated.write(to: metaURL, options: .atomic)

        let loaded = try await ANNSIndex.load(from: url)
        let loadedConfig = await loaded.configurationForTesting()
        #expect(loadedConfig.quantizedHNSWConfiguration.useQuantizedEdges)
    }

    private func tempIndexURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("metalanns-quantized-persistence-\(UUID().uuidString)")
            .appendingPathExtension("mann")
    }

    private func cleanup(_ url: URL) {
        let sidecars = [
            url,
            URL(fileURLWithPath: url.path + ".meta.json"),
            URL(fileURLWithPath: url.path + ".qhnsw.json")
        ]
        for file in sidecars {
            try? FileManager.default.removeItem(at: file)
        }
    }

    private func makeVectors(count: Int, dim: Int) -> [[Float]] {
        (0..<count).map { row in
            (0..<dim).map { col in
                let x = Float(row * dim + col)
                return sin(x * 0.017) + cos(x * 0.031)
            }
        }
    }
}
