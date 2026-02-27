import Foundation
import Testing
@testable import MetalANNSCore

@Suite("QuantizedHNSWConfiguration Tests")
struct QuantizedHNSWConfigurationTests {
    @Test("Default values")
    func defaultValues() {
        let config = QuantizedHNSWConfiguration()
        #expect(config.useQuantizedEdges)
        #expect(config.pqSubspaces == 4)
        #expect(config.base.enabled == HNSWConfiguration.default.enabled)
        #expect(config.base.M == HNSWConfiguration.default.M)
        #expect(config.base.maxLayers == HNSWConfiguration.default.maxLayers)
    }

    @Test("Disabled by flag")
    func disabledByFlag() {
        let config = QuantizedHNSWConfiguration(useQuantizedEdges: false)
        #expect(config.useQuantizedEdges == false)
    }

    @Test("Codable round trip")
    func codableRoundTrip() throws {
        let config = QuantizedHNSWConfiguration(
            base: HNSWConfiguration(enabled: true, M: 12, maxLayers: 4),
            useQuantizedEdges: false,
            pqSubspaces: 6
        )

        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(QuantizedHNSWConfiguration.self, from: encoded)
        #expect(decoded == config)
    }
}
