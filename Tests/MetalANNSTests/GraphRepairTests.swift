import Testing
@testable import MetalANNSCore

@Suite("Graph Repair Tests")
struct GraphRepairTests {
    @Test("RepairConfiguration defaults are sensible")
    func repairConfigDefaults() {
        let config = RepairConfiguration.default
        #expect(config.repairInterval == 100)
        #expect(config.repairDepth == 2)
        #expect(config.repairIterations == 5)
        #expect(config.enabled == true)
    }

    @Test("RepairConfiguration clamps invalid values")
    func repairConfigClamping() {
        let config = RepairConfiguration(repairInterval: -5, repairDepth: 0, repairIterations: 100)
        #expect(config.repairInterval == 0)
        #expect(config.repairDepth == 1)
        #expect(config.repairIterations == 20)
    }
}
