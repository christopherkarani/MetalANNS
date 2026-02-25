import Testing
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("ComputeBackend Protocol Tests")
struct BackendProtocolTests {
    @Test("BackendFactory creates a backend without crashing")
    func backendCreation() async throws {
        let backend = try BackendFactory.makeBackend()
        #expect(backend != nil)
    }
}
