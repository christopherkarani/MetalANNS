import Metal
import Testing
@testable import MetalANNSCore

@Suite("MetalDevice Tests")
struct MetalDeviceTests {
    @Test("MetalContext initializes on device with GPU")
    func initContext() throws {
        #if targetEnvironment(simulator)
        return
        #else
        let context = try MetalContext()
        #expect(context.device.name.isEmpty == false)
        #endif
    }

    @Test("PipelineCache compiles a function from the shader library")
    func pipelineCacheCompile() async throws {
        #if targetEnvironment(simulator)
        return
        #else
        let context = try MetalContext()
        let pipeline = try await context.pipelineCache.pipeline(for: "cosine_distance")
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
        #endif
    }
}
