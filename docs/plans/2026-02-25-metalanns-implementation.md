# MetalANNS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a GPU-native Approximate Nearest Neighbor Search Swift package for Apple Silicon using Metal compute shaders with a CAGRA-style NN-Descent graph.

**Architecture:** Three-target Swift package (MetalANNSCore, MetalANNS, MetalANNSBenchmarks). Dual backend: MetalBackend for GPU, AccelerateBackend for CPU/simulator. NN-Descent builds a fixed out-degree directed graph, beam search queries it. All GPU work hidden behind async actor API.

**Tech Stack:** Swift 6, Metal Shading Language, Accelerate (vDSP/BLAS), Swift Testing, OSLog. Zero external dependencies.

**Source Spec:** `/Users/chriskarani/Desktop/MetalANNS_Implementation_Plan.md`
**Design Doc:** `docs/plans/2026-02-25-metalanns-design.md`

---

## Critical Research Findings (Read Before Implementing)

These correct or extend the original spec:

1. **SPM + Metal**: `.process("Shaders")` in Package.swift. Load with `try device.makeDefaultLibrary(bundle: Bundle.module)`. **`swift build` CLI does NOT compile .metal files** — always use `xcodebuild` for builds and tests.
2. **Metal atomics**: Only `memory_order_relaxed` is supported. All atomics are 32-bit only (no 64-bit on M1, only min/max on M2+). Use separate `atomic_uint` buffers for IDs and distances.
3. **CAS pattern**: Use `as_type<uint>(float)` to reinterpret float bits as uint for atomic comparison. IEEE 754 preserves ordering for non-negative floats.
4. **NN-Descent local join**: The correct algorithm uses new/old distinction — skip old-old pairs. For GPU simplicity in v1, we can process all forward x reverse pairs and rely on cheap GPU iterations.
5. **Convergence**: Standard threshold is `delta = 0.001` (stop when < 0.1% of total edges change). Use threadgroup-local reduction + global `atomic_fetch_add`.

---

## Phase 1: Foundation — Metal Pipeline, Buffers, Distance Kernels

### Task 1: Package Scaffold

**Files:**
- Create: `Package.swift`
- Create: `Sources/MetalANNSCore/Shaders/Distance.metal` (placeholder)
- Create: `Sources/MetalANNSCore/MetalDevice.swift` (placeholder)
- Create: `Sources/MetalANNS/ANNSIndex.swift` (placeholder)
- Create: `Tests/MetalANNSTests/DistanceTests.swift` (placeholder)
- Create: `.gitignore`

**Step 1: Create Package.swift**

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MetalANNS",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "MetalANNS", targets: ["MetalANNS"])
    ],
    targets: [
        .target(
            name: "MetalANNSCore",
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "MetalANNS",
            dependencies: ["MetalANNSCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MetalANNSTests",
            dependencies: ["MetalANNS", "MetalANNSCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "MetalANNSBenchmarks",
            dependencies: ["MetalANNS", "MetalANNSCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
```

**Step 2: Create placeholder files**

Create minimal placeholder files so the package compiles:
- `Sources/MetalANNSCore/Shaders/Distance.metal` — empty metal file with `#include <metal_stdlib>`
- `Sources/MetalANNSCore/MetalDevice.swift` — `import Metal`
- `Sources/MetalANNS/ANNSIndex.swift` — `import MetalANNSCore`
- `Tests/MetalANNSTests/DistanceTests.swift` — `import Testing`
- `Sources/MetalANNSBenchmarks/main.swift` — `print("MetalANNS Benchmarks")`

**Step 3: Create .gitignore**

```
.DS_Store
*.xcuserstate
.build/
DerivedData/
*.xcodeproj/xcuserdata/
*.xcworkspace/xcuserdata/
```

**Step 4: Verify package resolves**

Run: `cd /Users/chriskarani/CodingProjects/MetalANNS && xcodebuild -scheme MetalANNS -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 5: Initialize git and commit**

```bash
cd /Users/chriskarani/CodingProjects/MetalANNS
git init
git add -A
git commit -m "chore: initialize MetalANNS Swift package scaffold"
```

---

### Task 2: Error Types and Metric Enum

**Files:**
- Create: `Sources/MetalANNS/Errors.swift`
- Create: `Sources/MetalANNS/IndexConfiguration.swift`
- Test: `Tests/MetalANNSTests/ConfigurationTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/ConfigurationTests.swift
import Testing
@testable import MetalANNS

@Suite("IndexConfiguration Tests")
struct ConfigurationTests {
    @Test("Default configuration has expected values")
    func defaultConfiguration() {
        let config = IndexConfiguration.default
        #expect(config.degree == 32)
        #expect(config.metric == .cosine)
        #expect(config.efConstruction == 100)
        #expect(config.efSearch == 64)
        #expect(config.maxIterations == 20)
        #expect(config.useFloat16 == false)
    }

    @Test("Metric enum has all three cases")
    func metricCases() {
        let metrics: [Metric] = [.cosine, .l2, .innerProduct]
        #expect(metrics.count == 3)
    }

    @Test("ANNSError cases are distinct")
    func errorCases() {
        let error1 = ANNSError.deviceNotSupported
        let error2 = ANNSError.dimensionMismatch(expected: 128, got: 64)
        // Verify they are Error-conforming
        #expect(error1.localizedDescription.isEmpty == false)
        _ = error2  // Compile check
    }
}
```

**Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/ConfigurationTests 2>&1 | tail -10`
Expected: FAIL — types not defined

**Step 3: Implement Errors.swift**

```swift
// Sources/MetalANNS/Errors.swift
import Foundation

public enum ANNSError: Error, Sendable {
    case deviceNotSupported
    case dimensionMismatch(expected: Int, got: Int)
    case idAlreadyExists(String)
    case idNotFound(String)
    case corruptFile(String)
    case constructionFailed(String)
    case searchFailed(String)
    case indexEmpty
}
```

**Step 4: Implement IndexConfiguration.swift**

```swift
// Sources/MetalANNS/IndexConfiguration.swift

public enum Metric: String, Sendable, Codable {
    case cosine
    case l2
    case innerProduct
}

public struct IndexConfiguration: Sendable {
    public var degree: Int
    public var metric: Metric
    public var efConstruction: Int
    public var efSearch: Int
    public var maxIterations: Int
    public var useFloat16: Bool
    public var convergenceThreshold: Float

    public static let `default` = IndexConfiguration(
        degree: 32,
        metric: .cosine,
        efConstruction: 100,
        efSearch: 64,
        maxIterations: 20,
        useFloat16: false,
        convergenceThreshold: 0.001
    )

    public init(
        degree: Int = 32,
        metric: Metric = .cosine,
        efConstruction: Int = 100,
        efSearch: Int = 64,
        maxIterations: Int = 20,
        useFloat16: Bool = false,
        convergenceThreshold: Float = 0.001
    ) {
        self.degree = degree
        self.metric = metric
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.maxIterations = maxIterations
        self.useFloat16 = useFloat16
        self.convergenceThreshold = convergenceThreshold
    }
}
```

**Step 5: Run test to verify it passes**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/ConfigurationTests 2>&1 | tail -10`
Expected: PASS

**Step 6: Commit**

```bash
git add Sources/MetalANNS/Errors.swift Sources/MetalANNS/IndexConfiguration.swift Tests/MetalANNSTests/ConfigurationTests.swift
git commit -m "feat: add ANNSError, Metric, and IndexConfiguration types"
```

---

### Task 3: Compute Backend Protocol

**Files:**
- Create: `Sources/MetalANNSCore/ComputeBackend.swift`
- Test: `Tests/MetalANNSTests/BackendProtocolTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/BackendProtocolTests.swift
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
```

**Step 2: Run test to verify it fails**

Expected: FAIL — protocol and factory not defined

**Step 3: Implement ComputeBackend protocol**

```swift
// Sources/MetalANNSCore/ComputeBackend.swift
import Foundation

/// Protocol abstracting GPU (Metal) and CPU (Accelerate) compute backends.
/// Enables testing on simulator and CI without a GPU.
public protocol ComputeBackend: Sendable {
    /// Compute distances from a query vector to all vectors in a corpus.
    func computeDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric
    ) async throws -> [Float]
}

/// Factory that selects Metal or Accelerate backend based on device availability.
public enum BackendFactory {
    public static func makeBackend() throws -> any ComputeBackend {
        #if targetEnvironment(simulator)
        return AccelerateBackend()
        #else
        if let metalBackend = try? MetalBackend() {
            return metalBackend
        }
        return AccelerateBackend()
        #endif
    }
}
```

Note: `MetalBackend` and `AccelerateBackend` are stubs at this point. We'll implement them in Tasks 4–5.

**Step 4: Create stub backends**

```swift
// Sources/MetalANNSCore/AccelerateBackend.swift
import Accelerate

public struct AccelerateBackend: ComputeBackend, Sendable {
    public init() {}

    public func computeDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric
    ) async throws -> [Float] {
        fatalError("Not yet implemented")
    }
}
```

```swift
// Sources/MetalANNSCore/MetalBackend.swift
import Metal

public final class MetalBackend: ComputeBackend, @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ANNSError.deviceNotSupported
        }
        guard let queue = device.makeCommandQueue() else {
            throw ANNSError.deviceNotSupported
        }
        self.device = device
        self.commandQueue = queue
    }

    public func computeDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric
    ) async throws -> [Float] {
        fatalError("Not yet implemented")
    }
}
```

**Step 5: Run test to verify it passes**

Expected: PASS (factory returns a backend)

**Step 6: Commit**

```bash
git add Sources/MetalANNSCore/ComputeBackend.swift Sources/MetalANNSCore/AccelerateBackend.swift Sources/MetalANNSCore/MetalBackend.swift Tests/MetalANNSTests/BackendProtocolTests.swift
git commit -m "feat: add ComputeBackend protocol with factory and stub backends"
```

---

### Task 4: Accelerate Distance Kernels (CPU Reference)

**Files:**
- Modify: `Sources/MetalANNSCore/AccelerateBackend.swift`
- Test: `Tests/MetalANNSTests/DistanceTests.swift`

**Step 1: Write the failing tests**

```swift
// Tests/MetalANNSTests/DistanceTests.swift
import Testing
import Accelerate
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("Distance Computation Tests")
struct DistanceTests {
    let backend = AccelerateBackend()

    @Test("Cosine distance of identical vectors is 0")
    func cosineIdentical() async throws {
        let v = [Float](repeating: 1.0, count: 128)
        let distances = try await v.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: v, vectors: ptr, vectorCount: 1, dim: 128, metric: .cosine
            )
        }
        #expect(distances.count == 1)
        #expect(abs(distances[0]) < 1e-5)
    }

    @Test("Cosine distance of orthogonal vectors is 1")
    func cosineOrthogonal() async throws {
        var v1 = [Float](repeating: 0, count: 4)
        v1[0] = 1.0
        var v2 = [Float](repeating: 0, count: 4)
        v2[1] = 1.0
        let distances = try await v2.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: v1, vectors: ptr, vectorCount: 1, dim: 4, metric: .cosine
            )
        }
        #expect(abs(distances[0] - 1.0) < 1e-5)
    }

    @Test("L2 distance of identical vectors is 0")
    func l2Identical() async throws {
        let v = [Float](repeating: 1.0, count: 128)
        let distances = try await v.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: v, vectors: ptr, vectorCount: 1, dim: 128, metric: .l2
            )
        }
        #expect(abs(distances[0]) < 1e-5)
    }

    @Test("L2 distance is squared Euclidean")
    func l2Squared() async throws {
        let q: [Float] = [1, 0, 0]
        let v: [Float] = [0, 1, 0]
        // squared distance = (1-0)^2 + (0-1)^2 + (0-0)^2 = 2
        let distances = try await v.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: q, vectors: ptr, vectorCount: 1, dim: 3, metric: .l2
            )
        }
        #expect(abs(distances[0] - 2.0) < 1e-5)
    }

    @Test("Inner product of unit vectors")
    func innerProduct() async throws {
        let q: [Float] = [1, 0, 0]
        let v: [Float] = [0.5, 0.5, 0]
        // dot product = 0.5
        let distances = try await v.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: q, vectors: ptr, vectorCount: 1, dim: 3, metric: .innerProduct
            )
        }
        // inner product distance = negative dot (lower = more similar)
        #expect(abs(distances[0] - (-0.5)) < 1e-5)
    }

    @Test("Batch distances: 1000 random 128-dim vectors")
    func batchDistances() async throws {
        let dim = 128
        let n = 1000
        var vectors = [Float](repeating: 0, count: n * dim)
        for i in 0..<vectors.count { vectors[i] = Float.random(in: -1...1) }
        var query = [Float](repeating: 0, count: dim)
        for i in 0..<dim { query[i] = Float.random(in: -1...1) }

        let distances = try await vectors.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: query, vectors: ptr, vectorCount: n, dim: dim, metric: .cosine
            )
        }
        #expect(distances.count == n)
        // Cosine distance in [0, 2]
        for d in distances {
            #expect(d >= -1e-5 && d <= 2.0 + 1e-5)
        }
    }

    @Test("Edge case: dim=1")
    func dim1() async throws {
        let q: [Float] = [3.0]
        let v: [Float] = [4.0]
        let distances = try await v.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: q, vectors: ptr, vectorCount: 1, dim: 1, metric: .l2
            )
        }
        #expect(abs(distances[0] - 1.0) < 1e-5) // (3-4)^2 = 1
    }

    @Test("Edge case: dim=1536 (large embedding)")
    func dimLarge() async throws {
        let dim = 1536
        let v = [Float](repeating: 1.0 / sqrt(Float(dim)), count: dim)
        let distances = try await v.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: v, vectors: ptr, vectorCount: 1, dim: dim, metric: .cosine
            )
        }
        #expect(abs(distances[0]) < 1e-4) // identical unit vector → distance ≈ 0
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/DistanceTests 2>&1 | tail -15`
Expected: FAIL — fatalError in AccelerateBackend

**Step 3: Implement AccelerateBackend distance computation**

```swift
// Sources/MetalANNSCore/AccelerateBackend.swift
import Accelerate
import Foundation

public struct AccelerateBackend: ComputeBackend, Sendable {
    public init() {}

    public func computeDistances(
        query: [Float],
        vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int,
        dim: Int,
        metric: Metric
    ) async throws -> [Float] {
        var results = [Float](repeating: 0, count: vectorCount)

        switch metric {
        case .cosine:
            computeCosineDistances(query: query, vectors: vectors, vectorCount: vectorCount, dim: dim, results: &results)
        case .l2:
            computeL2Distances(query: query, vectors: vectors, vectorCount: vectorCount, dim: dim, results: &results)
        case .innerProduct:
            computeInnerProductDistances(query: query, vectors: vectors, vectorCount: vectorCount, dim: dim, results: &results)
        }

        return results
    }

    private func computeCosineDistances(
        query: [Float], vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int, dim: Int, results: inout [Float]
    ) {
        // Compute query norm
        var queryNormSq: Float = 0
        vDSP_dotpr(query, 1, query, 1, &queryNormSq, vDSP_Length(dim))
        let queryNorm = sqrt(queryNormSq)

        for i in 0..<vectorCount {
            let vecStart = i * dim
            let vecSlice = UnsafeBufferPointer(
                start: vectors.baseAddress! + vecStart, count: dim
            )
            // Dot product
            var dot: Float = 0
            vDSP_dotpr(query, 1, vecSlice.baseAddress!, 1, &dot, vDSP_Length(dim))
            // Vec norm
            var vecNormSq: Float = 0
            vDSP_dotpr(vecSlice.baseAddress!, 1, vecSlice.baseAddress!, 1, &vecNormSq, vDSP_Length(dim))
            let vecNorm = sqrt(vecNormSq)
            // Cosine distance = 1 - (dot / (||q|| * ||v||))
            let denom = queryNorm * vecNorm
            if denom < 1e-10 {
                results[i] = 1.0
            } else {
                results[i] = 1.0 - (dot / denom)
            }
        }
    }

    private func computeL2Distances(
        query: [Float], vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int, dim: Int, results: inout [Float]
    ) {
        for i in 0..<vectorCount {
            let vecStart = i * dim
            var sumSq: Float = 0
            for d in 0..<dim {
                let diff = query[d] - vectors[vecStart + d]
                sumSq += diff * diff
            }
            results[i] = sumSq
        }
    }

    private func computeInnerProductDistances(
        query: [Float], vectors: UnsafeBufferPointer<Float>,
        vectorCount: Int, dim: Int, results: inout [Float]
    ) {
        for i in 0..<vectorCount {
            let vecStart = i * dim
            var dot: Float = 0
            vDSP_dotpr(
                query, 1,
                vectors.baseAddress! + vecStart, 1,
                &dot, vDSP_Length(dim)
            )
            results[i] = -dot  // negate so lower = more similar
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/DistanceTests 2>&1 | tail -15`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/AccelerateBackend.swift Tests/MetalANNSTests/DistanceTests.swift
git commit -m "feat: implement Accelerate distance kernels (cosine, L2, inner product)"
```

---

### Task 5: Metal Device & Pipeline Cache

**Files:**
- Create: `Sources/MetalANNSCore/MetalDevice.swift` (full implementation)
- Create: `Sources/MetalANNSCore/PipelineCache.swift`
- Test: `Tests/MetalANNSTests/MetalDeviceTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/MetalDeviceTests.swift
import Testing
import Metal
@testable import MetalANNSCore

@Suite("MetalDevice Tests")
struct MetalDeviceTests {
    @Test("MetalContext initializes on device with GPU")
    func initContext() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Metal not available on simulator")
        #else
        let context = try MetalContext()
        #expect(context.device.name.isEmpty == false)
        #endif
    }

    @Test("PipelineCache compiles a function from the shader library")
    func pipelineCacheCompile() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Metal not available on simulator")
        #else
        let context = try MetalContext()
        let pipeline = try await context.pipelineCache.pipeline(for: "cosine_distance")
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
        #endif
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — MetalContext not defined

**Step 3: Implement MetalContext**

```swift
// Sources/MetalANNSCore/MetalDevice.swift
import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "MetalDevice")

/// Central Metal device and command queue holder.
/// Not an actor because it is created once and its fields are immutable.
public final class MetalContext: Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    public let pipelineCache: PipelineCache

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ANNSError.deviceNotSupported
        }
        guard let queue = device.makeCommandQueue() else {
            throw ANNSError.deviceNotSupported
        }
        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            throw ANNSError.constructionFailed(
                "Failed to load Metal shader library: \(error)"
            )
        }

        self.device = device
        self.commandQueue = queue
        self.library = library
        self.pipelineCache = PipelineCache(device: device, library: library)

        logger.debug("MetalContext initialized: \(device.name)")
    }

    /// Create a command buffer, execute a closure to encode commands, commit and wait.
    public func execute(_ encode: (MTLCommandBuffer) throws -> Void) async throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw ANNSError.constructionFailed("Failed to create command buffer")
        }
        try encode(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw ANNSError.constructionFailed("Command buffer error: \(error)")
        }
    }
}
```

**Step 4: Implement PipelineCache**

```swift
// Sources/MetalANNSCore/PipelineCache.swift
import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "PipelineCache")

/// Thread-safe cache of compiled Metal compute pipeline states.
public actor PipelineCache {
    private let device: MTLDevice
    private let library: MTLLibrary
    private var cache: [String: MTLComputePipelineState] = [:]

    public init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        self.library = library
    }

    /// Get or compile a pipeline state for the given Metal function name.
    public func pipeline(for functionName: String) throws -> MTLComputePipelineState {
        if let cached = cache[functionName] {
            return cached
        }
        guard let function = library.makeFunction(name: functionName) else {
            throw ANNSError.constructionFailed("Metal function '\(functionName)' not found")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        cache[functionName] = pipeline
        logger.debug("Compiled pipeline: \(functionName)")
        return pipeline
    }
}
```

**Step 5: Run test to verify it passes**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/MetalDeviceTests 2>&1 | tail -15`
Expected: PASS (on Mac with GPU) or SKIP (on simulator)

**Step 6: Commit**

```bash
git add Sources/MetalANNSCore/MetalDevice.swift Sources/MetalANNSCore/PipelineCache.swift Tests/MetalANNSTests/MetalDeviceTests.swift
git commit -m "feat: add MetalContext with device lifecycle and PipelineCache"
```

---

### Task 6: Metal Distance Shaders

**Files:**
- Modify: `Sources/MetalANNSCore/Shaders/Distance.metal`
- Test: `Tests/MetalANNSTests/MetalDistanceTests.swift`

**Step 1: Write the failing test (GPU vs CPU comparison)**

```swift
// Tests/MetalANNSTests/MetalDistanceTests.swift
import Testing
import Metal
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("Metal Distance Shader Tests")
struct MetalDistanceTests {
    @Test("GPU cosine matches CPU for 1000 random 128-dim vectors")
    func gpuVsCpuCosine() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Metal not available on simulator")
        #endif
        let dim = 128
        let n = 1000
        var vectors = [Float](repeating: 0, count: n * dim)
        for i in 0..<vectors.count { vectors[i] = Float.random(in: -1...1) }
        var query = [Float](repeating: 0, count: dim)
        for i in 0..<dim { query[i] = Float.random(in: -1...1) }

        let cpuBackend = AccelerateBackend()
        let gpuBackend = try MetalBackend()

        let cpuResults = try await vectors.withUnsafeBufferPointer { ptr in
            try await cpuBackend.computeDistances(
                query: query, vectors: ptr, vectorCount: n, dim: dim, metric: .cosine)
        }
        let gpuResults = try await vectors.withUnsafeBufferPointer { ptr in
            try await gpuBackend.computeDistances(
                query: query, vectors: ptr, vectorCount: n, dim: dim, metric: .cosine)
        }

        #expect(cpuResults.count == gpuResults.count)
        for i in 0..<n {
            #expect(abs(cpuResults[i] - gpuResults[i]) < 1e-4,
                "Mismatch at index \(i): cpu=\(cpuResults[i]) gpu=\(gpuResults[i])")
        }
    }

    @Test("GPU L2 matches CPU for 1000 random 128-dim vectors")
    func gpuVsCpuL2() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Metal not available on simulator")
        #endif
        let dim = 128
        let n = 1000
        var vectors = [Float](repeating: 0, count: n * dim)
        for i in 0..<vectors.count { vectors[i] = Float.random(in: -1...1) }
        var query = [Float](repeating: 0, count: dim)
        for i in 0..<dim { query[i] = Float.random(in: -1...1) }

        let cpuBackend = AccelerateBackend()
        let gpuBackend = try MetalBackend()

        let cpuResults = try await vectors.withUnsafeBufferPointer { ptr in
            try await cpuBackend.computeDistances(
                query: query, vectors: ptr, vectorCount: n, dim: dim, metric: .l2)
        }
        let gpuResults = try await vectors.withUnsafeBufferPointer { ptr in
            try await gpuBackend.computeDistances(
                query: query, vectors: ptr, vectorCount: n, dim: dim, metric: .l2)
        }

        for i in 0..<n {
            #expect(abs(cpuResults[i] - gpuResults[i]) < 1e-3,
                "Mismatch at index \(i): cpu=\(cpuResults[i]) gpu=\(gpuResults[i])")
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Expected: FAIL — MetalBackend.computeDistances calls fatalError

**Step 3: Write Metal distance shaders**

```metal
// Sources/MetalANNSCore/Shaders/Distance.metal
#include <metal_stdlib>
using namespace metal;

/// Cosine distance: 1 - dot(q, v) / (||q|| * ||v||)
/// Each thread computes the distance between the query and one corpus vector.
kernel void cosine_distance(
    device const float *query        [[buffer(0)]],
    device const float *corpus       [[buffer(1)]],
    device float       *output       [[buffer(2)]],
    constant uint      &dim          [[buffer(3)]],
    constant uint      &n            [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float dot_qv = 0.0;
    float norm_q_sq = 0.0;
    float norm_v_sq = 0.0;

    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        float q = query[d];
        float v = corpus[base + d];
        dot_qv += q * v;
        norm_q_sq += q * q;
        norm_v_sq += v * v;
    }

    float denom = sqrt(norm_q_sq) * sqrt(norm_v_sq);
    output[tid] = (denom < 1e-10f) ? 1.0f : (1.0f - dot_qv / denom);
}

/// Squared L2 distance: sum((q[d] - v[d])^2)
kernel void l2_distance(
    device const float *query        [[buffer(0)]],
    device const float *corpus       [[buffer(1)]],
    device float       *output       [[buffer(2)]],
    constant uint      &dim          [[buffer(3)]],
    constant uint      &n            [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float sum_sq = 0.0;
    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        float diff = query[d] - corpus[base + d];
        sum_sq += diff * diff;
    }
    output[tid] = sum_sq;
}

/// Inner product distance: -dot(q, v) (negate so lower = more similar)
kernel void inner_product_distance(
    device const float *query        [[buffer(0)]],
    device const float *corpus       [[buffer(1)]],
    device float       *output       [[buffer(2)]],
    constant uint      &dim          [[buffer(3)]],
    constant uint      &n            [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float dot = 0.0;
    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        dot += query[d] * corpus[base + d];
    }
    output[tid] = -dot;
}
```

**Step 4: Implement MetalBackend.computeDistances**

```swift
// Update Sources/MetalANNSCore/MetalBackend.swift
// Replace the fatalError in computeDistances with actual GPU dispatch:

public func computeDistances(
    query: [Float],
    vectors: UnsafeBufferPointer<Float>,
    vectorCount: Int,
    dim: Int,
    metric: Metric
) async throws -> [Float] {
    let context = try MetalContext()

    let functionName: String
    switch metric {
    case .cosine: functionName = "cosine_distance"
    case .l2: functionName = "l2_distance"
    case .innerProduct: functionName = "inner_product_distance"
    }

    let pipeline = try await context.pipelineCache.pipeline(for: functionName)

    // Create buffers
    let queryBuffer = context.device.makeBuffer(
        bytes: query, length: query.count * MemoryLayout<Float>.stride,
        options: .storageModeShared)!
    let corpusBuffer = context.device.makeBuffer(
        bytes: vectors.baseAddress!, length: vectorCount * dim * MemoryLayout<Float>.stride,
        options: .storageModeShared)!
    let outputBuffer = context.device.makeBuffer(
        length: vectorCount * MemoryLayout<Float>.stride,
        options: .storageModeShared)!

    var dimU: UInt32 = UInt32(dim)
    var nU: UInt32 = UInt32(vectorCount)

    try await context.execute { commandBuffer in
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(corpusBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&dimU, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)

        let threadsPerGrid = MTLSize(width: vectorCount, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(
            width: min(vectorCount, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
    }

    // Read back results
    let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: vectorCount)
    return Array(UnsafeBufferPointer(start: ptr, count: vectorCount))
}
```

**Step 5: Run tests to verify they pass**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/MetalDistanceTests 2>&1 | tail -15`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add Sources/MetalANNSCore/Shaders/Distance.metal Sources/MetalANNSCore/MetalBackend.swift Tests/MetalANNSTests/MetalDistanceTests.swift
git commit -m "feat: implement Metal distance shaders (cosine, L2, inner product) with GPU tests"
```

---

## Phase 2: Graph Data Structures — GPU-Resident Buffers

### Task 7: VectorBuffer

**Files:**
- Create: `Sources/MetalANNSCore/VectorBuffer.swift`
- Test: `Tests/MetalANNSTests/VectorBufferTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/VectorBufferTests.swift
import Testing
import Metal
@testable import MetalANNSCore

@Suite("VectorBuffer Tests")
struct VectorBufferTests {
    @Test("Insert and read back single vector")
    func insertSingle() throws {
        let buf = try VectorBuffer(capacity: 10, dim: 3)
        let v: [Float] = [1.0, 2.0, 3.0]
        try buf.insert(vector: v, at: 0)
        let readBack = buf.vector(at: 0)
        #expect(readBack == v)
    }

    @Test("Batch insert 100 vectors and verify roundtrip")
    func batchInsert() throws {
        let dim = 128
        let n = 100
        let buf = try VectorBuffer(capacity: n, dim: dim)
        var vectors = [[Float]]()
        for _ in 0..<n {
            vectors.append((0..<dim).map { _ in Float.random(in: -1...1) })
        }
        try buf.batchInsert(vectors: vectors, startingAt: 0)
        for i in 0..<n {
            let readBack = buf.vector(at: i)
            for d in 0..<dim {
                #expect(abs(readBack[d] - vectors[i][d]) < 1e-7)
            }
        }
    }

    @Test("Count tracks insertions")
    func countTracking() throws {
        let buf = try VectorBuffer(capacity: 10, dim: 4)
        #expect(buf.count == 0)
        try buf.insert(vector: [1, 2, 3, 4], at: 0)
        buf.setCount(1)
        #expect(buf.count == 1)
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — VectorBuffer not defined

**Step 3: Implement VectorBuffer**

```swift
// Sources/MetalANNSCore/VectorBuffer.swift
import Metal
import Foundation

/// GPU-resident flat buffer storing `count` vectors of `dim` dimensions.
/// Layout: vector[i] starts at offset `i * dim` in the underlying Float32 buffer.
public final class VectorBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let dim: Int
    public let capacity: Int
    public private(set) var count: Int = 0

    private let rawPointer: UnsafeMutablePointer<Float>

    public init(capacity: Int, dim: Int, device: MTLDevice? = nil) throws {
        let dev = device ?? MTLCreateSystemDefaultDevice()!
        let byteLength = capacity * dim * MemoryLayout<Float>.stride
        guard let buf = dev.makeBuffer(length: max(byteLength, 4), options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate VectorBuffer")
        }
        self.buffer = buf
        self.dim = dim
        self.capacity = capacity
        self.rawPointer = buf.contents().bindMemory(to: Float.self, capacity: capacity * dim)
    }

    public func setCount(_ n: Int) { count = n }

    public func insert(vector: [Float], at index: Int) throws {
        guard vector.count == dim else {
            throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
        }
        guard index < capacity else {
            throw ANNSError.constructionFailed("Index \(index) exceeds capacity \(capacity)")
        }
        let offset = index * dim
        vector.withUnsafeBufferPointer { src in
            rawPointer.advanced(by: offset).update(from: src.baseAddress!, count: dim)
        }
    }

    public func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        for (i, v) in vectors.enumerated() {
            try insert(vector: v, at: start + i)
        }
    }

    public func vector(at index: Int) -> [Float] {
        let offset = index * dim
        return Array(UnsafeBufferPointer(start: rawPointer.advanced(by: offset), count: dim))
    }

    /// Direct unsafe pointer access for GPU operations
    public var floatPointer: UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(start: rawPointer, count: capacity * dim)
    }
}
```

**Step 4: Run tests to verify they pass**

Expected: ALL PASS

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/VectorBuffer.swift Tests/MetalANNSTests/VectorBufferTests.swift
git commit -m "feat: add VectorBuffer for GPU-resident vector storage"
```

---

### Task 8: GraphBuffer

**Files:**
- Create: `Sources/MetalANNSCore/GraphBuffer.swift`
- Test: `Tests/MetalANNSTests/GraphBufferTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/GraphBufferTests.swift
import Testing
@testable import MetalANNSCore

@Suite("GraphBuffer Tests")
struct GraphBufferTests {
    @Test("Set and read neighbors for node 0")
    func setAndReadNeighbors() throws {
        let graph = try GraphBuffer(capacity: 10, degree: 4)
        let neighborIDs: [UInt32] = [3, 7, 1, 9]
        let neighborDists: [Float] = [0.1, 0.3, 0.05, 0.8]
        try graph.setNeighbors(of: 0, ids: neighborIDs, distances: neighborDists)

        let ids = graph.neighborIDs(of: 0)
        let dists = graph.neighborDistances(of: 0)
        #expect(ids == neighborIDs)
        #expect(dists == neighborDists)
    }

    @Test("Nodes are independent")
    func nodeIndependence() throws {
        let graph = try GraphBuffer(capacity: 10, degree: 2)
        try graph.setNeighbors(of: 0, ids: [1, 2], distances: [0.1, 0.2])
        try graph.setNeighbors(of: 1, ids: [5, 6], distances: [0.5, 0.6])
        #expect(graph.neighborIDs(of: 0) == [1, 2])
        #expect(graph.neighborIDs(of: 1) == [5, 6])
    }

    @Test("Capacity and degree are correct")
    func capacityAndDegree() throws {
        let graph = try GraphBuffer(capacity: 100, degree: 32)
        #expect(graph.capacity == 100)
        #expect(graph.degree == 32)
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — GraphBuffer not defined

**Step 3: Implement GraphBuffer**

```swift
// Sources/MetalANNSCore/GraphBuffer.swift
import Metal
import Foundation

/// GPU-resident adjacency list stored as flat 2D arrays.
/// Layout: `adjacency[nodeID * degree + slot]` = neighbor UInt32 ID
///         `distances[nodeID * degree + slot]` = Float32 distance
public final class GraphBuffer: @unchecked Sendable {
    public let adjacencyBuffer: MTLBuffer
    public let distanceBuffer: MTLBuffer
    public let degree: Int
    public let capacity: Int
    public private(set) var nodeCount: Int = 0

    private let idPointer: UnsafeMutablePointer<UInt32>
    private let distPointer: UnsafeMutablePointer<Float>

    public init(capacity: Int, degree: Int, device: MTLDevice? = nil) throws {
        let dev = device ?? MTLCreateSystemDefaultDevice()!
        let idBytes = capacity * degree * MemoryLayout<UInt32>.stride
        let distBytes = capacity * degree * MemoryLayout<Float>.stride

        guard let adjBuf = dev.makeBuffer(length: max(idBytes, 4), options: .storageModeShared),
              let distBuf = dev.makeBuffer(length: max(distBytes, 4), options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate GraphBuffer")
        }

        self.adjacencyBuffer = adjBuf
        self.distanceBuffer = distBuf
        self.degree = degree
        self.capacity = capacity
        self.idPointer = adjBuf.contents().bindMemory(to: UInt32.self, capacity: capacity * degree)
        self.distPointer = distBuf.contents().bindMemory(to: Float.self, capacity: capacity * degree)

        // Initialize all distances to FLT_MAX (sentinel for empty slots)
        for i in 0..<(capacity * degree) {
            distPointer[i] = Float.greatestFiniteMagnitude
            idPointer[i] = UInt32.max
        }
    }

    public func setCount(_ n: Int) { nodeCount = n }

    public func setNeighbors(of nodeID: Int, ids: [UInt32], distances: [Float]) throws {
        guard ids.count == degree, distances.count == degree else {
            throw ANNSError.constructionFailed("Neighbor count must equal degree \(degree)")
        }
        let base = nodeID * degree
        for i in 0..<degree {
            idPointer[base + i] = ids[i]
            distPointer[base + i] = distances[i]
        }
    }

    public func neighborIDs(of nodeID: Int) -> [UInt32] {
        let base = nodeID * degree
        return (0..<degree).map { idPointer[base + $0] }
    }

    public func neighborDistances(of nodeID: Int) -> [Float] {
        let base = nodeID * degree
        return (0..<degree).map { distPointer[base + $0] }
    }
}
```

**Step 4: Run tests to verify they pass**

Expected: ALL PASS

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/GraphBuffer.swift Tests/MetalANNSTests/GraphBufferTests.swift
git commit -m "feat: add GraphBuffer for GPU-resident adjacency list storage"
```

---

### Task 9: MetadataBuffer and ID Mapping

**Files:**
- Create: `Sources/MetalANNSCore/MetadataBuffer.swift`
- Create: `Sources/MetalANNSCore/IDMap.swift`
- Test: `Tests/MetalANNSTests/MetadataTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/MetadataTests.swift
import Testing
@testable import MetalANNSCore

@Suite("Metadata & IDMap Tests")
struct MetadataTests {
    @Test("MetadataBuffer roundtrip")
    func metadataRoundtrip() throws {
        let meta = try MetadataBuffer()
        meta.entryPointID = 42
        meta.nodeCount = 1000
        meta.degree = 32
        meta.dim = 128
        #expect(meta.entryPointID == 42)
        #expect(meta.nodeCount == 1000)
        #expect(meta.degree == 32)
        #expect(meta.dim == 128)
    }

    @Test("IDMap bidirectional mapping")
    func idMapMapping() {
        var idMap = IDMap()
        let internal0 = idMap.assign(externalID: "doc-a")
        let internal1 = idMap.assign(externalID: "doc-b")
        #expect(internal0 == 0)
        #expect(internal1 == 1)
        #expect(idMap.externalID(for: 0) == "doc-a")
        #expect(idMap.internalID(for: "doc-a") == 0)
    }

    @Test("IDMap rejects duplicate external IDs")
    func idMapDuplicate() {
        var idMap = IDMap()
        _ = idMap.assign(externalID: "doc-a")
        #expect(idMap.assign(externalID: "doc-a") == nil)
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — types not defined

**Step 3: Implement MetadataBuffer and IDMap**

```swift
// Sources/MetalANNSCore/MetadataBuffer.swift
import Metal

/// GPU-accessible metadata for the index.
public final class MetadataBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    private let pointer: UnsafeMutablePointer<UInt32>

    // Layout: [entryPointID, nodeCount, degree, dim, iterationCount]
    public init(device: MTLDevice? = nil) throws {
        let dev = device ?? MTLCreateSystemDefaultDevice()!
        let byteLength = 5 * MemoryLayout<UInt32>.stride
        guard let buf = dev.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw ANNSError.constructionFailed("Failed to allocate MetadataBuffer")
        }
        self.buffer = buf
        self.pointer = buf.contents().bindMemory(to: UInt32.self, capacity: 5)
        memset(buf.contents(), 0, byteLength)
    }

    public var entryPointID: UInt32 {
        get { pointer[0] }
        set { pointer[0] = newValue }
    }
    public var nodeCount: UInt32 {
        get { pointer[1] }
        set { pointer[1] = newValue }
    }
    public var degree: UInt32 {
        get { pointer[2] }
        set { pointer[2] = newValue }
    }
    public var dim: UInt32 {
        get { pointer[3] }
        set { pointer[3] = newValue }
    }
    public var iterationCount: UInt32 {
        get { pointer[4] }
        set { pointer[4] = newValue }
    }
}
```

```swift
// Sources/MetalANNSCore/IDMap.swift
import Foundation

/// Bidirectional mapping between external String IDs and internal UInt32 node IDs.
public struct IDMap: Sendable, Codable {
    private var externalToInternal: [String: UInt32] = [:]
    private var internalToExternal: [UInt32: String] = [:]
    private var nextID: UInt32 = 0

    public init() {}

    /// Assign a new internal ID for an external ID. Returns nil if already exists.
    public mutating func assign(externalID: String) -> UInt32? {
        guard externalToInternal[externalID] == nil else { return nil }
        let id = nextID
        externalToInternal[externalID] = id
        internalToExternal[id] = externalID
        nextID += 1
        return id
    }

    public func internalID(for externalID: String) -> UInt32? {
        externalToInternal[externalID]
    }

    public func externalID(for internalID: UInt32) -> String? {
        internalToExternal[internalID]
    }

    public var count: Int { externalToInternal.count }
}
```

**Step 4: Run tests to verify they pass**

Expected: ALL PASS

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/MetadataBuffer.swift Sources/MetalANNSCore/IDMap.swift Tests/MetalANNSTests/MetadataTests.swift
git commit -m "feat: add MetadataBuffer and bidirectional IDMap"
```

---

## Phase 3: Graph Construction — NN-Descent via Metal

### Task 10: CPU NN-Descent (Reference Implementation)

This task implements a pure-CPU NN-Descent using Accelerate for distance computation. This serves as the ground truth for GPU construction and enables full TDD on simulator.

**Files:**
- Create: `Sources/MetalANNSCore/NNDescentCPU.swift`
- Test: `Tests/MetalANNSTests/NNDescentCPUTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/NNDescentCPUTests.swift
import Testing
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("CPU NN-Descent Tests")
struct NNDescentCPUTests {
    @Test("Constructs graph with correct dimensions")
    func graphDimensions() async throws {
        let n = 50
        let dim = 8
        let degree = 4
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        let (graph, _) = try await NNDescentCPU.build(
            vectors: vectors, degree: degree, metric: .cosine, maxIterations: 10
        )
        #expect(graph.count == n)
        for neighbors in graph {
            #expect(neighbors.count == degree)
        }
    }

    @Test("No self-loops in constructed graph")
    func noSelfLoops() async throws {
        let n = 50
        let dim = 8
        let degree = 4
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        let (graph, _) = try await NNDescentCPU.build(
            vectors: vectors, degree: degree, metric: .cosine, maxIterations: 10
        )
        for (nodeID, neighbors) in graph.enumerated() {
            for (nid, _) in neighbors {
                #expect(nid != UInt32(nodeID), "Self-loop found at node \(nodeID)")
            }
        }
    }

    @Test("Recall > 0.85 for 50 nodes, d=4, 5 iterations")
    func recallCheck() async throws {
        let n = 50
        let dim = 8
        let degree = 4
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        // Build graph
        let (graph, _) = try await NNDescentCPU.build(
            vectors: vectors, degree: degree, metric: .cosine, maxIterations: 10
        )

        // Compute exact brute-force kNN
        var totalRecall: Float = 0
        let backend = AccelerateBackend()
        let flat = vectors.flatMap { $0 }

        for i in 0..<n {
            let distances = try await flat.withUnsafeBufferPointer { ptr in
                try await backend.computeDistances(
                    query: vectors[i], vectors: ptr, vectorCount: n, dim: dim, metric: .cosine
                )
            }
            // Get exact top-degree neighbors (excluding self)
            let exactNeighbors = distances.enumerated()
                .filter { $0.offset != i }
                .sorted { $0.element < $1.element }
                .prefix(degree)
                .map { UInt32($0.offset) }

            let graphNeighbors = Set(graph[i].map(\.0))
            let exactSet = Set(exactNeighbors)
            let overlap = graphNeighbors.intersection(exactSet).count
            totalRecall += Float(overlap) / Float(degree)
        }

        let avgRecall = totalRecall / Float(n)
        #expect(avgRecall > 0.85, "Recall \(avgRecall) is below 0.85 threshold")
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — NNDescentCPU not defined

**Step 3: Implement NNDescentCPU**

```swift
// Sources/MetalANNSCore/NNDescentCPU.swift
import Foundation
import Accelerate
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "NNDescentCPU")

/// CPU reference implementation of NN-Descent for testing and simulator use.
/// Returns: (graph: [[(UInt32, Float)]], entryPoint: UInt32)
/// graph[i] is node i's neighbor list sorted by distance ascending, each entry is (neighborID, distance).
public enum NNDescentCPU {
    public static func build(
        vectors: [[Float]],
        degree: Int,
        metric: Metric,
        maxIterations: Int = 20,
        convergenceThreshold: Float = 0.001
    ) async throws -> (graph: [[(UInt32, Float)]], entryPoint: UInt32) {
        let n = vectors.count
        let dim = vectors[0].count
        let backend = AccelerateBackend()
        let flatVectors = vectors.flatMap { $0 }

        // Helper: compute distance between two nodes
        func dist(_ a: Int, _ b: Int) -> Float {
            var dot: Float = 0
            var normA: Float = 0
            var normB: Float = 0
            for d in 0..<dim {
                let va = vectors[a][d]
                let vb = vectors[b][d]
                switch metric {
                case .cosine:
                    dot += va * vb
                    normA += va * va
                    normB += vb * vb
                case .l2:
                    let diff = va - vb
                    dot += diff * diff
                case .innerProduct:
                    dot += va * vb
                }
            }
            switch metric {
            case .cosine:
                let denom = sqrt(normA) * sqrt(normB)
                return denom < 1e-10 ? 1.0 : (1.0 - dot / denom)
            case .l2:
                return dot
            case .innerProduct:
                return -dot
            }
        }

        // Step 1: Random initialization
        var graph = [[( UInt32, Float)]](repeating: [], count: n)
        for i in 0..<n {
            var neighbors = Set<Int>()
            while neighbors.count < degree {
                let r = Int.random(in: 0..<n)
                if r != i { neighbors.insert(r) }
            }
            graph[i] = neighbors.map { j in (UInt32(j), dist(i, j)) }
                .sorted { $0.1 < $1.1 }
        }

        // Step 2: NN-Descent iterations
        for iter in 0..<maxIterations {
            var updateCount = 0

            // Build reverse lists
            var reverse = [[UInt32]](repeating: [], count: n)
            for i in 0..<n {
                for (nid, _) in graph[i] {
                    reverse[Int(nid)].append(UInt32(i))
                }
            }

            // Local join: for each node, check pairs from forward x reverse
            for u in 0..<n {
                let forward = graph[u].map { Int($0.0) }
                let rev = reverse[u].map { Int($0) }

                // All pairs from forward union reverse
                let candidates = Set(forward + rev)
                let candidateArray = Array(candidates)

                for ci in 0..<candidateArray.count {
                    for cj in (ci + 1)..<candidateArray.count {
                        let a = candidateArray[ci]
                        let b = candidateArray[cj]
                        if a == b { continue }
                        let d = dist(a, b)

                        // Try to update a's neighbor list
                        if d < graph[a].last!.1 {
                            // Check not already a neighbor and not self
                            if !graph[a].contains(where: { Int($0.0) == b }) {
                                graph[a][graph[a].count - 1] = (UInt32(b), d)
                                graph[a].sort { $0.1 < $1.1 }
                                updateCount += 1
                            }
                        }
                        // Try to update b's neighbor list
                        if d < graph[b].last!.1 {
                            if !graph[b].contains(where: { Int($0.0) == a }) {
                                graph[b][graph[b].count - 1] = (UInt32(a), d)
                                graph[b].sort { $0.1 < $1.1 }
                                updateCount += 1
                            }
                        }
                    }
                }
            }

            logger.debug("NN-Descent iter \(iter): \(updateCount) updates")

            if Float(updateCount) < convergenceThreshold * Float(degree) * Float(n) {
                logger.debug("Converged after \(iter + 1) iterations")
                break
            }
        }

        // Step 3: Pick entry point (node with minimum mean distance to neighbors)
        var entryPoint: UInt32 = 0
        var bestMeanDist = Float.greatestFiniteMagnitude
        for i in 0..<n {
            let meanDist = graph[i].map(\.1).reduce(0, +) / Float(degree)
            if meanDist < bestMeanDist {
                bestMeanDist = meanDist
                entryPoint = UInt32(i)
            }
        }

        return (graph, entryPoint)
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/NNDescentCPUTests 2>&1 | tail -15`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add Sources/MetalANNSCore/NNDescentCPU.swift Tests/MetalANNSTests/NNDescentCPUTests.swift
git commit -m "feat: implement CPU NN-Descent reference (Accelerate backend)"
```

---

### Task 11: Metal NN-Descent Shaders — Random Init & Initial Distances

**Files:**
- Create: `Sources/MetalANNSCore/Shaders/NNDescent.metal`
- Test: `Tests/MetalANNSTests/NNDescentGPUTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/NNDescentGPUTests.swift
import Testing
import Metal
@testable import MetalANNSCore

@Suite("GPU NN-Descent Tests")
struct NNDescentGPUTests {
    @Test("Random init produces valid graph (no self-loops, valid IDs)")
    func randomInitValid() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Metal not available on simulator")
        #endif
        let context = try MetalContext()
        let n: UInt32 = 100
        let degree: UInt32 = 8
        let graph = try GraphBuffer(capacity: Int(n), degree: Int(degree), device: context.device)

        try await NNDescentGPU.randomInit(
            context: context, graph: graph, nodeCount: Int(n), seed: 42
        )

        for i in 0..<Int(n) {
            let ids = graph.neighborIDs(of: i)
            for nid in ids {
                #expect(nid != UInt32(i), "Self-loop at node \(i)")
                #expect(nid < n, "Invalid neighbor ID \(nid) at node \(i)")
            }
            // Check no duplicates
            #expect(Set(ids).count == Int(degree), "Duplicate neighbors at node \(i)")
        }
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — NNDescentGPU and random_init kernel not defined

**Step 3: Write the random_init Metal shader**

```metal
// Sources/MetalANNSCore/Shaders/NNDescent.metal
#include <metal_stdlib>
using namespace metal;

/// Fast LCG PRNG
inline uint lcg_next(uint state) {
    return state * 1664525u + 1013904223u;
}

/// Random graph initialization: assign d random neighbors to each node.
/// No self-loops, no duplicates (best effort via retry).
kernel void random_init(
    device uint     *adjacency    [[buffer(0)]],  // nodeCount * degree
    constant uint   &node_count   [[buffer(1)]],
    constant uint   &degree       [[buffer(2)]],
    constant uint   &seed         [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= node_count) return;

    uint state = seed ^ (tid * 2654435761u);  // per-node seed
    uint base = tid * degree;

    for (uint s = 0; s < degree; s++) {
        uint neighbor;
        bool valid = false;
        for (uint attempt = 0; attempt < 100 && !valid; attempt++) {
            state = lcg_next(state);
            neighbor = state % node_count;
            if (neighbor == tid) continue;
            // Check for duplicates
            valid = true;
            for (uint prev = 0; prev < s; prev++) {
                if (adjacency[base + prev] == neighbor) {
                    valid = false;
                    break;
                }
            }
        }
        adjacency[base + s] = neighbor;
    }
}

/// Compute initial distances for the random graph.
/// One thread per (node, neighbor_slot).
kernel void compute_initial_distances(
    device const float *vectors       [[buffer(0)]],  // nodeCount * dim
    device const uint  *adjacency     [[buffer(1)]],  // nodeCount * degree
    device float       *distances     [[buffer(2)]],  // nodeCount * degree
    constant uint      &node_count    [[buffer(3)]],
    constant uint      &degree        [[buffer(4)]],
    constant uint      &dim           [[buffer(5)]],
    constant uint      &metric_type   [[buffer(6)]],  // 0=cosine, 1=l2, 2=innerProduct
    uint tid [[thread_position_in_grid]]
) {
    uint total = node_count * degree;
    if (tid >= total) return;

    uint node = tid / degree;
    uint slot = tid % degree;
    uint neighbor = adjacency[tid];

    if (neighbor >= node_count) {
        distances[tid] = FLT_MAX;
        return;
    }

    uint baseA = node * dim;
    uint baseB = neighbor * dim;

    float result = 0.0;

    if (metric_type == 0) { // cosine
        float dot = 0, normA = 0, normB = 0;
        for (uint d = 0; d < dim; d++) {
            float a = vectors[baseA + d];
            float b = vectors[baseB + d];
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }
        float denom = sqrt(normA) * sqrt(normB);
        result = (denom < 1e-10f) ? 1.0f : (1.0f - dot / denom);
    } else if (metric_type == 1) { // l2
        for (uint d = 0; d < dim; d++) {
            float diff = vectors[baseA + d] - vectors[baseB + d];
            result += diff * diff;
        }
    } else { // innerProduct
        float dot = 0;
        for (uint d = 0; d < dim; d++) {
            dot += vectors[baseA + d] * vectors[baseB + d];
        }
        result = -dot;
    }

    distances[tid] = result;
}
```

**Step 4: Implement NNDescentGPU Swift wrapper (partial)**

```swift
// Sources/MetalANNSCore/NNDescentGPU.swift
import Metal
import os.log

private let logger = Logger(subsystem: "com.metalanns", category: "NNDescentGPU")

/// GPU implementation of NN-Descent graph construction using Metal compute shaders.
public enum NNDescentGPU {
    public static func randomInit(
        context: MetalContext,
        graph: GraphBuffer,
        nodeCount: Int,
        seed: UInt32 = 42
    ) async throws {
        let pipeline = try await context.pipelineCache.pipeline(for: "random_init")
        var n = UInt32(nodeCount)
        var d = UInt32(graph.degree)
        var s = seed

        try await context.execute { commandBuffer in
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 0)
            encoder.setBytes(&n, length: 4, index: 1)
            encoder.setBytes(&d, length: 4, index: 2)
            encoder.setBytes(&s, length: 4, index: 3)

            let threadsPerGrid = MTLSize(width: nodeCount, height: 1, depth: 1)
            let threadsPerGroup = MTLSize(
                width: min(nodeCount, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }

        logger.debug("Random init complete: \(nodeCount) nodes, degree=\(graph.degree)")
    }

    public static func computeInitialDistances(
        context: MetalContext,
        vectors: VectorBuffer,
        graph: GraphBuffer,
        nodeCount: Int,
        metric: Metric
    ) async throws {
        let pipeline = try await context.pipelineCache.pipeline(for: "compute_initial_distances")
        var n = UInt32(nodeCount)
        var d = UInt32(graph.degree)
        var dim = UInt32(vectors.dim)
        var metricType: UInt32 = switch metric {
            case .cosine: 0
            case .l2: 1
            case .innerProduct: 2
        }

        let totalThreads = nodeCount * graph.degree

        try await context.execute { commandBuffer in
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(vectors.buffer, offset: 0, index: 0)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 1)
            encoder.setBuffer(graph.distanceBuffer, offset: 0, index: 2)
            encoder.setBytes(&n, length: 4, index: 3)
            encoder.setBytes(&d, length: 4, index: 4)
            encoder.setBytes(&dim, length: 4, index: 5)
            encoder.setBytes(&metricType, length: 4, index: 6)

            let threadsPerGrid = MTLSize(width: totalThreads, height: 1, depth: 1)
            let threadsPerGroup = MTLSize(
                width: min(totalThreads, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }
    }
}
```

**Step 5: Run tests to verify they pass**

Expected: ALL PASS

**Step 6: Commit**

```bash
git add Sources/MetalANNSCore/Shaders/NNDescent.metal Sources/MetalANNSCore/NNDescentGPU.swift Tests/MetalANNSTests/NNDescentGPUTests.swift
git commit -m "feat: add Metal random_init and compute_initial_distances kernels"
```

---

### Task 12: Metal Reverse Edge & Local Join Kernels

**Files:**
- Modify: `Sources/MetalANNSCore/Shaders/NNDescent.metal` (add reverse + local join kernels)
- Modify: `Sources/MetalANNSCore/NNDescentGPU.swift` (add Swift wrappers)
- Test: `Tests/MetalANNSTests/NNDescentGPUTests.swift` (add integration test)

This is the most complex task. The local join kernel must:
1. Build reverse edges atomically
2. For each node, examine forward x reverse neighbor pairs
3. Compute distances and update graph via CAS
4. Count updates atomically for convergence

**Step 1: Add integration test**

Add to `NNDescentGPUTests.swift`:

```swift
@Test("Full GPU NN-Descent: 200 nodes, d=8, recall > 0.80")
func fullGPUConstruction() async throws {
    #if targetEnvironment(simulator)
    throw XCTSkip("Metal not available on simulator")
    #endif
    let context = try MetalContext()
    let n = 200
    let dim = 16
    let degree = 8
    let maxIter = 15

    // Generate random vectors
    let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
    let vectorBuffer = try VectorBuffer(capacity: n, dim: dim, device: context.device)
    try vectorBuffer.batchInsert(vectors: vectors, startingAt: 0)
    vectorBuffer.setCount(n)

    let graph = try GraphBuffer(capacity: n, degree: degree, device: context.device)

    // Run full GPU NN-Descent
    try await NNDescentGPU.build(
        context: context,
        vectors: vectorBuffer,
        graph: graph,
        nodeCount: n,
        metric: .cosine,
        maxIterations: maxIter
    )

    // Compute recall against brute-force
    let backend = AccelerateBackend()
    let flat = vectors.flatMap { $0 }
    var totalRecall: Float = 0

    for i in 0..<n {
        let distances = try await flat.withUnsafeBufferPointer { ptr in
            try await backend.computeDistances(
                query: vectors[i], vectors: ptr, vectorCount: n, dim: dim, metric: .cosine
            )
        }
        let exactNeighbors = Set(distances.enumerated()
            .filter { $0.offset != i }
            .sorted { $0.element < $1.element }
            .prefix(degree)
            .map { UInt32($0.offset) })

        let graphNeighbors = Set(graph.neighborIDs(of: i).filter { $0 != UInt32.max })
        let overlap = graphNeighbors.intersection(exactNeighbors).count
        totalRecall += Float(overlap) / Float(degree)
    }

    let avgRecall = totalRecall / Float(n)
    #expect(avgRecall > 0.80, "GPU recall \(avgRecall) is below 0.80 threshold")
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `NNDescentGPU.build` not implemented

**Step 3: Implement reverse edge and local join kernels**

Add to `NNDescent.metal`:

```metal
/// Build reverse edge list.
/// For each edge u->v in forward graph, atomically add u to reverse_list[v].
kernel void build_reverse_list(
    device const uint  *adjacency       [[buffer(0)]],
    device uint        *reverse_list    [[buffer(1)]],  // nodeCount * maxReverse
    device atomic_uint *reverse_counts  [[buffer(2)]],  // nodeCount
    constant uint      &node_count      [[buffer(3)]],
    constant uint      &degree          [[buffer(4)]],
    constant uint      &max_reverse     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = node_count * degree;
    if (tid >= total) return;

    uint u = tid / degree;
    uint v = adjacency[tid];
    if (v >= node_count) return;

    uint slot = atomic_fetch_add_explicit(&reverse_counts[v], 1u, memory_order_relaxed);
    if (slot < max_reverse) {
        reverse_list[v * max_reverse + slot] = u;
    }
}

/// Local join: for each node, check forward x reverse pairs and update graph.
/// One thread per node.
kernel void local_join(
    device const float *vectors          [[buffer(0)]],
    device atomic_uint *adj_ids          [[buffer(1)]],  // adjacency as atomic
    device atomic_uint *adj_dists_bits   [[buffer(2)]],  // distances as uint bits (atomic)
    device const uint  *reverse_list     [[buffer(3)]],
    device const uint  *reverse_counts_r [[buffer(4)]],  // non-atomic read copy
    constant uint      &node_count       [[buffer(5)]],
    constant uint      &degree           [[buffer(6)]],
    constant uint      &max_reverse      [[buffer(7)]],
    constant uint      &dim              [[buffer(8)]],
    constant uint      &metric_type      [[buffer(9)]],
    device atomic_uint *update_counter   [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= node_count) return;

    uint u = tid;
    uint fwd_base = u * degree;
    uint rev_base = u * max_reverse;
    uint rev_count = min(reverse_counts_r[u], max_reverse);

    // Collect forward neighbor IDs
    uint fwd[64];  // max degree
    uint fwd_count = 0;
    for (uint s = 0; s < degree && fwd_count < 64; s++) {
        uint nid = atomic_load_explicit(&adj_ids[fwd_base + s], memory_order_relaxed);
        if (nid < node_count) {
            fwd[fwd_count++] = nid;
        }
    }

    // Collect reverse neighbor IDs
    uint rev[128];  // max reverse
    uint actual_rev = min(rev_count, 128u);
    for (uint s = 0; s < actual_rev; s++) {
        rev[s] = reverse_list[rev_base + s];
    }

    // Check all pairs from forward x reverse
    for (uint fi = 0; fi < fwd_count; fi++) {
        for (uint ri = 0; ri < actual_rev; ri++) {
            uint a = fwd[fi];
            uint b = rev[ri];
            if (a == b) continue;

            // Compute distance between a and b
            uint baseA = a * dim;
            uint baseB = b * dim;
            float d = 0.0;

            if (metric_type == 0) { // cosine
                float dot = 0, normA = 0, normB = 0;
                for (uint dd = 0; dd < dim; dd++) {
                    float va = vectors[baseA + dd];
                    float vb = vectors[baseB + dd];
                    dot += va * vb;
                    normA += va * va;
                    normB += vb * vb;
                }
                float denom = sqrt(normA) * sqrt(normB);
                d = (denom < 1e-10f) ? 1.0f : (1.0f - dot / denom);
            } else if (metric_type == 1) { // l2
                for (uint dd = 0; dd < dim; dd++) {
                    float diff = vectors[baseA + dd] - vectors[baseB + dd];
                    d += diff * diff;
                }
            } else { // inner product
                float dot = 0;
                for (uint dd = 0; dd < dim; dd++) {
                    dot += vectors[baseA + dd] * vectors[baseB + dd];
                }
                d = -dot;
            }

            uint d_bits = as_type<uint>(d);

            // Try to insert b into a's neighbor list (replace worst)
            // Find worst slot in a's list
            uint a_base = a * degree;
            uint worst_slot = 0;
            uint worst_bits = 0;
            for (uint s = 0; s < degree; s++) {
                uint cur = atomic_load_explicit(&adj_dists_bits[a_base + s], memory_order_relaxed);
                if (cur > worst_bits) {
                    worst_bits = cur;
                    worst_slot = s;
                }
            }
            if (d_bits < worst_bits) {
                // Check b not already in a's list
                bool exists = false;
                for (uint s = 0; s < degree; s++) {
                    if (atomic_load_explicit(&adj_ids[a_base + s], memory_order_relaxed) == b) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    // CAS on distance
                    uint expected = worst_bits;
                    if (atomic_compare_exchange_weak_explicit(
                        &adj_dists_bits[a_base + worst_slot],
                        &expected, d_bits,
                        memory_order_relaxed, memory_order_relaxed))
                    {
                        atomic_store_explicit(&adj_ids[a_base + worst_slot], b, memory_order_relaxed);
                        atomic_fetch_add_explicit(update_counter, 1u, memory_order_relaxed);
                    }
                }
            }

            // Try to insert a into b's neighbor list (symmetric)
            uint b_base = b * degree;
            worst_slot = 0;
            worst_bits = 0;
            for (uint s = 0; s < degree; s++) {
                uint cur = atomic_load_explicit(&adj_dists_bits[b_base + s], memory_order_relaxed);
                if (cur > worst_bits) {
                    worst_bits = cur;
                    worst_slot = s;
                }
            }
            if (d_bits < worst_bits) {
                bool exists = false;
                for (uint s = 0; s < degree; s++) {
                    if (atomic_load_explicit(&adj_ids[b_base + s], memory_order_relaxed) == a) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    uint expected = worst_bits;
                    if (atomic_compare_exchange_weak_explicit(
                        &adj_dists_bits[b_base + worst_slot],
                        &expected, d_bits,
                        memory_order_relaxed, memory_order_relaxed))
                    {
                        atomic_store_explicit(&adj_ids[b_base + worst_slot], a, memory_order_relaxed);
                        atomic_fetch_add_explicit(update_counter, 1u, memory_order_relaxed);
                    }
                }
            }
        }
    }
}
```

**Step 4: Implement full NNDescentGPU.build orchestrator**

Add to `NNDescentGPU.swift`:

```swift
public static func build(
    context: MetalContext,
    vectors: VectorBuffer,
    graph: GraphBuffer,
    nodeCount: Int,
    metric: Metric,
    maxIterations: Int = 20,
    convergenceThreshold: Float = 0.001
) async throws {
    let degree = graph.degree
    let maxReverse = degree * 2

    // Step 1: Random init
    try await randomInit(context: context, graph: graph, nodeCount: nodeCount)

    // Step 2: Compute initial distances
    try await computeInitialDistances(
        context: context, vectors: vectors, graph: graph,
        nodeCount: nodeCount, metric: metric
    )

    // Allocate reverse list buffers
    let reverseListBuffer = context.device.makeBuffer(
        length: nodeCount * maxReverse * MemoryLayout<UInt32>.stride,
        options: .storageModeShared)!
    let reverseCountBuffer = context.device.makeBuffer(
        length: nodeCount * MemoryLayout<UInt32>.stride,
        options: .storageModeShared)!
    let updateCountBuffer = context.device.makeBuffer(
        length: MemoryLayout<UInt32>.stride,
        options: .storageModeShared)!

    let buildReversePipeline = try await context.pipelineCache.pipeline(for: "build_reverse_list")
    let localJoinPipeline = try await context.pipelineCache.pipeline(for: "local_join")

    var metricType: UInt32 = switch metric {
        case .cosine: 0
        case .l2: 1
        case .innerProduct: 2
    }
    var n = UInt32(nodeCount)
    var d = UInt32(degree)
    var mr = UInt32(maxReverse)
    var dim = UInt32(vectors.dim)

    // NN-Descent iteration loop
    for iter in 0..<maxIterations {
        // Clear reverse counts and update counter
        memset(reverseCountBuffer.contents(), 0, nodeCount * MemoryLayout<UInt32>.stride)
        memset(updateCountBuffer.contents(), 0, MemoryLayout<UInt32>.stride)

        // Build reverse list
        try await context.execute { commandBuffer in
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(buildReversePipeline)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 0)
            encoder.setBuffer(reverseListBuffer, offset: 0, index: 1)
            encoder.setBuffer(reverseCountBuffer, offset: 0, index: 2)
            encoder.setBytes(&n, length: 4, index: 3)
            encoder.setBytes(&d, length: 4, index: 4)
            encoder.setBytes(&mr, length: 4, index: 5)
            let total = nodeCount * degree
            encoder.dispatchThreads(
                MTLSize(width: total, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(total, buildReversePipeline.maxTotalThreadsPerThreadgroup),
                    height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Local join
        try await context.execute { commandBuffer in
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(localJoinPipeline)
            encoder.setBuffer(vectors.buffer, offset: 0, index: 0)
            encoder.setBuffer(graph.adjacencyBuffer, offset: 0, index: 1)
            encoder.setBuffer(graph.distanceBuffer, offset: 0, index: 2)
            encoder.setBuffer(reverseListBuffer, offset: 0, index: 3)
            encoder.setBuffer(reverseCountBuffer, offset: 0, index: 4)
            encoder.setBytes(&n, length: 4, index: 5)
            encoder.setBytes(&d, length: 4, index: 6)
            encoder.setBytes(&mr, length: 4, index: 7)
            encoder.setBytes(&dim, length: 4, index: 8)
            encoder.setBytes(&metricType, length: 4, index: 9)
            encoder.setBuffer(updateCountBuffer, offset: 0, index: 10)
            encoder.dispatchThreads(
                MTLSize(width: nodeCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(nodeCount, localJoinPipeline.maxTotalThreadsPerThreadgroup),
                    height: 1, depth: 1))
            encoder.endEncoding()
        }

        let updateCount = updateCountBuffer.contents()
            .bindMemory(to: UInt32.self, capacity: 1).pointee
        logger.debug("GPU NN-Descent iter \(iter): \(updateCount) updates")

        if Float(updateCount) < convergenceThreshold * Float(degree) * Float(nodeCount) {
            logger.debug("GPU converged after \(iter + 1) iterations")
            break
        }
    }

    graph.setCount(nodeCount)
}
```

**Step 5: Run tests to verify they pass**

Run: `xcodebuild test -scheme MetalANNS -destination 'platform=macOS' -only-testing MetalANNSTests/NNDescentGPUTests 2>&1 | tail -15`
Expected: ALL PASS (recall > 0.80)

**Step 6: Commit**

```bash
git add Sources/MetalANNSCore/Shaders/NNDescent.metal Sources/MetalANNSCore/NNDescentGPU.swift Tests/MetalANNSTests/NNDescentGPUTests.swift
git commit -m "feat: implement GPU NN-Descent with reverse edges, local join, and convergence"
```

---

### Task 13: Bitonic Sort Kernel

**Files:**
- Create: `Sources/MetalANNSCore/Shaders/Sort.metal`
- Modify: `Sources/MetalANNSCore/NNDescentGPU.swift` (add sort dispatch)
- Test: `Tests/MetalANNSTests/BitonicSortTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/BitonicSortTests.swift
import Testing
import Metal
@testable import MetalANNSCore

@Suite("Bitonic Sort Tests")
struct BitonicSortTests {
    @Test("Sort 100 nodes with d=32 by distance ascending")
    func sortNeighborLists() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Metal not available on simulator")
        #endif
        let context = try MetalContext()
        let n = 100
        let degree = 32
        let graph = try GraphBuffer(capacity: n, degree: degree, device: context.device)

        // Fill with random distances and IDs
        for i in 0..<n {
            let ids = (0..<degree).map { _ in UInt32.random(in: 0..<UInt32(n)) }
            let dists = (0..<degree).map { _ in Float.random(in: 0...10) }
            try graph.setNeighbors(of: i, ids: ids, distances: dists)
        }

        try await NNDescentGPU.sortNeighborLists(
            context: context, graph: graph, nodeCount: n
        )

        // Verify each node's list is sorted ascending
        for i in 0..<n {
            let dists = graph.neighborDistances(of: i)
            for j in 1..<degree {
                #expect(dists[j] >= dists[j - 1],
                    "Node \(i): dist[\(j)]=\(dists[j]) < dist[\(j-1)]=\(dists[j-1])")
            }
        }
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL

**Step 3: Write Sort.metal kernel** (use the bitonic sort from research)

Implement the bitonic sort kernel in `Sources/MetalANNSCore/Shaders/Sort.metal` as described in the research findings (threadgroup shared memory, one threadgroup per node, `degree/2` threads per threadgroup).

**Step 4: Add `sortNeighborLists` to NNDescentGPU.swift** and integrate into the build loop (call after each local join iteration).

**Step 5: Run tests to verify they pass**

**Step 6: Commit**

```bash
git commit -m "feat: add bitonic sort kernel for neighbor list ordering"
```

---

## Phase 4: Search — Beam Search & Query API

### Task 14: CPU Beam Search (Reference)

**Files:**
- Create: `Sources/MetalANNSCore/BeamSearchCPU.swift`
- Create: `Sources/MetalANNS/SearchResult.swift`
- Test: `Tests/MetalANNSTests/SearchTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/MetalANNSTests/SearchTests.swift
import Testing
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("Search Tests")
struct SearchTests {
    @Test("CPU beam search returns k results")
    func cpuSearchReturnsK() async throws {
        let n = 100
        let dim = 16
        let degree = 8
        let k = 5
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        // Build CPU graph
        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors, degree: degree, metric: .cosine, maxIterations: 10
        )

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let results = try await BeamSearchCPU.search(
            query: query, vectors: vectors, graph: graphData,
            entryPoint: Int(entryPoint), k: k, ef: 32, metric: .cosine
        )

        #expect(results.count == k)
        // Results should be sorted by distance ascending
        for i in 1..<results.count {
            #expect(results[i].score >= results[i - 1].score)
        }
    }

    @Test("CPU search recall > 0.90 on 1000 vectors")
    func cpuSearchRecall() async throws {
        let n = 1000
        let dim = 32
        let degree = 16
        let k = 10
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        let (graphData, entryPoint) = try await NNDescentCPU.build(
            vectors: vectors, degree: degree, metric: .cosine, maxIterations: 15
        )

        let backend = AccelerateBackend()
        let flat = vectors.flatMap { $0 }
        var totalRecall: Float = 0
        let numQueries = 20

        for _ in 0..<numQueries {
            let query = (0..<dim).map { _ in Float.random(in: -1...1) }

            let results = try await BeamSearchCPU.search(
                query: query, vectors: vectors, graph: graphData,
                entryPoint: Int(entryPoint), k: k, ef: 64, metric: .cosine
            )

            let exactDists = try await flat.withUnsafeBufferPointer { ptr in
                try await backend.computeDistances(
                    query: query, vectors: ptr, vectorCount: n, dim: dim, metric: .cosine
                )
            }
            let exactTop = Set(exactDists.enumerated()
                .sorted { $0.element < $1.element }
                .prefix(k)
                .map { UInt32($0.offset) })
            let approxTop = Set(results.map(\.internalID))
            let overlap = approxTop.intersection(exactTop).count
            totalRecall += Float(overlap) / Float(k)
        }

        let avgRecall = totalRecall / Float(numQueries)
        #expect(avgRecall > 0.90, "Search recall \(avgRecall) below 0.90")
    }
}
```

**Step 2: Implement SearchResult**

```swift
// Sources/MetalANNS/SearchResult.swift
public struct SearchResult: Sendable {
    public let id: String
    public let score: Float
    public let internalID: UInt32

    public init(id: String, score: Float, internalID: UInt32) {
        self.id = id
        self.score = score
        self.internalID = internalID
    }
}
```

**Step 3: Implement BeamSearchCPU**

The CPU beam search navigates the graph greedily:
1. Start from entry point, add to candidate priority queue
2. Pop best unvisited candidate, expand its neighbors
3. Compute distances to neighbors, add to queue if better
4. Repeat until no unvisited candidates
5. Return top-k from queue

**Step 4: Run tests, commit**

---

### Task 15: Metal Beam Search Kernel

**Files:**
- Create: `Sources/MetalANNSCore/Shaders/Search.metal`
- Create: `Sources/MetalANNSCore/SearchGPU.swift`
- Test: `Tests/MetalANNSTests/MetalSearchTests.swift`

Implement the GPU beam search kernel with:
- One threadgroup per query for batch search
- Candidate queue in threadgroup shared memory (size = ef_search)
- Visited bitset as threadgroup array of UInt32
- Parallel neighbor distance computation within threadgroup
- Output top-k (nodeID, distance) pairs per query

**Step 1: Test GPU search recall matches CPU search recall**

**Step 2: Write Search.metal kernel**

**Step 3: Write SearchGPU.swift wrapper**

**Step 4: Run tests, commit**

---

## Phase 5: Persistence & Incremental Operations

### Task 16: Index Serialization

**Files:**
- Create: `Sources/MetalANNS/Persistence.swift`
- Test: `Tests/MetalANNSTests/PersistenceTests.swift`

File format:
```
Header: 'MANN' (4 bytes) | version (UInt32) | nodeCount (UInt32) | degree (UInt32) | dim (UInt32) | metric (UInt32)
Body:   vectorBuffer bytes | adjacencyBuffer bytes | distanceBuffer bytes | idMap JSON length (UInt32) | idMap JSON
```

**Step 1: Write tests**
- Build small index, save, load, run queries, verify identical results
- Corrupt header, verify load throws `ANNSError.corruptFile`

**Step 2: Implement save/load**

**Step 3: Run tests, commit**

---

### Task 17: Incremental Insert

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift`
- Test: `Tests/MetalANNSTests/IncrementalTests.swift`

Strategy: search for d nearest neighbors, use as initial neighbors for new node, run 3 local NN-Descent iterations on affected neighborhood.

**Step 1: Write test** — build on 500 vectors, insert 50 more, verify recall degrades < 3%

**Step 2: Implement insert**

**Step 3: Run tests, commit**

---

### Task 18: Soft Deletion

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift`
- Test: `Tests/MetalANNSTests/DeletionTests.swift`

**Step 1: Write test** — insert 100, delete 20, verify deleted IDs never in results

**Step 2: Implement soft delete with `Set<UInt32>`**

**Step 3: Run tests, commit**

---

## Phase 6: Public API & Polish

### Task 19: ANNSIndex Actor (Public API)

**Files:**
- Modify: `Sources/MetalANNS/ANNSIndex.swift`
- Test: `Tests/MetalANNSTests/ANNSIndexTests.swift`

```swift
public actor ANNSIndex {
    public init(configuration: IndexConfiguration = .default)
    public func build(vectors: [[Float]], ids: [String]) async throws
    public func insert(_ vector: [Float], id: String) async throws
    public func delete(id: String) throws
    public func search(query: [Float], k: Int) async throws -> [SearchResult]
    public func batchSearch(queries: [[Float]], k: Int) async throws -> [[SearchResult]]
    public func save(to url: URL) throws
    public static func load(from url: URL) throws -> ANNSIndex
    public var count: Int { get }
}
```

**Step 1: Write integration test** — full lifecycle: init, build, search, insert, delete, save, load, search again

**Step 2: Implement the actor** wiring all internal components together

**Step 3: Run all tests, commit**

---

### Task 20: Full Integration Test & Recall Benchmark

**Files:**
- Test: `Tests/MetalANNSTests/IntegrationTests.swift`
- Create: `Sources/MetalANNSBenchmarks/BenchmarkRunner.swift`

**Step 1: Write comprehensive integration test**
- 1000 random 128-dim vectors
- Build graph
- 100 queries, compute recall@10
- Assert recall > 0.92

**Step 2: Write BenchmarkRunner**
- Build index with configurable size
- Run queries, measure p50/p95/p99 latency
- Compute recall@1/10/100

**Step 3: Run all tests, commit**

---

### Task 21: Final Cleanup & Release Checklist

**Files:**
- Create: `README.md`
- Create: `BENCHMARKS.md`
- Final pass on all public API documentation

**Step 1: Run full test suite**

```bash
xcodebuild test -scheme MetalANNS -destination 'platform=macOS' 2>&1 | tail -30
```

**Step 2: Verify release criteria**
- [ ] All unit tests pass
- [ ] Recall@10 > 0.92 on random 128-dim 1000-vector dataset
- [ ] Zero memory leaks (Instruments Leaks profile)
- [ ] Thread Sanitizer clean
- [ ] Package has no external dependencies

**Step 3: Write README and BENCHMARKS**

**Step 4: Tag v1.0.0**

```bash
git tag v1.0.0
```

---

## Task Dependency Graph

```
Task 1 (scaffold)
  ├── Task 2 (errors + config)
  ├── Task 3 (backend protocol)
  │     ├── Task 4 (Accelerate distances)
  │     └── Task 5 (MetalContext + pipeline cache)
  │           └── Task 6 (Metal distance shaders)
  ├── Task 7 (VectorBuffer)
  ├── Task 8 (GraphBuffer)
  └── Task 9 (MetadataBuffer + IDMap)

Task 4 + Task 7 + Task 8
  └── Task 10 (CPU NN-Descent)

Task 6 + Task 7 + Task 8
  ├── Task 11 (GPU random init + initial distances)
  └── Task 12 (GPU reverse edges + local join)
        └── Task 13 (bitonic sort)

Task 10 + Task 13
  └── Task 14 (CPU beam search)
        └── Task 15 (GPU beam search)

Task 15
  ├── Task 16 (persistence)
  ├── Task 17 (incremental insert)
  └── Task 18 (soft deletion)

Tasks 16-18
  └── Task 19 (public API actor)
        └── Task 20 (integration test + benchmark)
              └── Task 21 (README + release)
```

## Estimated Task Count: 21 tasks across 6 phases

**Phase 1 (Foundation):** Tasks 1-6
**Phase 2 (Data Structures):** Tasks 7-9
**Phase 3 (Construction):** Tasks 10-13
**Phase 4 (Search):** Tasks 14-15
**Phase 5 (Persistence):** Tasks 16-18
**Phase 6 (Public API):** Tasks 19-21
