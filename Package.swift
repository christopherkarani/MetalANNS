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
            dependencies: ["MetalANNS", "MetalANNSCore", "MetalANNSBenchmarks"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "MetalANNSBenchmarks",
            dependencies: ["MetalANNS", "MetalANNSCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
