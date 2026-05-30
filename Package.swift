// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MetalTensor",
    platforms: [
        .iOS(.v12)
    ],
    products: [
        .library(
            name: "MetalTensor",
            targets: ["MetalTensor"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/feng1919/MetalImage", branch: "master")
    ],
    targets: [
        .target(
            name: "MetalTensor",
            dependencies: [
                .product(name: "MetalImage", package: "MetalImage")
            ],
            path: "Framework/MetalTensor",
            exclude: [
                "Info.plist"
            ],
            resources: [
                .copy("MNNPreprocess.metal"),
                .copy("SWAP_BR.bin")
            ],
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("."),
                .headerSearchPath("include"),
                .headerSearchPath("SSD")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.iOS])),
                .linkedFramework("CoreGraphics", .when(platforms: [.iOS])),
                .linkedFramework("CoreMedia", .when(platforms: [.iOS])),
                .linkedFramework("Metal", .when(platforms: [.iOS])),
                .linkedFramework("MetalPerformanceShaders", .when(platforms: [.iOS])),
                .linkedFramework("QuartzCore", .when(platforms: [.iOS])),
                .linkedFramework("UIKit", .when(platforms: [.iOS]))
            ]
        ),
        .testTarget(
            name: "MetalTensorTests",
            dependencies: ["MetalTensor"],
            path: "Framework/MetalTensorTests",
            exclude: [
                "Info.plist"
            ]
        )
    ]
)
