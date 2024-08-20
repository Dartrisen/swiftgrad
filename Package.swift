// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swiftgrad",
    targets: [
        .executableTarget(
            name: "SwiftGrad",
            dependencies: []),
        .testTarget(
            name: "SwiftGradTests",
            dependencies: ["SwiftGrad"]),
    ]
)
