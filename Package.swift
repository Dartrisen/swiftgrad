// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftGrad",
    platforms: [
        .macOS(.v10_15)
    ],
    targets: [
        .executableTarget(
            name: "SwiftGrad",
            dependencies: []),
        .testTarget(
            name: "SwiftGradTests",
            dependencies: ["SwiftGrad"],
            path: "Tests/SwiftGradTests"),
    ]
)
