import Foundation

/// A protocol for types that support basic binary arithmetic operations.
protocol BinaryOperations {
    static func + (lhs: Self, rhs: Self) -> Self
    static func + (lhs: Self, rhs: Double) -> Self
    static func + (lhs: Double, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Double) -> Self
    static prefix func - (v: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Double, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Double) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Double) -> Self
    static func / (lhs: Double, rhs: Self) -> Self
}
