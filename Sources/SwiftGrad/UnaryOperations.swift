import Foundation

/// A protocol for types that support unary operations, such as exponentiation.
protocol UnaryOperations {
    func tanh() -> Self
    func exp() -> Value
}
