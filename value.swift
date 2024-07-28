import Foundation

protocol ArithmeticOperations {
    static func + (lhs: Self, rhs: Self) -> Self
}

final class Value: ArithmeticOperations, CustomStringConvertible {
    var data: Double

    init(_ data: Double) {
        self.data = data
    }

    var description: String {
        return "Value(data: \(data))"
    }

    static func + (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data + rhs.data)
        return out
    }
}

// Define inputs
let x1 = Value(2.0)
let x2 = Value(0.0)

print(x1 + x2)
