import Foundation

protocol ArithmeticOperations {
    static func + (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
}

final class Value: ArithmeticOperations, CustomStringConvertible {
    var data: Double
    var _prev: Set<Value>
    var _op: String
    var label: String

    init(_ data: Double, _children: [Value] = [], _op: String = "", label: String = "") {
        self.data = data
        self._prev = Set(_children)
        self._op = _op
        self.label = label
    }

    var description: String {
        return "Value(label: \(label), data: \(data))"
    }

    static func + (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data + rhs.data, _children: [lhs, rhs], _op: "+")
        return out
    }

    static func * (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data * rhs.data, _children: [lhs, rhs], _op: "*")
        return out
    }
}

extension Value: Hashable {
    static func == (lhs: Value, rhs: Value) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

// Define inputs
let x1 = Value(2.0, label: "x1")
let x2 = Value(0.0, label: "x2")

// Define weights
let w1 = Value(-3.0, label: "w1")
let w2 = Value(1.0, label: "w2")

// Define bias
let b = Value(6.8813735870195432, label: "b")

// Perform operations
let x1w1 = x1 * w1
x1w1.label = "x1*w1"

// Print results
print(x1w1)
