import Foundation

enum ValueError: Error {
    case divisionByZero
    case invalidOperation(String)
}

infix operator **: MultiplicationPrecedence

extension Value: BinaryOperations {

    static func + (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data + rhs.data, _children: [lhs, rhs], _op: "+")
        out._backward = { [weak lhs, weak rhs] in
            lhs?.grad += out.grad
            rhs?.grad += out.grad
        }
        return out
    }

    static func * (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data * rhs.data, _children: [lhs, rhs], _op: "*")

        out._backward = { [weak lhs, weak rhs] in
            lhs?.grad += (rhs?.data ?? 0) * out.grad
            rhs?.grad += (lhs?.data ?? 0) * out.grad
        }
        return out
    }

    static func ** (lhs: Value, rhs: Double) -> Value {
        let out = Value(pow(lhs.data, rhs), _children: [lhs], _op: "**\(rhs)")

        out._backward = {
            lhs.grad += rhs * pow(lhs.data, rhs - 1) * out.grad
        }
        return out
    }

    static func + (lhs: Value, rhs: Double) -> Value {
        return lhs + Value(rhs)
    }

    static func + (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) + rhs
    }

    static prefix func - (v: Value) -> Value {
        return v * -1.0
    }

    static func - (lhs: Value, rhs: Value) -> Value {
        return lhs + (-rhs)
    }

    static func - (lhs: Value, rhs: Double) -> Value {
        return lhs - Value(rhs)
    }

    static func - (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) - rhs
    }

    static func * (lhs: Value, rhs: Double) -> Value {
        return lhs * Value(rhs)
    }

    static func * (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) * rhs
    }

    static func / (lhs: Value, rhs: Value) -> Value {
        return lhs * (rhs ** -1)
    }

    static func / (lhs: Value, rhs: Double) -> Value {
        return lhs / Value(rhs)
    }

    static func / (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) / rhs
    }
}
