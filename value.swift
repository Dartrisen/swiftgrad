import Foundation

protocol ArithmeticOperations {
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

infix operator **: MultiplicationPrecedence

final class Value: ArithmeticOperations, CustomStringConvertible {
    var data: Double
    var grad: Double = 0.0
    var _backward: () -> Void = {}
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
        return "Value(label: \(label), data: \(data), grad: \(grad))"
    }

    static func + (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data + rhs.data, _children: [lhs, rhs], _op: "+")

        out._backward = {
            lhs.grad += 1.0 * out.grad
            rhs.grad += 1.0 * out.grad
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

    static func * (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data * rhs.data, _children: [lhs, rhs], _op: "*")

        out._backward = {
            lhs.grad += rhs.data * out.grad
            rhs.grad += lhs.data * out.grad
        }
        return out
    }

    static func * (lhs: Value, rhs: Double) -> Value {
        return lhs * Value(rhs)
    }

    static func * (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) * rhs
    }

    static func ** (lhs: Value, rhs: Double) -> Value {
        let out = Value(pow(lhs.data, rhs), _children: [lhs], _op: "**\(rhs)")

        out._backward = {
            lhs.grad += rhs * pow(lhs.data, rhs - 1) * out.grad
        }
        return out
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

    func exp() -> Value {
        let out = Value(Darwin.exp(data), _children: [self], _op: "exp")

        out._backward = {
            self.grad += out.data * out.grad
        }
        return out
    }

    func backward() {
        var topo: [Value] = []
        var visited: Set<Value> = []

        func buildTopo(_ v: Value) {
            if !visited.contains(v) {
                visited.insert(v)
                for child in v._prev {
                    buildTopo(child)
                }
                topo.append(v)
            }
        }

        buildTopo(self)
        self.grad = 1.0
        for v in topo.reversed() {
            v._backward()
        }
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

let x2w2 = x2 * w2
x2w2.label = "x2*w2"

let x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"

let n = x1w1x2w2 + b
n.label = "n"

let e = (2 * n).exp()
let o = (e - 1) / (e + 1)
o.label = "o"
o.backward()

// Print results
print(x1w1)
print(x2w2)
print(x1w1x2w2)
print(n)
print(o)
