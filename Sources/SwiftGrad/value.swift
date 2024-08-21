import Foundation

infix operator **: MultiplicationPrecedence

/// Represents a value in the computational graph, with support for automatic differentiation.
final class Value: CustomStringConvertible, UnaryOperations {
    var data: Double
    var grad: Double = 0.0
    var _backward: () -> () = {}
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

    func exp() -> Value {
        let out = Value(Foundation.exp(data), _children: [self], _op: "exp")

        out._backward = {
            self.grad += out.data * out.grad
        }
        return out
    }

    func tanh() -> Value {
        let t = Foundation.tanh(data)
        let out = Value(t, _children: [self], _op: "tanh")
        out._backward = { [weak self] in
            guard let self = self else { return }
            self.grad += (1 - t * t) * out.grad
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
