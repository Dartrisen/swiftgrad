import Foundation

infix operator **: MultiplicationPrecedence

/// Represents a value in the computational graph, with support for automatic differentiation.
final class Value: CustomStringConvertible {
    var data: Double
    var grad: Double = 0.0
    var _backward: () -> () = {}
    var _prev: Set<Value>
    var _op: String
    var label: String

    /// Initializes a new `Value` instance.
    ///
    /// - Parameters:
    ///   - data: The scalar value.
    ///   - _children: The values that this `Value` depends on.
    ///   - _op: The operation that produced this value.
    init(_ data: Double, _children: [Value] = [], _op: String = "", label: String = "") {
        self.data = data
        self._prev = Set(_children)
        self._op = _op
        self.label = label
    }

    var description: String {
        return "Value(label: \(label), data: \(data), grad: \(grad))"
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
