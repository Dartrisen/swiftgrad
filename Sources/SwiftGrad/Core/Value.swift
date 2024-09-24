import Foundation

/// Represents a value in the computational graph, with support for automatic differentiation.
final class Value {
    var data: Double
    var grad: Double = 0.0
    var _backward: () -> () = {}
    var _prev: Set<Value>
    var _op: String
    var label: String
    var cachedTopo: [Value]? = nil

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

    func backward() {
        var topo = cachedTopo ?? []
        if topo.isEmpty {
            topo = topoSortIterative(self)
            cachedTopo = topo
        }
        self.grad = 1.0
        for v in topo.reversed() {
            v._backward()
        }
    }
}
