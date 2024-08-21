import Foundation

extension Value: UnaryOperations {

    /// Applies the exponential function to a `Value` and returns a new `Value`.
    ///
    /// - Returns: A new `Value` representing the result of `exp`.
    func exp() -> Value {
        let out = Value(Foundation.exp(self.data), _children: [self], _op: "exp")
        out._backward = {
            self.grad += out.data * out.grad
        }
        return out
    }

    /// Applies the hyperbolic tangent function to a `Value` and returns a new `Value`.
    ///
    /// - Returns: A new `Value` representing the result of `tanh`.
    func tanh() -> Value {
        let t = Foundation.tanh(self.data)
        let out = Value(t, _children: [self], _op: "tanh")
        out._backward = { [weak self] in
            self?.grad += (1 - t * t) * out.grad
        }
        return out
    }
}
