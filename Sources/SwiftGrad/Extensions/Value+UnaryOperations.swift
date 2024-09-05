import Foundation

extension Value: UnaryOperations {

    /// Applies the exponential function to a `Value` and returns a new `Value`.
    ///
    /// - Returns: A new `Value` representing the result of `exp`.
    func exp() -> Value {
        let out = Value(Foundation.exp(self.data), _children: [self], _op: "exp")
        /// d/dx exp(x) = exp(x) = out.data
        out._backward = {
            self.grad += out.data * out.grad
        }
        return out
    }
    
    /// Applies the logarithmic function to a `Value` and returns a new `Value`.
    ///
    /// - Returns: A new `Value` representing the result of `log`.
    func log() -> Value {
        let out = Value(Foundation.log(self.data), _children: [self], _op: "log")
        /// d/dx log(x) = 1/x = 1/self.data
        out._backward = {
            self.grad += out.grad / self.data
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
