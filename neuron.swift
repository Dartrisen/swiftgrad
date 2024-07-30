import Foundation

class Neuron: Module {
    var weights: [Value]
    var bias: Value

    init(inputSize: Int) {
        self.weights = (0..<inputSize).map { _ in Value(Double.random(in: -1...1)) }
        self.bias = Value(Double.random(in: -1...1))
    }

    func forward(_ inputs: [Value]) -> Value {
        var sum = Value(0.0)
        for (input, weight) in zip(inputs, weights) {
            sum = sum + (input * weight)
        }
        sum = sum + bias
        let out = sum.tanh()
        return out
    }

    override var parameters: [Value] {
        return weights + [bias]
    }
}
