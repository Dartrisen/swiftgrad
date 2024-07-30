import Foundation

class Layer: Module {
    var neurons: [Neuron]

    init(inputSize: Int, outputSize: Int) {
        self.neurons = (0..<outputSize).map { _ in Neuron(inputSize: inputSize) }
    }

    func forward(_ inputs: [Value]) -> [Value] {
        return neurons.map { $0.forward(inputs) }
    }

    override var parameters: [Value] {
        return neurons.flatMap { $0.parameters }
    }
}
