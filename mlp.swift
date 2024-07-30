import Foundation

class MLP: Module {
    var layers: [Layer]

    init(inputs: Int, outputs: [Int]) {
        let sz = [inputs] + outputs
        self.layers = (0..<outputs.count).map { i in
            Layer(inputSize: sz[i], outputSize: sz[i+1])
        }
    }

    func forward(_ x: [Value]) -> [Value] {
        var output = x
        for layer in layers {
            output = layer.forward(output)
        }
        return output
    }

    override var parameters: [Value] {
        return layers.flatMap { $0.parameters }
    }
}