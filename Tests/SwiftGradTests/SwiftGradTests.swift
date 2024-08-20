import XCTest

func XCTAssertAlmostEqual(_ a: Double, _ b: Double, tolerance: Double = 1e-6, file: StaticString = #file, line: UInt = #line) {
    XCTAssertTrue(abs(a - b) < tolerance, "Expected \(a) to be approximately equal to \(b)", file: file, line: line)
}

final class NeuronLayerTests: XCTestCase {

    // Test for the Neuron class
    func testNeuron() {
        let inputs = [Value(1.0), Value(-2.0)]
        let neuron = Neuron(inputSize: inputs.count)

        let output = neuron.forward(inputs)
        output.grad = 1.0
        output._backward()

        for (i, input) in inputs.enumerated() {
            print("Gradient of input[\(i)]: \(input.grad)")
            XCTAssertNotEqual(input.grad, 0.0, "Gradient of input[\(i)] should not be zero")
        }
        for (i, weight) in neuron.weights.enumerated() {
            print("Gradient of weight[\(i)]: \(weight.grad)")
            XCTAssertNotEqual(weight.grad, 0.0, "Gradient of weight[\(i)] should not be zero")
        }
        print("Gradient of bias: \(neuron.bias.grad)")
        XCTAssertNotEqual(neuron.bias.grad, 0.0, "Gradient of bias should not be zero")

        print("Neuron test passed.")
    }

    // Test for the Layer class
    func testLayer() {
        let inputs = [Value(1.0), Value(-2.0)]
        let layer = Layer(inputSize: inputs.count, outputSize: 3)

        let outputs = layer.forward(inputs)
        for output in outputs {
            output.grad = 1.0
        }
        for output in outputs {
            output._backward()
        }

        for (i, input) in inputs.enumerated() {
            print("Gradient of input[\(i)]: \(input.grad)")
            XCTAssertNotEqual(input.grad, 0.0, "Gradient of input[\(i)] should not be zero")
        }
        for neuron in layer.neurons {
            for (i, weight) in neuron.weights.enumerated() {
                print("Gradient of neuron weight[\(i)]: \(weight.grad)")
                XCTAssertNotEqual(weight.grad, 0.0, "Gradient of neuron weight[\(i)] should not be zero")
            }
            print("Gradient of neuron bias: \(neuron.bias.grad)")
            XCTAssertNotEqual(neuron.bias.grad, 0.0, "Gradient of neuron bias should not be zero")
        }

        print("Layer test passed.")
    }
}

NeuronLayerTests.defaultTestSuite.run()
