import XCTest
@testable import SwiftGrad

func XCTAssertAlmostEqual(_ a: Double, _ b: Double, tolerance: Double = 1e-6, file: StaticString = #file, line: UInt = #line) {
    XCTAssertTrue(abs(a - b) < tolerance, "Expected \(a) to be approximately equal to \(b)", file: file, line: line)
}

class ValueTests: XCTestCase {

    // Test initialization of a Value
    func testInitialization() {
        let v = Value(3.14, label: "x")
        
        // Check that data is initialized correctly
        XCTAssertEqual(v.data, 3.14, "Data should be initialized to 3.14")
        
        // Check that grad is initialized to zero
        XCTAssertEqual(v.grad, 0.0, "Gradient should be initialized to 0.0")
        
        // Check that label is initialized correctly
        XCTAssertEqual(v.label, "x", "Label should be 'x'")
        
        // Check that _op and _prev are empty by default
        XCTAssertEqual(v._op, "", "Operation should be an empty string by default")
        XCTAssertTrue(v._prev.isEmpty, "Previous values set should be empty by default")
    }

    // Test the backward method (simple case where there are no dependencies)
    func testBackwardNoDependencies() {
        let v = Value(3.14)
        
        // Manually set the gradient to check if it's updated during backward pass
        v.backward()
        
        // Check that the gradient is set to 1 after the backward call
        XCTAssertEqual(v.grad, 1.0, "Gradient should be 1.0 after calling backward")
    }

    // Test backward with a simple dependency
    func testBackwardWithDependencies() {
        let x = Value(2.0)
        let y = Value(3.0)
        let z = Value(x.data * y.data, _children: [x, y], _op: "*")
        
        // Set backward logic for z (partial derivatives)
        z._backward = {
            x.grad += y.data * z.grad
            y.grad += x.data * z.grad
        }
        
        // Perform backward pass from z
        z.backward()

        // Check if gradients were correctly propagated
        XCTAssertEqual(x.grad, 3.0, "Gradient of x should be 3 (dz/dx = y)")
        XCTAssertEqual(y.grad, 2.0, "Gradient of y should be 2 (dz/dy = x)")
    }

    // Test the computational graph and backward pass
    func testBackwardComplexGraph() {
        let a = Value(2.0)
        let b = Value(3.0)
        let c = Value(a.data + b.data, _children: [a, b], _op: "+")
        let d = Value(c.data * 4.0, _children: [c], _op: "*")
        
        // Define backward logic for c and d
        c._backward = {
            a.grad += 1.0 * c.grad
            b.grad += 1.0 * c.grad
        }
        d._backward = {
            c.grad += 4.0 * d.grad
        }

        // Perform the backward pass starting from d
        d.backward()

        // Check gradients
        XCTAssertEqual(a.grad, 4.0, "Gradient of a should be 4 (from d)")
        XCTAssertEqual(b.grad, 4.0, "Gradient of b should be 4 (from d)")
    }
}

class NeuronTests: XCTestCase {

    func testInitialization() {
        let inputSize = 5
        let neuron = Neuron(inputSize: inputSize)
        
        // Check that weights are initialized correctly
        XCTAssertEqual(neuron.weights.count, inputSize, "Weights count should match input size.")
        XCTAssertEqual(neuron.parameters.count, inputSize + 1, "Parameters count should include weights and bias.")
        
        // Check that bias is initialized
        XCTAssertNotNil(neuron.bias, "Bias should be initialized.")
    }

    func testForward() {
        let inputSize = 5
        let neuron = Neuron(inputSize: inputSize)
        
        // Create some test inputs
        let inputs = (0..<inputSize).map { _ in Value(Double.random(in: -1...1)) }
        
        // Call the forward method
        let output = neuron.forward(inputs)

        // Check that the output is a valid Value (assuming Value can be checked)
        XCTAssertNotNil(output, "Output should not be nil.")
        
        // Check that the output is within the expected range for tanh
        XCTAssertTrue(output.data >= -1.0 && output.data <= 1.0, "Output should be in the range of -1 to 1.")
    }
}

class LayerTests: XCTestCase {

    // Test initialization of a Layer
    func testInitialization() {
        let inputSize = 3
        let outputSize = 2
        let layer = Layer(inputSize: inputSize, outputSize: outputSize)
        
        // Check that the correct number of neurons is initialized
        XCTAssertEqual(layer.neurons.count, outputSize, "Layer should have \(outputSize) neurons")
        
        // Check that each neuron has the correct input size
        for neuron in layer.neurons {
            XCTAssertEqual(neuron.weights.count, inputSize, "Each neuron should have \(inputSize) weights")
        }
    }

    // Test the forward method (simple forward pass)
    func testForward() {
        let inputSize = 3
        let outputSize = 2
        let layer = Layer(inputSize: inputSize, outputSize: outputSize)
        
        // Create test inputs
        let inputs = (0..<inputSize).map { _ in Value(Double.random(in: -1...1)) }
        
        // Perform forward pass through the layer
        let outputs = layer.forward(inputs)
        
        // Check that the number of outputs matches the number of neurons (output size)
        XCTAssertEqual(outputs.count, outputSize, "The number of outputs should match the number of neurons (output size).")
        
        // Check that each output is a valid Value
        for output in outputs {
            XCTAssertNotNil(output.data, "Output should not be nil")
        }
    }

    // Test that parameters method correctly aggregates all neuron parameters
    func testParameters() {
        let inputSize = 3
        let outputSize = 2
        let layer = Layer(inputSize: inputSize, outputSize: outputSize)
        
        // Collect all parameters from the layer
        let parameters = layer.parameters
        
        // Check that the number of parameters is correct
        let expectedParameterCount = (inputSize + 1) * outputSize // each neuron has `inputSize` weights + 1 bias
        XCTAssertEqual(parameters.count, expectedParameterCount, "The number of parameters should be \(expectedParameterCount)")
        
        // Check that none of the parameters are nil
        for param in parameters {
            XCTAssertNotNil(param.data, "Parameter should not be nil")
        }
    }
}
