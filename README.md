# swiftgrad

A small autograd engine based on Karpathy's [repo](https://github.com/karpathy/micrograd) written in Swift. Same approach, the neuron, layer, and MLP classes were implemented.

This Swift project provides a basic framework for automatic differentiation, a key technique used in machine learning and optimization to compute gradients. The core class, `Value`, represents nodes in a computational graph, encapsulating both a data value and its gradient (derivative). This framework supports both unary and binary operations, allowing for the construction and backpropagation through complex computational graphs.

### Key Components:
- **Autograd:** The `Value` class tracks operations applied to its instances, automatically building a computational graph. Autograd computes gradients by backpropagating through this graph using the chain rule, making it essential for gradient-based optimization tasks.
- **Unary Operations:** Unary operations like `tanh()` and `exp()` are supported through protocol conformance, operating on a single operand.
- **Binary Operations:** Overloaded operators such as `+`, `-`, `*`, and `/` handle element-wise operations between `Value` instances, enabling the construction of complex mathematical expressions.

### Autograd and the Backward Method:
- **Autograd:** Automatically computes gradients of functions by building and traversing a computational graph. This is especially useful in training neural networks where gradients are required for optimizing model parameters.
- **Backward Method:** Once a forward pass is completed, the `backward` method is called on the output node. This initiates backpropagation, which traverses the computational graph in reverse, calculating and accumulating gradients for each node with respect to the final output.

---

>This project is designed for Mac users who want to experiment with automatic differentiation in Swift without relying on external libraries like PyTorch. Swift's simplicity and performance make it an ideal language for this purpose. The ultimate goal is to develop a lightweight and efficient alternative to Tinygrad, specifically tailored for Swift users on Mac. If you're interested in contributing, feel free to reach out to me on Twitter.

# Installation
`swift package clean`

`swift build`

`swift run`

# Example

```swift
func main() {
    let xs = [
        [Value(2.0), Value(3.0), Value(-1.0)],
        [Value(3.0), Value(-1.0), Value(0.5)],
        [Value(0.5), Value(1.0), Value(1.0)],
        [Value(1.0), Value(1.0), Value(-1.0)]
    ]
    let ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)]

    let model = MLP(inputs: 3, outputs: [4, 4, 1])

    let numEpochs = 20
    let lr = 0.05

    for k in 0..<numEpochs {
        let yPred = xs.map { model.forward($0) }
        let loss = yPred
            .enumerated()
            .reduce(Value(0.0)) { (sum, elem) in
                let (idx, pred) = elem
                let diff = pred[0] - ys[idx]
                return sum + (diff * diff)
            }

        model.zeroGrad()
        loss.backward()

        for p in model.parameters {
            p.data += -lr * p.grad
        }

        print(k, loss.data)
    }
}
```

# Further work
- Error handling
- Documentation

# Features to add
- Threading / Acceleration / SIMD
- Resolving the accessing issues (private/public) with Neuron/Layer/MLP
- Tensors, tensoric operations (instead of Value)
- Boltzmann machines
- LLMs ?
