# swiftgrad

A small autograd engine based on Karpathy's [repo](https://github.com/karpathy/micrograd) written in Swift. Same approach, the neuron, layer, and MLP classes were implemented.

# Installation

`swiftc -o main value.swift module.swift neuron.swift layer.swift mlp.swift main.swift`
`./main`

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