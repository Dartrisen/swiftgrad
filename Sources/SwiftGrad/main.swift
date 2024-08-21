import Foundation

// Define inputs
let x1 = Value(2.0, label: "x1")
let x2 = Value(0.0, label: "x2")

// Define weights
let w1 = Value(-3.0, label: "w1")
let w2 = Value(1.0, label: "w2")

// Define bias
let b = Value(6.8813735870195432, label: "b")

// Perform operations
let x1w1 = x1 * w1
x1w1.label = "x1*w1"

let x2w2 = x2 * w2
x2w2.label = "x2*w2"

let x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"

let n = x1w1x2w2 + b
n.label = "n"

let e = (2 * n).exp()
let o = (e - 1) / (e + 1)
o.label = "o"
o.backward()

// Print results
print(x1w1)
print(x2w2)
print(x1w1x2w2)
print(n)
print(e)
print(o)

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

    let startTime = CFAbsoluteTimeGetCurrent()

    for k in 0..<numEpochs {
        // Forward
        let yPred = xs.map { model.forward($0) }
        let loss = yPred
            .enumerated()
            .reduce(Value(0.0)) { (sum, elem) in
                let (idx, pred) = elem
                let diff = pred[0] - ys[idx]
                return sum + (diff * diff)
            }

        // Backward
        model.zeroGrad()
        loss.backward()

        // Update weights
        for p in model.parameters {
            p.data += -lr * p.grad
        }

        print(k, loss.data)
    }

    let endTime = CFAbsoluteTimeGetCurrent()
    let elapsedTime = endTime - startTime

    print("Total time taken: \(elapsedTime) seconds")
}

main()
