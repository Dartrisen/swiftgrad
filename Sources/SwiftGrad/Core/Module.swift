import Foundation

class Module {
    var parameters: [Value] {
        return []
    }

    func zeroGrad() {
        for p in parameters {
            p.grad = 0.0
        }
    }
}
