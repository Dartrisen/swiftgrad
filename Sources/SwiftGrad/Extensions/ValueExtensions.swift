//
//  ValueExtensions.swift
//  SwiftGrad
//
//  Created by dartrisen on 24.09.2024.
//
import Foundation

extension Value: Hashable {
    static func == (lhs: Value, rhs: Value) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

extension Value: CustomStringConvertible {
    var description: String {
        return "Value(label: \(label), data: \(data), grad: \(grad))"
    }
}
