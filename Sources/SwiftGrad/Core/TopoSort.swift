//
//  TopoSort.swift
//  SwiftGrad
//
//  Created by dartrisen on 24.09.2024.
//
import Foundation

func topoSortIterative(_ start: Value) -> [Value] {
    var topo: [Value] = []
    var visited: Set<Value> = []
    var stack: [Value] = [start]

    while let node = stack.popLast() {
        if visited.contains(node) {
            topo.append(node)
            continue
        }
        
        visited.insert(node)
        stack.append(node) // Re-push the node to add it to topo on the second visit
        
        for child in node._prev {
            if !visited.contains(child) {
                stack.append(child)
            }
        }
    }
    return topo
}
