# CHANGELOG

## 1.0.1 (XX / XX / 2024)

#### Improvements

- Separated `BinaryOperations` and `UnaryOperations` protocols from the `Value` class implementation for better modularity.
- Restructured the project for improved organization and maintainability.
- Added unit tests for the `Value`, `Neuron`, and `Layer` classes to ensure reliability.
- Optimized `topoSort` function with a more efficient iterative graph sorting algorithm.

## 1.0.0 (8 / 20 / 2024)

#### Features
- Introduced `Value`, `Module`, `Neuron`, `Layer`, and `MLP` classes.
- **Binary operations**: Implemented addition (+), subtraction (-), multiplication (*), and division (/).
- **Unary operations**: Added support for tanh and exp functions.
