# ZQL-06 | Quantum AI and Machine Learning

## Overview
Quantum Machine Learning (QML) explores the intersection of quantum computing and artificial intelligence. By leveraging the exponentially large Hilbert space of quantum systems, QML aims to process complex data and train models more efficiently than classical counterparts.

## Zuntenium QML Roadmap

The Zuntenium Quantum Consortium is actively developing the `zuntenium.qml` module. Below are the key architectures currently under development.

### 1. Quantum Support Vector Machines (QSVM)
QSVMs use quantum kernels to map classical data into a high-dimensional quantum feature space. 
*Status:* **Planned (Q3 2026)**
* The `QuantumKernel` class will allow users to define feature maps using parameterized circuits (like `ZZFeatureMap`).

### 2. Quantum Neural Networks (QNN)
QNNs are parameterized quantum circuits trained via gradient descent (using the Parameter Shift Rule) to act as classifiers or generative models.
*Status:* **Planned (Q4 2026)**
* Zuntenium will provide PyTorch bindings (`zuntenium.pytorch.TorchConnector`) allowing backpropagation natively through quantum circuits running on the MHOS infrastructure.

### 3. Zuntenium AI-Engine
Currently, Zuntenium uses classical AI *internally* to optimize its transpiler via Reinforcement Learning. Soon, this AI-Engine will be exposed to users to automatically discover optimal ansätze for VQE workloads.

## Getting Prepared
Until the native `zuntenium.qml` package is released, you can construct basic QNNs manually using the VQE hybrid loops learned in [ZQL-05 | Variational and Hybrid](ZQL-05-variational-hybrid.md) by defining customized loss Hamiltonians.
