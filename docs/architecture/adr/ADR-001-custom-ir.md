# ADR-001: Use Custom QuantumIR Instead of Qiskit DAGCircuit

**Date:** 2025-01-01
**Status:** Accepted
**Deciders:** ZUNTENIUM core team

## Context

We need an intermediate representation (IR) for quantum circuits that serves as the contract between the circuit builder, transpiler, and backends.

## Decision

We implement a custom `QuantumIR` dataclass rather than depending on Qiskit's `DAGCircuit`.

## Consequences

**Positive:**
- Zero dependency on Qiskit — no version lock, no import overhead
- Full control over shape, serialization format, and ML embedding hooks
- IR can embed training data annotations for the v0.4 AI transpiler
- Lightweight: a simple list of `IRNode` dataclasses, JSON-serializable

**Negative:**
- Must maintain our own QASM 2/3 parser
- Cannot directly consume Qiskit circuits without a translation shim (planned for v0.2)

## Alternatives Considered

- **Qiskit DAGCircuit**: Too coupled to Qiskit's gate library; heavy import time
- **OpenQASM string as IR**: Not structured enough for pass-based transforms
