"""
zuntenium/quantum_core/statevector/sim_circuit.py
=================================================
Low-level simulation circuit: fluent builder + internal instruction type.

Extracted from statevector_simulator.py as part of O6 God-class decomposition.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from zuntenium.quantum_core.statevector.gate_library import GateLibrary

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MatrixNxN = np.ndarray    # shape (N, N),   dtype complex128


class _SimInstruction:
    """Immutable record of a single gate operation in a simulation circuit.

    This is the simulator-internal instruction type. The public API uses
    ``zuntenium.circuit.instruction.Instruction`` instead.
    """
    __slots__ = ('matrix', 'qubits', 'name')

    def __init__(self, matrix: MatrixNxN, qubits: Sequence[int], name: str = "U") -> None:
        self.matrix = matrix
        self.qubits = tuple(qubits)
        self.name   = name

    def __repr__(self) -> str:
        return f"_SimInstruction({self.name}, qubits={self.qubits})"


class SimCircuit:
    """
    Low-level simulation circuit: fluent builder that operates directly on
    gate matrices and a ``StatevectorSimulator``.

    .. note::
        This is the *simulator-internal* circuit class. The public API
        circuit class is ``zuntenium.circuit.circuit.QuantumCircuit``.

    Each gate method appends a ``_SimInstruction`` and returns self for chaining.

    Example
    -------
    >>> qc = SimCircuit(2)
    >>> qc.h(0).cx(0, 1)   # Bell state
    >>> qc.run(simulator)
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._instructions: List[_SimInstruction] = []

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def instructions(self) -> List[_SimInstruction]:
        return self._instructions

    @property
    def depth(self) -> int:
        """Circuit depth: longest path through the circuit DAG (simplified)."""
        if not self._instructions:
            return 0
        layers: List[set] = []
        for instr in self._instructions:
            placed = False
            for layer in reversed(layers):
                if any(q in layer for q in instr.qubits):
                    idx = layers.index(layer)
                    if idx + 1 < len(layers):
                        layers[idx + 1].update(instr.qubits)
                    else:
                        layers.append(set(instr.qubits))
                    placed = True
                    break
            if not placed:
                if layers:
                    layers[0].update(instr.qubits)
                else:
                    layers.append(set(instr.qubits))
        return len(layers)

    # ------------------------------------------------------------------
    # Fluent gate API
    # ------------------------------------------------------------------

    def _add(self, name: str, qubits: Sequence[int], **params) -> "SimCircuit":
        matrix = GateLibrary.get(name, **params)
        self._instructions.append(_SimInstruction(matrix, qubits, name))
        return self

    def x(self, qubit: int) -> "SimCircuit":
        return self._add('X', [qubit])

    def y(self, qubit: int) -> "SimCircuit":
        return self._add('Y', [qubit])

    def z(self, qubit: int) -> "SimCircuit":
        return self._add('Z', [qubit])

    def h(self, qubit: int) -> "SimCircuit":
        return self._add('H', [qubit])

    def s(self, qubit: int) -> "SimCircuit":
        return self._add('S', [qubit])

    def t(self, qubit: int) -> "SimCircuit":
        return self._add('T', [qubit])

    def sdg(self, qubit: int) -> "SimCircuit":
        return self._add('SDG', [qubit])

    def tdg(self, qubit: int) -> "SimCircuit":
        return self._add('TDG', [qubit])

    def rx(self, theta: float, qubit: int) -> "SimCircuit":
        return self._add('RX', [qubit], theta=theta)

    def ry(self, theta: float, qubit: int) -> "SimCircuit":
        return self._add('RY', [qubit], theta=theta)

    def rz(self, theta: float, qubit: int) -> "SimCircuit":
        return self._add('RZ', [qubit], theta=theta)

    def phase(self, phi: float, qubit: int) -> "SimCircuit":
        return self._add('PHASE', [qubit], phi=phi)

    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> "SimCircuit":
        return self._add('U3', [qubit], theta=theta, phi=phi, lam=lam)

    def cx(self, control: int, target: int) -> "SimCircuit":
        return self._add('CNOT', [control, target])

    def cz(self, control: int, target: int) -> "SimCircuit":
        return self._add('CZ', [control, target])

    def swap(self, q0: int, q1: int) -> "SimCircuit":
        return self._add('SWAP', [q0, q1])

    def iswap(self, q0: int, q1: int) -> "SimCircuit":
        return self._add('ISWAP', [q0, q1])

    def ccx(self, c0: int, c1: int, target: int) -> "SimCircuit":
        return self._add('TOFFOLI', [c0, c1, target])

    def append_gate(self, matrix: MatrixNxN, qubits: Sequence[int], name: str = "Custom") -> "SimCircuit":
        """
        Append an arbitrary unitary gate to the circuit.

        Parameters
        ----------
        matrix : unitary matrix (2^k × 2^k for a k-qubit gate)
        qubits : target qubit indices (length k)
        name   : display name for circuit drawing
        """
        k = len(qubits)
        expected_dim = 1 << k
        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match "
                f"{k} qubits (expected ({expected_dim},{expected_dim}))."
            )
        self._instructions.append(_SimInstruction(matrix.astype(np.complex128), qubits, name))
        return self

    def barrier(self, *qubits: int) -> "SimCircuit":
        """
        No-op annotation marker (for visualization / transpiler hints).
        Does not affect execution.
        """
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        simulator: "StatevectorSimulator",
        reset: bool = True,
    ) -> "StatevectorSimulator":
        """
        Execute this circuit on the given simulator.

        Parameters
        ----------
        simulator : StatevectorSimulator (or any SimulatorBackend)
        reset     : if True, reset to |0...0⟩ before running

        Returns
        -------
        The simulator (for chaining: qc.run(sim).sample(1000))
        """
        if simulator.num_qubits != self._num_qubits:
            raise ValueError(
                f"Circuit has {self._num_qubits} qubits but simulator "
                f"has {simulator.num_qubits} qubits."
            )
        if reset:
            simulator.initialize(self._num_qubits)
        for instr in self._instructions:
            simulator.apply_gate(instr.matrix, instr.qubits)
        return simulator

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def draw(self) -> str:
        """Minimal ASCII circuit diagram."""
        lines = [f"q{i}: " for i in range(self._num_qubits)]
        for instr in self._instructions:
            label = instr.name[:4].center(6)
            for i in range(self._num_qubits):
                if i in instr.qubits:
                    lines[i] += f"─[{label}]─"
                else:
                    lines[i] += "──────────"
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SimCircuit({self._num_qubits} qubits, "
            f"{len(self._instructions)} gates, depth={self.depth})")
