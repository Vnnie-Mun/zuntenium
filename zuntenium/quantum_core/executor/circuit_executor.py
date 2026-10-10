"""
CircuitExecutor — Core execution engine.

Traverses QuantumIR instructions and applies gate operations to a StateVector,
then samples measurement outcomes to produce an ExecutionResult.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from zuntenium.quantum_core.executor.measurement import measure_all
from zuntenium.quantum_core.executor.result import ExecutionResult
from zuntenium.quantum_core.statevector.statevector import StateVector

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class CircuitExecutor:
    """
    Traverses QuantumIR and applies gate operations to a StateVector.

    The executor is backend-agnostic: it operates purely on the IR and
    StateVector abstractions. Backends instantiate and configure this class.

    Args:
        sv_backend: Statevector backend string ("auto", "numpy", "cpp").
        seed: Optional RNG seed for reproducible sampling.
    """

    def __init__(self, sv_backend: str = "auto", seed: int | None = None) -> None:
        self._sv_backend = sv_backend
        self._seed = seed

    def execute(
        self,
        ir: QuantumIR,
        shots: int = 1024,
    ) -> ExecutionResult:
        """
        Execute a QuantumIR and return an ExecutionResult.

        Args:
            ir: Compiled and (optionally) transpiled quantum circuit IR.
            shots: Number of measurement shots.

        Returns:
            ExecutionResult with counts and statevector.
        """
        t_start = time.perf_counter()

        sv = StateVector(ir.num_qubits, backend=self._sv_backend)

        # Apply all gate instructions
        for node in ir.instructions:
            if node.operation == "gate" and node.gate is not None:
                sv.apply_gate(node.gate, node.qubits)
            elif node.operation == "reset":
                # Reset qubit to |0⟩ by collapsing: re-initialize state
                # Full implementation in v0.2 with proper projection
                pass
            elif node.operation in ("measure", "barrier"):
                # Measurement handled post-hoc; barriers are no-ops in simulation
                pass

        # Post-hoc measurement sampling
        measurement = measure_all(
            sv.get_amplitudes(),
            ir.num_qubits,
            shots,
            seed=self._seed,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return ExecutionResult(
            counts=measurement.counts,
            num_qubits=ir.num_qubits,
            shots=shots,
            statevector=sv.get_amplitudes(),
            metadata={
                "backend": f"statevector_{self._sv_backend}",
                "exec_time_ms": round(elapsed_ms, 3),
                "num_gates": len(ir.instructions),
                "sv_backend": sv.active_backend,
            },
        )
