"""
StatevectorBackend — Primary simulation backend for ZUNTENIUM v0.1.

Uses NumPy for ≤20 qubits and dispatches to the C++ extension above that.
Integrates CircuitExecutor and TranspilerPipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zuntenium.backend.base_backend import AbstractBackend, ValidationResult
from zuntenium.quantum_core.executor.circuit_executor import CircuitExecutor
from zuntenium.transpiler.pipeline import TranspilerPipeline

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.quantum_core.executor.result import ExecutionResult


class StatevectorBackend(AbstractBackend):
    """
    Default v0.1 statevector simulator.

    Uses NumPy for ≤20 qubits, dispatches to C++ above that.
    Transpiler optimization level 2 is applied by default.

    Args:
        precision: "double" (float64) or "single" (float32). Default: "double".
        seed: RNG seed for reproducible sampling. None = random.
        optimization_level: Transpiler optimization level 0–3. Default: 2.
    """

    name = "zuntenium_statevector_simulator"
    is_simulator = True
    max_qubits = 30   # v0.1 practical limit

    def __init__(
        self,
        precision: str = "double",
        seed: int | None = None,
        optimization_level: int = 2,
    ) -> None:
        self._precision = precision
        self._seed = seed
        self._executor = CircuitExecutor(sv_backend="auto", seed=seed)
        self._transpiler = TranspilerPipeline(level=optimization_level)

    def run(
        self,
        circuit: QuantumCircuit | QuantumIR,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        """
        Execute a circuit on the statevector simulator.

        Args:
            circuit: QuantumCircuit or QuantumIR.
            shots: Number of measurement samples.
            **options: override_transpiler (bool), optimization_level (int).

        Returns:
            ExecutionResult with counts and statevector.
        """
        from zuntenium.circuit.circuit import QuantumCircuit

        # Accept either QuantumCircuit or pre-compiled QuantumIR
        if isinstance(circuit, QuantumCircuit):
            ir: QuantumIR = circuit.to_ir()
        else:
            ir = circuit

        # Transpile unless caller bypasses
        if not options.get("skip_transpile", False):
            level = int(options.get("optimization_level", 2))
            if level != self._transpiler.level:
                transpiler = TranspilerPipeline(level=level)
            else:
                transpiler = self._transpiler
            ir = transpiler.run(ir, target=self.target)

        return self._executor.execute(ir, shots=shots)

    def validate_circuit(self, circuit: QuantumCircuit) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if circuit.num_qubits > self.max_qubits:
            errors.append(
                f"Circuit has {circuit.num_qubits} qubits but "
                f"{self.name} supports at most {self.max_qubits}."
            )
        if circuit.num_qubits > 20:
            warnings.append(
                f"Circuit has {circuit.num_qubits} qubits. "
                "NumPy backend will be slow; C++ binding recommended."
            )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @property
    def supported_gates(self) -> list[str]:  # type: ignore[override]
        return [
            "H", "X", "Y", "Z", "S", "T",
            "Rx", "Ry", "Rz", "U1", "U2", "U3",
            "CX", "CZ", "CY", "SWAP", "ISWAP",
            "CRx", "CRy", "CRz", "CCX", "CSWAP",
        ]
