"""ZUNTENIUM quantum execution integration layer."""

from __future__ import annotations

import contextlib
import io
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from zuntenium.quantum_core.statevector import StatevectorSimulator
from zuntenium.transpiler.v2 import Circuit, Gate, OptimizationReport, Transpiler
from zuntenium.transpiler.v2.engine import _build_random


OPTIMIZATION_LEVEL_PASSES: Dict[int, List[str]] = {
    0: [],
    1: ["cancel", "rotation", "fusion"],
    2: ["cancel", "rotation", "commute", "fusion", "depth"],
    3: ["cancel", "rotation", "commute", "fusion", "depth", "dead"],
}
_DEFAULT_LEVEL = 2


def _gate_matrix(gate: Gate) -> np.ndarray:
    """Return the matrix for a v2 IR gate."""
    if gate.name == "ECR":
        return np.array(
            [[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]],
            dtype=np.complex128,
        ) / math.sqrt(2)
    if gate.name == "XX":
        theta = math.pi / 4 if not gate.params else gate.params[0]
        cosine, sine = math.cos(theta), math.sin(theta)
        return np.array(
            [[cosine, 0, 0, -1j * sine], [0, cosine, -1j * sine, 0],
             [0, -1j * sine, cosine, 0], [-1j * sine, 0, 0, cosine]],
            dtype=np.complex128,
        )
    if gate.name == "CSWAP":
        matrix = np.eye(8, dtype=np.complex128)
        matrix[5, 5] = matrix[6, 6] = 0
        matrix[5, 6] = matrix[6, 5] = 1
        return matrix
    matrix = gate.matrix()
    if gate.name in {"CCX", "CCNOT", "TOFFOLI"}:
        matrix = np.eye(8, dtype=np.complex128)
        matrix[6, 6] = matrix[7, 7] = 0
        matrix[6, 7] = matrix[7, 6] = 1
    return matrix


@dataclass
class ExecutionResult:
    statevector: np.ndarray
    probabilities: np.ndarray
    counts: Dict[str, int]
    num_qubits: int
    shots: int
    opt_report: Optional[OptimizationReport]
    execution_time_ms: float
    total_time_ms: float
    circuit_depth: int
    num_gates_executed: int
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def expectation_z(self, qubit: int) -> float:
        if not 0 <= qubit < self.num_qubits:
            raise IndexError(f"qubit must be in 0..{self.num_qubits - 1}")
        indices = np.arange(len(self.probabilities))
        signs = 1 - 2 * ((indices >> qubit) & 1)
        return float(np.dot(signs.astype(np.float64), self.probabilities))

    def all_z_expectations(self) -> np.ndarray:
        return np.array([self.expectation_z(q) for q in range(self.num_qubits)])

    def summary(self) -> str:
        lines = [
            f"ExecutionResult(n={self.num_qubits}, shots={self.shots})",
            f"  Gates executed : {self.num_gates_executed}",
            f"  Circuit depth  : {self.circuit_depth}",
            f"  Exec time      : {self.execution_time_ms:.2f}ms",
            f"  Total time     : {self.total_time_ms:.2f}ms",
        ]
        if self.opt_report:
            report = self.opt_report
            lines.append(
                f"  Optimization   : {report.original_gates}->{report.optimized_gates} gates "
                f"({report.gate_reduction_pct:.1f}%down), "
                f"{report.original_depth}->{report.optimized_depth} depth "
                f"({report.depth_reduction_pct:.1f}%down)"
            )
        return "\n".join(lines)


class CompatibilityError(ValueError):
    """Raised when a circuit is incompatible with the simulator."""


def validate_circuit_compatibility(circuit: Circuit, max_qubits: int = 20) -> None:
    if circuit.num_qubits < 1:
        raise CompatibilityError("Circuit must contain at least 1 qubit.")
    if circuit.num_qubits > max_qubits:
        raise CompatibilityError(
            f"Circuit has {circuit.num_qubits} qubits which exceeds simulator limit "
            f"{max_qubits}."
        )
    supported = {
        "X", "Y", "Z", "H", "S", "T", "SDG", "TDG", "SX", "SXD", "I",
        "RX", "RY", "RZ", "PHASE", "U3", "CNOT", "CX", "CZ", "SWAP",
        "ISWAP", "CSWAP", "CCX", "CCNOT", "TOFFOLI", "ECR", "XX",
        "M", "MEASURE", "MEAS",
    }
    for index, gate in enumerate(circuit.gates):
        if any(q < 0 or q >= circuit.num_qubits for q in gate.qubits):
            raise CompatibilityError(f"Gate [{index}] {gate} references an invalid qubit.")
        if len(set(gate.qubits)) != len(gate.qubits):
            raise CompatibilityError(f"Gate [{index}] {gate} has duplicate qubit indices.")
        if any(not math.isfinite(p) for p in gate.params):
            raise CompatibilityError(f"Gate [{index}] {gate} has a non-finite parameter.")
        if gate.name not in supported:
            raise CompatibilityError(f"Gate [{index}] {gate.name!r} has no simulator matrix.")
        if not gate.is_measurement:
            expected = 2 ** len(gate.qubits)
            if _gate_matrix(gate).shape != (expected, expected):
                raise CompatibilityError(
                    f"Gate [{index}] {gate.name} matrix does not match its qubit count."
                )


def _apply_circuit_to_simulator(circuit: Circuit, simulator: StatevectorSimulator) -> None:
    for gate in circuit.gates:
        if not gate.is_measurement:
            simulator.apply_gate(_gate_matrix(gate), list(gate.qubits))


def _circuit_to_simulator(circuit: Circuit, simulator: StatevectorSimulator) -> None:
    """Compatibility name for the pipeline's sole IR-to-simulator bridge."""
    _apply_circuit_to_simulator(circuit, simulator)


class QuantumExecutor:
    """Validate, transpile, execute, and package v2 circuits."""

    def __init__(
        self,
        simulator: StatevectorSimulator,
        transpiler: Optional[Transpiler] = None,
        default_optimization_level: int = _DEFAULT_LEVEL,
        default_shots: int = 1024,
        selector: Optional[Any] = None,
    ) -> None:
        if default_optimization_level not in OPTIMIZATION_LEVEL_PASSES:
            raise ValueError("default_optimization_level must be between 0 and 3")
        if default_shots < 0:
            raise ValueError("default_shots must be non-negative")
        self._sim = simulator
        self._tr = transpiler or Transpiler()
        self._default_level = default_optimization_level
        self._default_shots = default_shots
        self._selector = selector

    @classmethod
    def default(
        cls,
        n_qubits: int,
        seed: Optional[int] = None,
        optimization_level: int = _DEFAULT_LEVEL,
        selector: Optional[Any] = None,
    ) -> "QuantumExecutor":
        if optimization_level not in OPTIMIZATION_LEVEL_PASSES:
            raise ValueError("optimization_level must be between 0 and 3")
        return cls(
            StatevectorSimulator(n_qubits, seed=seed),
            Transpiler(passes=OPTIMIZATION_LEVEL_PASSES[optimization_level]),
            default_optimization_level=optimization_level,
            selector=selector,
        )

    def execute(
        self,
        circuit: Circuit,
        optimize: bool = True,
        optimization_level: Optional[int] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        shots: Optional[int] = None,
        debug: bool = False,
        hardware_config: Optional[Any] = None,
    ) -> ExecutionResult:
        started = time.perf_counter()
        level = self._default_level if optimization_level is None else optimization_level
        if level not in OPTIMIZATION_LEVEL_PASSES:
            raise ValueError("optimization_level must be between 0 and 3")
        shots_count = self._default_shots if shots is None else shots
        if shots_count < 0:
            raise ValueError("shots must be non-negative")
        validate_circuit_compatibility(circuit, max_qubits=self._sim.num_qubits)
        if debug:
            self._debug_print_input(circuit)
        if self._selector is not None:
            decision = self._selector.select(circuit)
            selected = self._selector.get_backend(decision)
            result = selected.execute(circuit, shots=shots_count)
            result.metadata["selection_decision"] = decision
            result.metadata["selected_backend"] = decision.selected_backend
            result.counts["__decision__"] = decision.selected_backend
            return result
        if hardware_config is not None:
            from zuntenium.transpiler.hardware_transpiler import HardwareAwareTranspiler

            hardware_result = HardwareAwareTranspiler(hardware_config).transpile(circuit)
            execution_started = time.perf_counter()
            self._sim.initialize(self._sim.num_qubits)
            # The statevector fallback keeps logical-gate semantics; the hardware
            # result supplies routing, native-gate, and cost metadata.
            _apply_circuit_to_simulator(circuit, self._sim)
            execution_ms = (time.perf_counter() - execution_started) * 1000
            statevector = self._sim.get_statevector()
            probabilities = np.abs(statevector) ** 2
            counts = self._sim.sample(shots_count) if shots_count else {}
            return ExecutionResult(
                statevector, probabilities, counts, self._sim.num_qubits, shots_count,
                None, execution_ms, (time.perf_counter() - started) * 1000,
                hardware_result.final_depth, hardware_result.final_gates,
            )
        optimized, report = self._transpile_if_needed(
            circuit, optimize, level, backend_config
        )
        execution_started = time.perf_counter()
        self._sim.initialize(self._sim.num_qubits)
        _apply_circuit_to_simulator(optimized, self._sim)
        execution_ms = (time.perf_counter() - execution_started) * 1000
        statevector = self._sim.get_statevector()
        probabilities = np.abs(statevector) ** 2
        counts = self._sim.sample(shots_count) if shots_count else {}
        result = ExecutionResult(
            statevector, probabilities, counts, self._sim.num_qubits, shots_count,
            report, execution_ms, (time.perf_counter() - started) * 1000,
            optimized.depth(), optimized.num_gates,
        )
        if debug:
            self._debug_print_result(optimized, report, result)
        return result

    def _transpile_if_needed(
        self,
        circuit: Circuit,
        optimize: bool,
        level: int,
        backend_config: Optional[Dict[str, Any]],
    ) -> Tuple[Circuit, Optional[OptimizationReport]]:
        if not optimize or level == 0:
            return circuit, None
        optimized, report = self._tr.transpile(
            circuit, backend_config=backend_config,
            passes=OPTIMIZATION_LEVEL_PASSES[level], debug=False,
        )
        return optimized, report

    @property
    def simulator(self) -> StatevectorSimulator:
        return self._sim

    @property
    def transpiler(self) -> Transpiler:
        return self._tr

    def _debug_print_input(self, circuit: Circuit) -> None:
        print("\n" + "-" * 60)
        print(f"[EXECUTOR] Input circuit: {circuit}")
        print(circuit.draw())

    def _debug_print_result(
        self, circuit: Circuit, report: Optional[OptimizationReport], result: ExecutionResult
    ) -> None:
        print("\n" + "-" * 60)
        if report:
            print("[EXECUTOR] Optimization report:")
            print(f"  {report.summary()}")
            print(report.passes_summary())
        print("\n[EXECUTOR] Optimized circuit:")
        print(circuit.draw())
        print("\n[EXECUTOR] Execution result:")
        print(result.summary())
        print("-" * 60)

    def execute_batch(
        self,
        circuits: List[Circuit],
        optimize: bool = True,
        optimization_level: Optional[int] = None,
        shots: Optional[int] = None,
    ) -> List[ExecutionResult]:
        return [
            self.execute(circuit, optimize, optimization_level, shots=shots)
            for circuit in circuits
        ]


class TranspilerAwareVQE:
    def __init__(
        self,
        ansatz_builder: Callable[[np.ndarray], Circuit],
        hamiltonian: Dict[int, float],
        executor: QuantumExecutor,
        shots: int = 0,
    ) -> None:
        self._builder = ansatz_builder
        self._hamiltonian = hamiltonian
        self._executor = executor
        self._shots = shots
        self._calls = 0
        self._history: List[float] = []

    def cost(self, params: np.ndarray) -> float:
        result = self._executor.execute(self._builder(params), shots=self._shots)
        energy = sum(
            coefficient * result.expectation_z(qubit)
            for qubit, coefficient in self._hamiltonian.items()
            if 0 <= qubit < result.num_qubits
        )
        self._calls += 1
        self._history.append(float(energy))
        return float(energy)

    def run(self, initial_params: np.ndarray, method: str = "COBYLA", max_iter: int = 200) -> Dict[str, Any]:
        try:
            from scipy.optimize import minimize
            result = minimize(
                self.cost, initial_params, method=method,
                options={"maxiter": max_iter, "rhobeg": 0.1},
            )
            return {
                "energy": float(result.fun), "params": result.x, "calls": self._calls,
                "history": self._history, "converged": bool(result.success),
            }
        except ImportError:
            params = initial_params.copy()
            best = self.cost(params)
            for _ in range(max_iter):
                for index in range(len(params)):
                    for delta in (0.1, -0.1, 0.05, -0.05):
                        candidate = params.copy(); candidate[index] += delta
                        value = self.cost(candidate)
                        if value < best:
                            best, params = value, candidate
            return {"energy": best, "params": params, "calls": self._calls, "history": self._history}


class TranspilerAwareQuantumLayer:
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        entanglement: str = "linear",
        optimization_level: int = 2,
        seed: Optional[int] = 0,
    ) -> None:
        if entanglement not in {"linear", "circular", "full"}:
            raise ValueError("entanglement must be linear, circular, or full")
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self._entanglement = entanglement
        self._executor = QuantumExecutor.default(n_qubits, seed, optimization_level)
        rng = np.random.default_rng(seed if seed is not None else 42)
        self.weights = rng.uniform(-math.pi, math.pi, (n_layers, n_qubits, 3))

    def _build_circuit(self, values: np.ndarray, params: np.ndarray) -> Circuit:
        circuit = Circuit(self.n_qubits, "ql_circuit")
        for qubit in range(self.n_qubits):
            value = float(values[qubit]) if qubit < len(values) else 0.0
            circuit.add_gate(Gate("RY", (qubit,), params=(math.pi * math.tanh(value),)))
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                for name, parameter in zip(("RX", "RY", "RZ"), params[layer, qubit]):
                    circuit.add_gate(Gate(name, (qubit,), params=(float(parameter),)))
            if self._entanglement == "linear":
                pairs = [(q, q + 1) for q in range(self.n_qubits - 1)]
            elif self._entanglement == "circular":
                pairs = [(q, (q + 1) % self.n_qubits) for q in range(self.n_qubits)]
            else:
                pairs = [(q, other) for q in range(self.n_qubits) for other in range(q + 1, self.n_qubits)]
            for control, target in pairs:
                if control != target:
                    circuit.add_gate(Gate("CNOT", (target,), controls=(control,)))
        return circuit

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        output = np.zeros((x_batch.shape[0], self.n_qubits), dtype=np.float64)
        for index, values in enumerate(x_batch):
            normalized = (values - float(np.mean(values))) / (float(np.std(values)) + 1e-6)
            output[index] = self._executor.execute(
                self._build_circuit(normalized, self.weights), shots=0
            ).all_z_expectations()
        return output


def benchmark_integration(n_trials: int = 30, verbose: bool = True) -> Dict[str, Any]:
    configs = [(4, 30), (8, 50), (10, 100), (15, 100), (20, 200)]
    results: Dict[str, Any] = {}
    for n, depth in configs:
        circuits = [_build_random(n, depth) for _ in range(n_trials)]
        raw_times, optimized_times, transpile_times, reductions = [], [], [], []
        raw_sim = StatevectorSimulator(n, seed=0)
        optimized_sim = StatevectorSimulator(n, seed=0)
        transpiler = Transpiler(passes=OPTIMIZATION_LEVEL_PASSES[2])
        for circuit in circuits:
            started = time.perf_counter(); raw_sim.initialize(n)
            _apply_circuit_to_simulator(circuit, raw_sim)
            raw_times.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter(); optimized, report = transpiler.transpile(circuit)
            transpile_times.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter(); optimized_sim.initialize(n)
            _apply_circuit_to_simulator(optimized, optimized_sim)
            optimized_times.append((time.perf_counter() - started) * 1000)
            reductions.append(report.gate_reduction_pct)
        label = f"n={n} d={depth}"
        results[label] = {
            "raw_ms": statistics.median(raw_times),
            "exec_ms": statistics.median(optimized_times),
            "tr_ms": statistics.median(transpile_times),
            "speedup": statistics.median(raw_times) / max(statistics.median(optimized_times), 1e-6),
            "gate_pct_reduced": statistics.mean(reductions),
        }
        if verbose:
            print(label, results[label])
    return results


def _build_random_circuit(n: int, depth: int, seed: int = 0) -> Circuit:
    """Build a deterministic random circuit for integration benchmarks."""
    if seed == 0:
        return _build_random(n, depth)
    circuit = Circuit(n, f"random_n{n}_d{depth}")
    rng = np.random.default_rng(seed)
    one_qubit = ["X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"]
    two_qubit = ["CNOT", "CZ", "SWAP"]
    for _ in range(depth):
        if n > 1 and rng.random() < 0.3:
            first, second = rng.choice(n, size=2, replace=False)
            name = two_qubit[int(rng.integers(len(two_qubit)))]
            if name == "CNOT":
                circuit.add_gate(Gate(name, (int(second),), controls=(int(first),)))
            else:
                circuit.add_gate(Gate(name, (int(first), int(second))))
        else:
            qubit = int(rng.integers(n))
            name = one_qubit[int(rng.integers(len(one_qubit)))]
            params = (float(rng.uniform(0, 2 * math.pi)),) if name in {"RX", "RY", "RZ"} else ()
            circuit.add_gate(Gate(name, (qubit,), params=params))
    return circuit


class IntegrationTestSuite:
    """Small executable smoke suite covering the integration boundary."""

    def __init__(self) -> None:
        self._passed = 0
        self._failed = 0

    def _assert(self, condition: bool, message: str) -> None:
        if not condition:
            self._failed += 1
            raise AssertionError(message)
        self._passed += 1

    def test_identity_circuit(self) -> None:
        result = QuantumExecutor.default(2).execute(Circuit(2), shots=0)
        self._assert(result.num_gates_executed == 0, "identity circuit applies no gates")
        self._assert(np.isclose(result.probabilities[0], 1.0), "identity preserves |00>")

    def test_bell_state(self) -> None:
        circuit = Circuit(2)
        circuit.add_gate(Gate("H", (0,)))
        circuit.add_gate(Gate("CNOT", (1,), controls=(0,)))
        result = QuantumExecutor.default(2).execute(circuit, shots=0)
        self._assert(np.allclose(result.probabilities, [0.5, 0, 0, 0.5]), "Bell probabilities")
        self._assert(np.allclose(result.all_z_expectations(), [0, 0]), "Bell Z expectations")

    def test_ghz_state(self) -> None:
        circuit = Circuit(3)
        circuit.add_gate(Gate("H", (0,)))
        circuit.add_gate(Gate("CNOT", (1,), controls=(0,)))
        circuit.add_gate(Gate("CNOT", (2,), controls=(1,)))
        result = QuantumExecutor.default(3).execute(circuit, shots=0)
        self._assert(np.isclose(result.probabilities[0], 0.5), "GHZ |000>")
        self._assert(np.isclose(result.probabilities[7], 0.5), "GHZ |111>")

    def test_optimization_and_raw_execution(self) -> None:
        circuit = Circuit(1)
        circuit.add_gate(Gate("H", (0,)))
        circuit.add_gate(Gate("H", (0,)))
        executor = QuantumExecutor.default(1, optimization_level=1)
        optimized = executor.execute(circuit, shots=0)
        raw = executor.execute(circuit, optimize=False, shots=0)
        self._assert(optimized.num_gates_executed == 0, "optimization cancels H.H")
        self._assert(raw.opt_report is None, "raw execution has no optimization report")

    def test_rotation_merge(self) -> None:
        circuit = Circuit(1)
        for angle in (0.3, 0.5, 0.2):
            circuit.add_gate(Gate("RX", (0,), params=(angle,)))
        reference = Circuit(1)
        reference.add_gate(Gate("RX", (0,), params=(1.0,)))
        executor = QuantumExecutor.default(1)
        optimized = executor.execute(circuit, shots=0)
        expected = executor.execute(reference, optimize=False, shots=0)
        self._assert(abs(np.vdot(optimized.statevector, expected.statevector)) > 1 - 1e-6,
                     "rotation merge preserves state")

    def test_parameterized_gate(self) -> None:
        circuit = Circuit(1)
        circuit.add_gate(Gate("RX", (0,), params=(math.pi / 2,)))
        result = QuantumExecutor.default(1).execute(circuit, shots=0)
        expected = np.array([1 / math.sqrt(2), -1j / math.sqrt(2)])
        self._assert(np.allclose(result.statevector, expected), "RX(pi/2) state")

    def test_optimization_level_zero(self) -> None:
        circuit = Circuit(1)
        circuit.add_gate(Gate("H", (0,)))
        circuit.add_gate(Gate("H", (0,)))
        result = QuantumExecutor.default(1).execute(circuit, optimization_level=0, shots=0)
        self._assert(result.num_gates_executed == 2, "level zero keeps raw gates")

    def test_level_one_cancellation(self) -> None:
        circuit = Circuit(1)
        circuit.add_gate(Gate("H", (0,)))
        circuit.add_gate(Gate("H", (0,)))
        result = QuantumExecutor.default(1).execute(circuit, optimization_level=1, shots=0)
        self._assert(result.num_gates_executed == 0, "level one cancels H.H")

    def test_redundant_gates(self) -> None:
        circuit = Circuit(1)
        for _ in range(6):
            circuit.add_gate(Gate("X", (0,)))
        result = QuantumExecutor.default(1).execute(circuit, shots=0)
        self._assert(result.num_gates_executed == 0, "six X gates cancel")
        self._assert(np.isclose(result.probabilities[0], 1.0), "six X gates preserve |0>")

    def test_toffoli(self) -> None:
        circuit = Circuit(3)
        circuit.add_gate(Gate("X", (0,)))
        circuit.add_gate(Gate("X", (1,)))
        circuit.add_gate(Gate("CCX", (2,), controls=(0, 1)))
        result = QuantumExecutor.default(3, optimization_level=0).execute(circuit, shots=0)
        self._assert(np.isclose(result.probabilities[7], 1.0), "Toffoli maps |110> to |111>")

    def test_level_equivalence(self) -> None:
        circuit = Circuit(2)
        circuit.add_gate(Gate("RX", (0,), params=(0.5,)))
        circuit.add_gate(Gate("RX", (0,), params=(0.5,)))
        circuit.add_gate(Gate("H", (1,)))
        raw = QuantumExecutor.default(2, optimization_level=0).execute(circuit, shots=0)
        optimized = QuantumExecutor.default(2, optimization_level=2).execute(circuit, shots=0)
        self._assert(abs(np.vdot(raw.statevector, optimized.statevector)) > 1 - 1e-6,
                     "level zero and level two preserve state")
        self._assert(optimized.num_gates_executed <= raw.num_gates_executed,
                     "level two does not increase gate count")

    def test_twenty_qubit_performance(self) -> None:
        circuit = Circuit(20)
        for qubit in range(20):
            circuit.add_gate(Gate("H", (qubit,)))
        result = QuantumExecutor.default(20, optimization_level=1).execute(circuit, shots=0)
        self._assert(result.statevector.shape == (2 ** 20,), "20-qubit statevector shape")

    def test_validation(self) -> None:
        circuit = Circuit(2)
        circuit._gates.append(Gate("X", (2,)))
        circuit._rebuild_timelines()
        try:
            validate_circuit_compatibility(circuit, max_qubits=2)
        except CompatibilityError:
            self._passed += 1
        else:
            raise AssertionError("out-of-bounds gate was accepted")

    def test_batch_and_shims(self) -> None:
        circuits = [Circuit(1) for _ in range(2)]
        layer = TranspilerAwareQuantumLayer(1, n_layers=1, optimization_level=1)
        self._assert(len(QuantumExecutor.default(1).execute_batch(circuits, shots=0)) == 2,
                     "batch execution returns one result per circuit")
        self._assert(layer.forward(np.ones((2, 1))).shape == (2, 1), "layer output shape")

    def test_vqe_and_expectations(self) -> None:
        def ansatz(params: np.ndarray) -> Circuit:
            circuit = Circuit(1)
            circuit.add_gate(Gate("RY", (0,), params=(float(params[0]),)))
            return circuit

        executor = QuantumExecutor.default(1)
        vqe = TranspilerAwareVQE(ansatz, {0: 1.0}, executor, shots=0)
        self._assert(isinstance(vqe.cost(np.array([0.2])), float), "VQE returns float")
        self._assert(-1.0 <= vqe.cost(np.array([0.2])) <= 1.0, "VQE energy range")

    def test_debug_output(self) -> None:
        circuit = Circuit(1)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            QuantumExecutor.default(1).execute(circuit, debug=True, shots=0)
        output = buffer.getvalue()
        self._assert("[EXECUTOR] Input circuit" in output, "debug input header")
        self._assert("[EXECUTOR] Execution result" in output, "debug result header")

    def run_all(self) -> None:
        for name in sorted(method for method in dir(self) if method.startswith("test_")):
            getattr(self, name)()
        print(f"IntegrationTestSuite: {self._passed} passed")


__all__ = [
    "OPTIMIZATION_LEVEL_PASSES", "ExecutionResult", "CompatibilityError",
    "validate_circuit_compatibility", "QuantumExecutor", "TranspilerAwareVQE",
    "TranspilerAwareQuantumLayer", "benchmark_integration", "_circuit_to_simulator",
    "_build_random_circuit", "IntegrationTestSuite",
]
