"""
StabilizerSimulator — Full Gottesman-Knill tableau simulator.

Efficiently simulates Clifford circuits using a binary stabilizer tableau
of size (2n) × (2n + 1). Supports H, S, X, Y, Z, CNOT/CX, CZ, SWAP,
and measurement. Scales to thousands of qubits in polynomial time.

Reference:
    Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits",
    Phys. Rev. A 70, 052328 (2004).

The tableau tracks n stabilizer generators and n destabilizer generators.
Each generator is encoded as a binary row [x₁..xₙ | z₁..zₙ | phase],
where the Pauli operator is (-1)^phase · X^x · Z^z (up to factors of i).
"""
from __future__ import annotations

import numpy as np
from typing import Any

from zuntenium.backends.ibm_backend import BackendInfo, QuantumBackend


class StabilizerTableau:
    """
    Binary stabilizer tableau for n qubits.

    Layout:
        Rows 0..n-1:     destabilizers
        Rows n..2n-1:    stabilizers
        Columns 0..n-1:  X bits
        Columns n..2n-1: Z bits
        Column 2n:       phase bit (0 → +1, 1 → -1)
    """

    def __init__(self, n: int) -> None:
        self.n = n
        # (2n) × (2n + 1) binary tableau
        self.table = np.zeros((2 * n, 2 * n + 1), dtype=np.uint8)
        # Initialize identity tableau:
        # destabilizer[i] = X_i, stabilizer[i] = Z_i
        for i in range(n):
            self.table[i, i] = 1          # destabilizer X bits
            self.table[n + i, n + i] = 1  # stabilizer Z bits

    def _x(self, row: int, qubit: int) -> int:
        """Get the X bit for a given row and qubit."""
        return int(self.table[row, qubit])

    def _z(self, row: int, qubit: int) -> int:
        """Get the Z bit for a given row and qubit."""
        return int(self.table[row, self.n + qubit])

    def _r(self, row: int) -> int:
        """Get the phase bit for a row."""
        return int(self.table[row, 2 * self.n])

    def _set_x(self, row: int, qubit: int, val: int) -> None:
        self.table[row, qubit] = val & 1

    def _set_z(self, row: int, qubit: int, val: int) -> None:
        self.table[row, self.n + qubit] = val & 1

    def _set_r(self, row: int, val: int) -> None:
        self.table[row, 2 * self.n] = val & 1

    # ── Row multiplication (Aaronson-Gottesman rowsum) ────────────────

    def _rowsum(self, h: int, i: int) -> None:
        """
        Replace row h with the product of rows h and i.

        Implements the rowsum procedure from Aaronson-Gottesman (2004),
        tracking phase correctly through the Pauli group multiplication.
        """
        n = self.n

        # Compute phase contribution
        phase_sum = 2 * self._r(h) + 2 * self._r(i)
        for j in range(n):
            xi, zi = self._x(i, j), self._z(i, j)
            xh, zh = self._x(h, j), self._z(h, j)
            # g(x1,z1,x2,z2) contribution to phase
            if xi == 1 and zi == 1:
                # Y: Y·I=0, Y·X=1, Y·Z=-1, Y·Y=0
                phase_sum += xh - zh
            elif xi == 1 and zi == 0:
                # X: X·Z=1, X·Y=-1
                phase_sum += zh * (2 * xh - 1)
            elif xi == 0 and zi == 1:
                # Z: Z·X=-1, Z·Y=1
                phase_sum += xh * (1 - 2 * zh)

        # XOR the Pauli bits
        for j in range(n):
            self.table[h, j] ^= self.table[i, j]
            self.table[h, n + j] ^= self.table[i, n + j]

        # Set phase: phase_sum mod 4 → {0,1}
        self._set_r(h, 1 if (phase_sum % 4) == 2 else 0)

    # ── Clifford gate implementations ─────────────────────────────────

    def hadamard(self, qubit: int) -> None:
        """Apply Hadamard gate: X↔Z, phase update."""
        n = self.n
        for i in range(2 * n):
            # Phase update: r ^= x·z
            xi = self._x(i, qubit)
            zi = self._z(i, qubit)
            self._set_r(i, self._r(i) ^ (xi & zi))
            # Swap X and Z bits
            self._set_x(i, qubit, zi)
            self._set_z(i, qubit, xi)

    def phase_gate(self, qubit: int) -> None:
        """Apply S gate: Z → Z, X → Y (i.e., X → XZ)."""
        n = self.n
        for i in range(2 * n):
            xi = self._x(i, qubit)
            zi = self._z(i, qubit)
            # Phase update: r ^= x·z
            self._set_r(i, self._r(i) ^ (xi & zi))
            # Z bit: z ^= x
            self._set_z(i, qubit, zi ^ xi)

    def cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate."""
        n = self.n
        for i in range(2 * n):
            xc = self._x(i, control)
            zc = self._z(i, control)
            xt = self._x(i, target)
            zt = self._z(i, target)
            # Phase: r ^= x_control · z_target · (x_target ⊕ z_control ⊕ 1)
            self._set_r(
                i, self._r(i) ^ (xc & zt & (xt ^ zc ^ 1))
            )
            # X propagation: x_target ^= x_control
            self._set_x(i, target, xt ^ xc)
            # Z propagation: z_control ^= z_target
            self._set_z(i, control, zc ^ zt)

    def x_gate(self, qubit: int) -> None:
        """Apply Pauli X: HZH. X flips the phase where Z bit is set."""
        n = self.n
        for i in range(2 * n):
            zi = self._z(i, qubit)
            self._set_r(i, self._r(i) ^ zi)

    def y_gate(self, qubit: int) -> None:
        """Apply Pauli Y: SHXHS†. Flips phase where X or Z bit is set."""
        n = self.n
        for i in range(2 * n):
            xi = self._x(i, qubit)
            zi = self._z(i, qubit)
            self._set_r(i, self._r(i) ^ (xi ^ zi))

    def z_gate(self, qubit: int) -> None:
        """Apply Pauli Z: Flips phase where X bit is set."""
        n = self.n
        for i in range(2 * n):
            xi = self._x(i, qubit)
            self._set_r(i, self._r(i) ^ xi)

    def cz(self, q0: int, q1: int) -> None:
        """Apply CZ gate: H on target, CNOT, H on target."""
        self.hadamard(q1)
        self.cnot(q0, q1)
        self.hadamard(q1)

    def swap(self, q0: int, q1: int) -> None:
        """Apply SWAP gate: three CNOTs."""
        self.cnot(q0, q1)
        self.cnot(q1, q0)
        self.cnot(q0, q1)

    def sdg(self, qubit: int) -> None:
        """Apply S† gate: S³."""
        self.phase_gate(qubit)
        self.phase_gate(qubit)
        self.phase_gate(qubit)

    # ── Measurement ───────────────────────────────────────────────────

    def measure(self, qubit: int, rng: np.random.Generator | None = None) -> int:
        """
        Measure a qubit in the computational basis.

        Implements the Aaronson-Gottesman measurement algorithm:
        1. If any stabilizer has x_qubit = 1, outcome is random.
        2. Otherwise, outcome is deterministic from destabilizers.

        Returns 0 or 1.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = self.n

        # Check if any stabilizer generator (rows n..2n-1) has x_qubit = 1
        p = None
        for i in range(n, 2 * n):
            if self._x(i, qubit) == 1:
                p = i
                break

        if p is not None:
            # Random outcome
            # Set destabilizer[p-n] = stabilizer[p]
            # Then rowsum all other rows with x_qubit=1 into p
            for i in range(2 * n):
                if i != p and self._x(i, qubit) == 1:
                    self._rowsum(i, p)
            # Set destabilizer[p - n] = old stabilizer[p]
            self.table[p - n] = self.table[p].copy()
            # Reset stabilizer[p] to just Z_qubit with random phase
            self.table[p] = 0
            self._set_z(p, qubit, 1)
            outcome = int(rng.integers(0, 2))
            self._set_r(p, outcome)
            return outcome
        else:
            # Deterministic outcome
            # Scratch row approach: compute the product of destabilizers
            # where the corresponding stabilizer has x_qubit = 1
            # Actually: use a temporary extra row
            scratch = np.zeros(2 * n + 1, dtype=np.uint8)
            # Find destabilizers with x_qubit = 1
            for i in range(n):
                if self._x(i, qubit) == 1:
                    # Multiply scratch by stabilizer[i + n]
                    # We simulate rowsum manually on scratch
                    stab_row = i + n
                    phase_sum = 2 * int(scratch[2 * n]) + 2 * self._r(stab_row)
                    for j in range(n):
                        xi = self._x(stab_row, j)
                        zi = self._z(stab_row, j)
                        xs = int(scratch[j])
                        zs = int(scratch[n + j])
                        if xi == 1 and zi == 1:
                            phase_sum += xs - zs
                        elif xi == 1 and zi == 0:
                            phase_sum += zs * (2 * xs - 1)
                        elif xi == 0 and zi == 1:
                            phase_sum += xs * (1 - 2 * zs)
                    for j in range(n):
                        scratch[j] ^= self.table[stab_row, j]
                        scratch[n + j] ^= self.table[stab_row, n + j]
                    scratch[2 * n] = 1 if (phase_sum % 4) == 2 else 0

            return int(scratch[2 * n])


# ── Gate dispatch map ─────────────────────────────────────────────────

_CLIFFORD_DISPATCH = {
    "H": "hadamard",
    "X": "x_gate",
    "Y": "y_gate",
    "Z": "z_gate",
    "S": "phase_gate",
    "SDG": "sdg",
    "CX": "cnot",
    "CNOT": "cnot",
    "CZ": "cz",
    "SWAP": "swap",
}


# ── StabilizerSimulator backend ───────────────────────────────────────

class StabilizerSimulator(QuantumBackend):
    """
    Full stabilizer simulator using the Gottesman-Knill tableau.

    Efficiently simulates circuits composed entirely of Clifford gates
    (H, S, X, Y, Z, CNOT, CZ, SWAP) in O(n²) time per gate, where
    n is the number of qubits. Can handle thousands of qubits.

    Non-Clifford gates (T, Rx, Ry, Rz with arbitrary angles) will
    raise ValueError.
    """

    # Clifford gates that can be mapped to special rotation angles
    _CLIFFORD_ROTATIONS = {
        "RZ": {0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi,
               -np.pi / 2, -np.pi, -3 * np.pi / 2},
        "RX": {0.0, np.pi, 2 * np.pi, -np.pi},
        "RY": {0.0, np.pi, 2 * np.pi, -np.pi},
    }

    def __init__(self, max_qubits: int = 10000) -> None:
        self._max_qubits = max_qubits

    def _validate_clifford(self, gate_name: str, params: tuple | list = ()) -> str | None:
        """
        Check if a gate is Clifford. Returns the canonical Clifford name
        if it maps to one, or None if it's non-Clifford.
        """
        upper = gate_name.upper()
        if upper in _CLIFFORD_DISPATCH:
            return upper

        # Check parametric gates at Clifford angles
        if upper in self._CLIFFORD_ROTATIONS and params:
            angle = float(params[0]) % (2 * np.pi)
            for cliff_angle in self._CLIFFORD_ROTATIONS[upper]:
                if abs(angle - (cliff_angle % (2 * np.pi))) < 1e-9:
                    return upper  # Will decompose below
            return None

        # SX = √X is Clifford
        if upper in ("SX", "SXD", "SXDG"):
            return upper

        # Measure/barrier/reset are fine
        if upper in ("MEASURE", "MEAS", "M", "BARRIER", "RESET", "I"):
            return upper

        return None

    def _apply_gate(
        self,
        tableau: StabilizerTableau,
        name: str,
        qubits: tuple[int, ...] | list[int],
        params: tuple | list = (),
    ) -> None:
        """Apply a single gate to the tableau."""
        upper = name.upper()

        # Direct Clifford dispatch
        if upper in _CLIFFORD_DISPATCH:
            method = getattr(tableau, _CLIFFORD_DISPATCH[upper])
            if upper in ("CX", "CNOT", "CZ", "SWAP"):
                method(qubits[0], qubits[1])
            else:
                method(qubits[0])
            return

        # SX gate: H · S · H
        if upper == "SX":
            tableau.hadamard(qubits[0])
            tableau.phase_gate(qubits[0])
            tableau.hadamard(qubits[0])
            return

        if upper in ("SXD", "SXDG"):
            tableau.hadamard(qubits[0])
            tableau.sdg(qubits[0])
            tableau.hadamard(qubits[0])
            return

        # I (identity) — no-op
        if upper == "I":
            return

        # Parametric Clifford rotations at special angles
        if upper == "RZ" and params:
            angle = float(params[0]) % (2 * np.pi)
            # RZ(0) = I, RZ(π/2) = S, RZ(π) = Z, RZ(3π/2) = S†
            if abs(angle) < 1e-9 or abs(angle - 2 * np.pi) < 1e-9:
                return  # identity
            if abs(angle - np.pi / 2) < 1e-9:
                tableau.phase_gate(qubits[0])
                return
            if abs(angle - np.pi) < 1e-9:
                tableau.z_gate(qubits[0])
                return
            if abs(angle - 3 * np.pi / 2) < 1e-9:
                tableau.sdg(qubits[0])
                return

        if upper == "RX" and params:
            angle = float(params[0]) % (2 * np.pi)
            if abs(angle) < 1e-9 or abs(angle - 2 * np.pi) < 1e-9:
                return
            if abs(angle - np.pi) < 1e-9:
                tableau.x_gate(qubits[0])
                return

        if upper == "RY" and params:
            angle = float(params[0]) % (2 * np.pi)
            if abs(angle) < 1e-9 or abs(angle - 2 * np.pi) < 1e-9:
                return
            if abs(angle - np.pi) < 1e-9:
                tableau.y_gate(qubits[0])
                return

    def execute(self, circuit: Any, shots: int = 1024, **kwargs: Any) -> Any:
        """
        Execute a Clifford circuit using the stabilizer tableau.

        Args:
            circuit: A transpiler v2 Circuit object.
            shots: Number of measurement shots.

        Returns:
            ExecutionResult with measurement counts.

        Raises:
            ValueError: If the circuit contains non-Clifford gates.
        """
        from zuntenium.backends.ibm_backend import SimulatorBackend

        n_qubits = circuit.num_qubits
        if n_qubits > self._max_qubits:
            raise ValueError(
                f"Circuit has {n_qubits} qubits, exceeding stabilizer "
                f"simulator limit of {self._max_qubits}."
            )

        # Validate all gates are Clifford
        measurement_qubits: list[int] = []
        gate_ops: list[tuple[str, tuple, tuple]] = []

        for gate in circuit.gates:
            is_meas = getattr(gate, 'is_measurement', False)
            gname_upper = getattr(gate, 'name', '').upper()
            if is_meas or gname_upper in ('MEASURE', 'MEAS', 'M'):
                measurement_qubits.extend(gate.qubits)
                continue

            cliff_name = self._validate_clifford(gate.name, gate.params)
            if cliff_name is None:
                raise ValueError(
                    f"Non-Clifford gate '{gate.name}' with params "
                    f"{gate.params} cannot be simulated by the stabilizer "
                    f"simulator. Use the statevector backend instead."
                )

            if cliff_name in ("BARRIER", "RESET", "MEASURE", "MEAS", "M"):
                continue

            gate_ops.append((gate.name, gate.qubits, gate.params))

        # Build and evolve the tableau
        tableau = StabilizerTableau(n_qubits)
        for name, qubits, params in gate_ops:
            self._apply_gate(tableau, name, qubits, params)

        # If no measurements specified, measure all qubits
        if not measurement_qubits:
            measurement_qubits = list(range(n_qubits))

        # Sample measurements
        rng = np.random.default_rng()
        counts: dict[str, int] = {}

        for _ in range(max(shots, 1)):
            # Create a fresh copy of the tableau for each shot
            # (measurement is destructive)
            tab_copy = StabilizerTableau(n_qubits)
            tab_copy.table = tableau.table.copy()

            bits = []
            for q in measurement_qubits:
                bits.append(str(tab_copy.measure(q, rng)))
            bitstring = "".join(bits)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        # Build result in the same format as SimulatorBackend
        from zuntenium.backends.ibm_backend import ExecutionResult as IBMResult
        try:
            return IBMResult(
                counts=counts,
                probabilities={k: v / max(shots, 1) for k, v in counts.items()},
                statevector=None,
                num_qubits=n_qubits,
                metadata={
                    "backend": "stabilizer",
                    "shots": shots,
                    "simulator_type": "gottesman_knill_tableau",
                    "num_qubits": n_qubits,
                },
            )
        except Exception:
            # Fallback to a simple namespace if IBMResult has different signature
            class _Result:
                def __init__(self, **kw: Any):
                    self.__dict__.update(kw)
                    self.metadata = kw.get("metadata", {})
            return _Result(
                counts=counts,
                probabilities={k: v / max(shots, 1) for k, v in counts.items()},
                statevector=None,
                num_qubits=n_qubits,
                metadata={
                    "backend": "stabilizer",
                    "shots": shots,
                    "simulator_type": "gottesman_knill_tableau",
                    "num_qubits": n_qubits,
                },
            )

    def get_backend_info(self) -> BackendInfo:
        return BackendInfo(
            "stabilizer", "local", self._max_qubits, True, 0.0, 0.0,
        )

    def supports_noise_model(self) -> bool:
        return False

    def supports_batching(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def get_available_backends(self, **kwargs: Any) -> list[BackendInfo]:
        return [self.get_backend_info()]


__all__ = ["StabilizerTableau", "StabilizerSimulator"]
