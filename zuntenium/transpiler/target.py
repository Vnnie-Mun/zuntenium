"""
Target — device capabilities and constraints.

Describes what a backend can do: which gates it supports, qubit count,
connectivity, gate fidelities, and timing. Used by the transpiler to
generate hardware-compatible circuits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zuntenium.transpiler.coupling_map import CouplingMap


@dataclass
class GateProperties:
    """Properties for a specific gate on specific qubits."""
    gate_error: float = 0.0          # error rate [0, 1]
    gate_length_ns: float = 0.0      # execution time in nanoseconds
    calibration_date: str = ""


@dataclass
class Target:
    """
    Hardware target describing device capabilities.

    Passed to transpiler passes so they can optimize circuits for a
    specific backend's constraints.

    Attributes:
        name: Device name.
        num_qubits: Number of physical qubits.
        coupling_map: Qubit connectivity.
        supported_gates: Set of gate names the device natively supports.
        gate_properties: Optional per-gate performance properties.
        t1_us: T1 (amplitude damping) times per qubit in microseconds.
        t2_us: T2 (dephasing) times per qubit in microseconds.
    """
    name: str = "simulator"
    num_qubits: int = 30
    coupling_map: CouplingMap | None = None
    supported_gates: set[str] = field(default_factory=lambda: {
        "H", "X", "Y", "Z", "S", "T",
        "Rx", "Ry", "Rz", "U1", "U2", "U3",
        "CX", "CZ", "SWAP", "CRz",
    })
    gate_properties: dict[str, dict[tuple[int, ...], GateProperties]] = field(
        default_factory=dict
    )
    t1_us: dict[int, float] = field(default_factory=dict)
    t2_us: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports_gate(self, gate_name: str) -> bool:
        return gate_name in self.supported_gates

    def is_simulator(self) -> bool:
        return self.name == "simulator" or "sim" in self.name.lower()

    @classmethod
    def simulator(cls, num_qubits: int = 30) -> Target:
        """Return a default simulator target (all-to-all, all gates)."""
        return cls(
            name="zuntenium_statevector_simulator",
            num_qubits=num_qubits,
            coupling_map=CouplingMap.fully_connected(num_qubits),
        )

    def __repr__(self) -> str:
        return f"Target(name='{self.name}', num_qubits={self.num_qubits})"
