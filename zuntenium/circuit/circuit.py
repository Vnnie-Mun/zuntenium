"""
QuantumCircuit — Fluent builder for quantum circuits.

All gate methods return self for chaining:
    qc = QuantumCircuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
"""
from __future__ import annotations

from typing import Any

import numpy as np

from zuntenium.circuit.instruction import Instruction
from zuntenium.circuit.register import ClassicalRegister, QuantumRegister
from zuntenium.circuit.validator import CircuitValidator
from zuntenium.quantum_core.exceptions import CircuitCompilationError, QubitIndexError
from zuntenium.quantum_core.gates import (
    BaseGate,
    CRxGate,
    CRyGate,
    CRzGate,
    CXGate,
    CYGate,
    CZGate,
    FredkinGate,
    HGate,
    ISWAPGate,
    RxGate,
    RyGate,
    RzGate,
    SGate,
    SWAPGate,
    TGate,
    ToffoliGate,
    U1Gate,
    U2Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)


class QuantumCircuit:
    """
    Fluent builder for quantum circuits.

    All gate methods return self for chaining. Call .compile() before
    passing to a backend (or let the backend compile automatically).

    Args:
        num_qubits: Number of qubits.
        num_clbits: Number of classical bits (for measurement storage).
        name: Optional circuit name.

    Example::

        qc = QuantumCircuit(2, name="bell")
        qc.h(0).cx(0, 1).measure_all()
        result = backend.run(qc, shots=1024)
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int = 0,
        name: str = "",
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self._qreg = QuantumRegister(num_qubits)
        self._creg = ClassicalRegister(max(num_clbits, 0))
        self._instructions: list[Instruction] = []
        self._name = name or f"circuit_{id(self):x}"
        self._parameters: dict[str, float] = {}
        self._compiled: bool = False
        self._ir: Any = None  # QuantumIR, set after compile()

    # ── Single-qubit gates ────────────────────────────────────────────────

    def h(self, qubit: int) -> QuantumCircuit:
        """Hadamard gate."""
        return self._add_gate(HGate(), [qubit])

    def x(self, qubit: int) -> QuantumCircuit:
        """Pauli-X (NOT) gate."""
        return self._add_gate(XGate(), [qubit])

    def y(self, qubit: int) -> QuantumCircuit:
        """Pauli-Y gate."""
        return self._add_gate(YGate(), [qubit])

    def z(self, qubit: int) -> QuantumCircuit:
        """Pauli-Z gate."""
        return self._add_gate(ZGate(), [qubit])

    def s(self, qubit: int) -> QuantumCircuit:
        """S phase gate."""
        return self._add_gate(SGate(), [qubit])

    def t(self, qubit: int) -> QuantumCircuit:
        """T gate."""
        return self._add_gate(TGate(), [qubit])

    def rx(self, theta: float, qubit: int) -> QuantumCircuit:
        """Rx rotation by angle theta."""
        return self._add_gate(RxGate(theta), [qubit])

    def ry(self, theta: float, qubit: int) -> QuantumCircuit:
        """Ry rotation by angle theta."""
        return self._add_gate(RyGate(theta), [qubit])

    def rz(self, theta: float, qubit: int) -> QuantumCircuit:
        """Rz rotation by angle theta."""
        return self._add_gate(RzGate(theta), [qubit])

    def u1(self, lam: float, qubit: int) -> QuantumCircuit:
        """U1(λ) = Rz(λ) up to global phase."""
        return self._add_gate(U1Gate(lam), [qubit])

    def u2(self, phi: float, lam: float, qubit: int) -> QuantumCircuit:
        """U2(φ, λ)."""
        return self._add_gate(U2Gate(phi, lam), [qubit])

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> QuantumCircuit:
        """U3(θ, φ, λ) — General single-qubit unitary."""
        return self._add_gate(U3Gate(theta, phi, lam), [qubit])

    # ── Two-qubit gates ───────────────────────────────────────────────────

    def cx(self, control: int, target: int) -> QuantumCircuit:
        """CNOT gate."""
        return self._add_gate(CXGate(), [control, target])

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Controlled-Z gate."""
        return self._add_gate(CZGate(), [control, target])

    def cy(self, control: int, target: int) -> QuantumCircuit:
        """Controlled-Y gate."""
        return self._add_gate(CYGate(), [control, target])

    def swap(self, q1: int, q2: int) -> QuantumCircuit:
        """SWAP gate."""
        return self._add_gate(SWAPGate(), [q1, q2])

    def iswap(self, q1: int, q2: int) -> QuantumCircuit:
        """iSWAP gate."""
        return self._add_gate(ISWAPGate(), [q1, q2])

    def crx(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Controlled-Rx."""
        return self._add_gate(CRxGate(theta), [control, target])

    def cry(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Controlled-Ry."""
        return self._add_gate(CRyGate(theta), [control, target])

    def crz(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Controlled-Rz."""
        return self._add_gate(CRzGate(theta), [control, target])

    # ── Multi-qubit gates ─────────────────────────────────────────────────

    def ccx(self, ctrl0: int, ctrl1: int, target: int) -> QuantumCircuit:
        """Toffoli (CCX) gate."""
        return self._add_gate(ToffoliGate(), [ctrl0, ctrl1, target])

    def cswap(self, control: int, t0: int, t1: int) -> QuantumCircuit:
        """Fredkin (CSWAP) gate."""
        return self._add_gate(FredkinGate(), [control, t0, t1])

    def append(self, gate: BaseGate, qubits: list[int]) -> QuantumCircuit:
        """Append an arbitrary gate."""
        return self._add_gate(gate, qubits)

    # ── Operations ────────────────────────────────────────────────────────

    def measure(self, qubit: int, clbit: int) -> QuantumCircuit:
        """Measure qubit into classical bit. Auto-expands the classical register if needed."""
        from zuntenium.circuit.register import ClassicalRegister
        self._validate_qubit(qubit)
        # Auto-expand classical register if needed
        if clbit >= self._creg.size:
            self._creg = ClassicalRegister(clbit + 1)
        self._instructions.append(
            Instruction(operation="measure", qubits=[qubit], clbits=[clbit])
        )
        self._compiled = False
        return self

    def measure_all(self) -> QuantumCircuit:
        """Measure all qubits. Adds classical bits if needed."""
        n = self._qreg.size
        if self._creg.size < n:
            self._creg = ClassicalRegister(n)
        for q in range(n):
            self.measure(q, q)
        return self

    def barrier(self, *qubits: int) -> QuantumCircuit:
        """Add a barrier (no-op in simulation, used for visualization)."""
        targets = list(qubits) if qubits else list(range(self._qreg.size))
        self._instructions.append(
            Instruction(operation="barrier", qubits=targets)
        )
        return self

    def reset(self, qubit: int) -> QuantumCircuit:
        """Reset qubit to |0⟩."""
        self._validate_qubit(qubit)
        self._instructions.append(
            Instruction(operation="reset", qubits=[qubit])
        )
        self._compiled = False
        return self

    def compose(self, other: QuantumCircuit) -> QuantumCircuit:
        """
        Append all instructions from another circuit to this one.

        The other circuit must have ≤ num_qubits qubits.
        """
        if other.num_qubits > self.num_qubits:
            raise CircuitCompilationError(
                f"Cannot compose: other has {other.num_qubits} qubits, "
                f"self has {self.num_qubits}."
            )
        for instr in other._instructions:
            self._instructions.append(instr)
        self._compiled = False
        return self

    def bind_parameters(self, params: dict[str, float] | list[float] | np.ndarray) -> QuantumCircuit:
        """
        Return a new circuit with parametric gate angles substituted.

        Args:
            params: Dict of {name: value} or an ordered list/array of values.

        Returns:
            New QuantumCircuit with all parameters bound.
        """
        bound = QuantumCircuit(self.num_qubits, self._creg.size, name=self._name + "_bound")
        if isinstance(params, (list, np.ndarray)):
            param_names = list(self._parameters.keys())
            params = {name: float(val) for name, val in zip(param_names, params)}
        bound._parameters = dict(params)
        for instr in self._instructions:
            if instr.operation == "gate" and instr.gate is not None:
                gate = instr.gate
                # Bind parameterized gates
                if hasattr(gate, "bind") and gate.is_parametric:
                    try:
                        gate = gate.bind(params)
                    except (KeyError, TypeError):
                        pass
                bound._instructions.append(
                    Instruction(operation="gate", gate=gate, qubits=instr.qubits)
                )
            else:
                bound._instructions.append(instr)
        return bound

    # ── Compilation & IR ──────────────────────────────────────────────────

    def compile(self) -> QuantumCircuit:
        """Validate circuit and build the cached QuantumIR."""
        validator = CircuitValidator(self)
        validator.validate()
        self._ir = self._build_ir()
        self._compiled = True
        return self

    def to_ir(self) -> Any:
        """Return the QuantumIR (compiles if needed)."""
        if not self._compiled or self._ir is None:
            self.compile()
        return self._ir

    def _build_ir(self) -> Any:
        from zuntenium.circuit.ir.ir_node import IRNode
        from zuntenium.circuit.ir.quantum_ir import QuantumIR
        nodes: list[IRNode] = []
        for instr in self._instructions:
            node = IRNode(
                operation=instr.operation,
                gate=instr.gate,
                qubits=list(instr.qubits),
                clbits=list(instr.clbits),
                params=list(instr.params),
            )
            nodes.append(node)
        return QuantumIR(
            num_qubits=self._qreg.size,
            num_clbits=self._creg.size,
            instructions=nodes,
            metadata={"name": self._name},
        )

    # ── Inversion & Mirror Circuits ────────────────────────────────────────

    def inverse(self) -> QuantumCircuit:
        """
        Return the inverse (Hermitian conjugate C†) of this circuit.
        Reverses instruction order and conjugates each gate.
        """
        inv_qc = QuantumCircuit(
            self._qreg.size, self._creg.size, name=f"{self._name}_dg"
        )
        for inst in reversed(self._instructions):
            if inst.operation == "gate" and inst.gate is not None:
                inv_qc._add_gate(inst.gate.dagger(), inst.qubits)
            elif inst.operation == "barrier":
                inv_qc.barrier(*inst.qubits)
        return inv_qc

    def mirror(self) -> QuantumCircuit:
        """
        Create a self-verifying mirror circuit: C followed by C†.
        On a noiseless simulator, always recovers |0...0⟩ with 100% fidelity.
        """
        mirror_qc = QuantumCircuit(
            self._qreg.size, self._creg.size, name=f"{self._name}_mirror"
        )
        for inst in self._instructions:
            if inst.operation == "gate" and inst.gate is not None:
                mirror_qc._add_gate(inst.gate, inst.qubits)
            elif inst.operation == "barrier":
                mirror_qc.barrier(*inst.qubits)
        for inst in reversed(self._instructions):
            if inst.operation == "gate" and inst.gate is not None:
                mirror_qc._add_gate(inst.gate.dagger(), inst.qubits)
            elif inst.operation == "barrier":
                mirror_qc.barrier(*inst.qubits)
        return mirror_qc

    # ── Visualization ──────────────────────────────────────────────────────

    def draw(self, style: str = "text") -> str:
        """Draw the circuit."""
        from zuntenium.circuit.visualization.text_drawer import TextDrawer
        return TextDrawer(self).draw()

    def __str__(self) -> str:
        return self.draw()

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter Notebook and Google Colab."""
        text = self.draw()
        return (
            f"<div style='font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; "
            f"background-color: #0d1117; color: #58a6ff; padding: 14px; border-radius: 8px; "
            f"border: 1px solid #30363d; overflow-x: auto; white-space: pre; line-height: 1.3;'>"
            f"{text}</div>"
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def num_qubits(self) -> int:
        return self._qreg.size

    @property
    def num_clbits(self) -> int:
        return self._creg.size

    @property
    def name(self) -> str:
        return self._name

    @property
    def depth(self) -> int:
        """Circuit depth (critical path length over qubit layers)."""
        if not self._instructions:
            return 0
        # Simple version: count layers respecting qubit parallelism
        latest: dict[int, int] = {}
        for instr in self._instructions:
            if instr.operation not in ("gate", "measure", "reset"):
                continue
            layer = max((latest.get(q, 0) for q in instr.qubits), default=0) + 1
            for q in instr.qubits:
                latest[q] = layer
        return max(latest.values()) if latest else 0

    @property
    def num_gates(self) -> int:
        return sum(1 for i in self._instructions if i.operation == "gate")

    @property
    def instructions(self) -> list[Instruction]:
        return list(self._instructions)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _add_gate(self, gate: BaseGate, qubits: list[int]) -> QuantumCircuit:
        for q in qubits:
            self._validate_qubit(q)
        self._instructions.append(
            Instruction(operation="gate", gate=gate, qubits=qubits, params=gate.params)
        )
        self._compiled = False
        return self

    def _validate_qubit(self, qubit: int) -> None:
        if not (0 <= qubit < self._qreg.size):
            raise QubitIndexError(qubit, self._qreg.size)

    def __repr__(self) -> str:
        return (
            f"QuantumCircuit(name='{self._name}', "
            f"num_qubits={self.num_qubits}, "
            f"num_gates={self.num_gates}, "
            f"depth={self.depth})"
        )
