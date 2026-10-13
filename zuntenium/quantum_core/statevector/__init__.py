"""statevector sub-package.

O6 Decomposition: Classes are now organized in focused modules:
- simulator_backend.py   — SimulatorBackend (ABC)
- statevector_state.py   — StatevectorState
- gate_library.py        — GateLibrary
- gate_applicator.py     — GateApplicator
- sim_circuit.py         — _SimInstruction, SimCircuit
- measurement_engine.py  — MeasurementEngine
- statevector_simulator.py — StatevectorSimulator (facade + re-exports)
"""
from __future__ import annotations

from zuntenium.quantum_core.statevector.statevector import StateVector
from zuntenium.quantum_core.statevector.simulator_backend import SimulatorBackend
from zuntenium.quantum_core.statevector.statevector_state import StatevectorState
from zuntenium.quantum_core.statevector.gate_library import GateLibrary
from zuntenium.quantum_core.statevector.gate_applicator import GateApplicator
from zuntenium.quantum_core.statevector.sim_circuit import SimCircuit
from zuntenium.quantum_core.statevector.measurement_engine import MeasurementEngine
from zuntenium.quantum_core.statevector.statevector_simulator import (
    StatevectorSimulator,
    create_simulator,
)

# Backward-compatibility alias (was previously exported as SimulatorQuantumCircuit)
SimulatorQuantumCircuit = SimCircuit

__all__ = [
    "StateVector",
    "SimulatorBackend",
    "StatevectorSimulator",
    "StatevectorState",
    "GateLibrary",
    "GateApplicator",
    "MeasurementEngine",
    "SimCircuit",
    "SimulatorQuantumCircuit",  # backward compat
    "create_simulator",
]
