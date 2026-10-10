"""
QuantumRegister and ClassicalRegister.

Registers are named collections of qubit or classical bit indices.
They are used by QuantumCircuit for addressing and visualization.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantumRegister:
    """
    A named register of ``size`` qubits.

    Indices are contiguous integers starting from ``start``.

    Args:
        size: Number of qubits in this register.
        name: Optional symbolic name (default: "q").
        start: Starting qubit index (default 0).
    """
    size: int
    name: str = "q"
    start: int = 0

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"QuantumRegister size must be >= 1, got {self.size}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        """Return the absolute qubit index for a register-relative index."""
        if not (0 <= idx < self.size):
            raise IndexError(f"Register '{self.name}' index {idx} out of range [0, {self.size})")
        return self.start + idx

    def indices(self) -> list[int]:
        """Return all absolute qubit indices in this register."""
        return list(range(self.start, self.start + self.size))

    def __repr__(self) -> str:
        return f"QuantumRegister(name='{self.name}', size={self.size}, start={self.start})"


@dataclass
class ClassicalRegister:
    """
    A named register of ``size`` classical bits.

    Args:
        size: Number of classical bits.
        name: Optional symbolic name (default: "c").
        start: Starting bit index (default 0).
    """
    size: int
    name: str = "c"
    start: int = 0

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"ClassicalRegister size must be >= 0, got {self.size}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        if not (0 <= idx < self.size):
            raise IndexError(f"Register '{self.name}' index {idx} out of range [0, {self.size})")
        return self.start + idx

    def indices(self) -> list[int]:
        return list(range(self.start, self.start + self.size))

    def __repr__(self) -> str:
        return f"ClassicalRegister(name='{self.name}', size={self.size}, start={self.start})"
