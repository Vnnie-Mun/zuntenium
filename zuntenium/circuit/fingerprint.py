"""
Circuit Fingerprinting — Structural hashing and identity signatures.

Provides multiple fingerprinting strategies for quantum circuits:
  1. Structural hash: order-sensitive hash of the gate sequence.
  2. Gate-count signature: order-invariant feature vector.
  3. Topology fingerprint: connectivity-invariant hash.

Used for circuit deduplication, cache lookup, and training data indexing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


@dataclass(frozen=True)
class CircuitFingerprint:
    """
    Immutable fingerprint of a quantum circuit.

    Attributes:
        structural_hash:   SHA-256 of the ordered gate sequence.
        gate_signature:    Sorted dict of {gate_name: count}.
        topology_hash:     Hash of the qubit connectivity graph.
        num_qubits:        Number of qubits.
        depth:             Circuit depth.
        gate_count:        Total number of gates.
        feature_vector:    Numeric feature vector for ML indexing.
    """
    structural_hash: str
    gate_signature: dict[str, int]
    topology_hash: str
    num_qubits: int
    depth: int
    gate_count: int
    feature_vector: tuple[float, ...] = ()

    def matches(self, other: CircuitFingerprint) -> bool:
        """Check if two fingerprints represent the same circuit."""
        return self.structural_hash == other.structural_hash

    def similar_to(self, other: CircuitFingerprint, threshold: float = 0.9) -> bool:
        """
        Check if two circuits are structurally similar using
        gate-count cosine similarity.
        """
        if not self.feature_vector or not other.feature_vector:
            return self.gate_signature == other.gate_signature
        a = np.array(self.feature_vector)
        b = np.array(other.feature_vector)
        if len(a) != len(b):
            # Pad shorter to match
            max_len = max(len(a), len(b))
            a = np.pad(a, (0, max_len - len(a)))
            b = np.pad(b, (0, max_len - len(b)))
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return False
        cosine = float(np.dot(a, b) / (norm_a * norm_b))
        return cosine >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "structural_hash": self.structural_hash,
            "gate_signature": self.gate_signature,
            "topology_hash": self.topology_hash,
            "num_qubits": self.num_qubits,
            "depth": self.depth,
            "gate_count": self.gate_count,
            "feature_vector": list(self.feature_vector),
        }

    def __repr__(self) -> str:
        return (
            f"CircuitFingerprint(hash={self.structural_hash[:12]}..., "
            f"n={self.num_qubits}, depth={self.depth}, gates={self.gate_count})"
        )


# ── Canonical gate vocabulary for feature vector ──────────────────────

_CANONICAL_GATES = [
    "H", "X", "Y", "Z", "S", "T", "SDG", "TDG",
    "RX", "RY", "RZ", "U1", "U2", "U3",
    "CX", "CZ", "CY", "SWAP", "ISWAP",
    "CRX", "CRY", "CRZ",
    "CCX", "CSWAP",
    "MEASURE", "RESET", "BARRIER",
]


# ── Fingerprinting from QuantumIR ─────────────────────────────────────

class CircuitFingerprinter:
    """
    Generates fingerprints for quantum circuits.

    Supports both QuantumIR and QuantumCircuit inputs.
    """

    @staticmethod
    def _structural_hash_ir(ir: QuantumIR) -> str:
        """
        Compute SHA-256 of the ordered gate sequence.

        Encodes each instruction as (gate_name, qubits, rounded_params).
        """
        hasher = hashlib.sha256()
        hasher.update(f"n={ir.num_qubits}".encode())

        for node in ir.instructions:
            if node.is_gate() and node.gate is not None:
                name = node.gate.name
                qubits = tuple(node.qubits)
                # Round params to 6 decimal places for float stability
                params = tuple(round(p, 6) for p in node.params)
                token = f"{name}|{qubits}|{params}"
            else:
                token = f"{node.operation}|{tuple(node.qubits)}"
            hasher.update(token.encode())

        return hasher.hexdigest()

    @staticmethod
    def _gate_signature_ir(ir: QuantumIR) -> dict[str, int]:
        """Count of each gate type (order-invariant)."""
        counts: dict[str, int] = {}
        for node in ir.instructions:
            if node.is_gate() and node.gate is not None:
                name = node.gate.name.upper()
            else:
                name = node.operation.upper()
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _topology_hash_ir(ir: QuantumIR) -> str:
        """
        Hash the qubit connectivity graph (which pairs of qubits interact).

        This is invariant to gate ordering — only the set of
        interacting qubit pairs matters.
        """
        edges: set[tuple[int, int]] = set()
        for node in ir.instructions:
            if node.is_gate() and len(node.qubits) >= 2:
                qubits = sorted(node.qubits)
                for i in range(len(qubits)):
                    for j in range(i + 1, len(qubits)):
                        edges.add((qubits[i], qubits[j]))

        # Also include single-qubit gate presence per qubit
        qubit_has_gate: set[int] = set()
        for node in ir.instructions:
            if node.is_gate():
                qubit_has_gate.update(node.qubits)

        hasher = hashlib.sha256()
        hasher.update(f"n={ir.num_qubits}".encode())
        hasher.update(f"edges={sorted(edges)}".encode())
        hasher.update(f"active={sorted(qubit_has_gate)}".encode())
        return hasher.hexdigest()

    @staticmethod
    def _feature_vector_ir(ir: QuantumIR) -> tuple[float, ...]:
        """
        Compute a numeric feature vector for ML indexing.

        Components:
          [0]     num_qubits (normalized)
          [1]     depth (normalized)
          [2]     total_gates (normalized)
          [3..3+K] gate counts for each canonical gate type
          [K+3]   entanglement_ratio (fraction of multi-qubit gates)
          [K+4]   parametric_ratio (fraction of parametric gates)
          [K+5]   connectivity_density
        """
        gate_counts: dict[str, int] = {}
        total_gates = 0
        multi_qubit = 0
        parametric = 0

        for node in ir.instructions:
            if not node.is_gate():
                continue
            total_gates += 1
            name = node.gate.name.upper() if node.gate else "UNKNOWN"
            gate_counts[name] = gate_counts.get(name, 0) + 1
            if len(node.qubits) >= 2:
                multi_qubit += 1
            if node.params:
                parametric += 1

        # Connectivity density
        edges: set[tuple[int, int]] = set()
        for node in ir.instructions:
            if node.is_gate() and len(node.qubits) >= 2:
                qs = sorted(node.qubits)
                for i in range(len(qs)):
                    for j in range(i + 1, len(qs)):
                        edges.add((qs[i], qs[j]))
        max_edges = ir.num_qubits * (ir.num_qubits - 1) / 2
        conn_density = len(edges) / max(max_edges, 1)

        features: list[float] = [
            float(ir.num_qubits),
            float(ir.depth()),
            float(total_gates),
        ]
        # Gate counts in canonical order
        for gate_name in _CANONICAL_GATES:
            features.append(float(gate_counts.get(gate_name, 0)))

        features.append(multi_qubit / max(total_gates, 1))
        features.append(parametric / max(total_gates, 1))
        features.append(conn_density)

        return tuple(features)

    @classmethod
    def fingerprint_ir(cls, ir: QuantumIR) -> CircuitFingerprint:
        """Generate a full fingerprint from a QuantumIR."""
        return CircuitFingerprint(
            structural_hash=cls._structural_hash_ir(ir),
            gate_signature=cls._gate_signature_ir(ir),
            topology_hash=cls._topology_hash_ir(ir),
            num_qubits=ir.num_qubits,
            depth=ir.depth(),
            gate_count=ir.num_gates(),
            feature_vector=cls._feature_vector_ir(ir),
        )

    @classmethod
    def fingerprint_circuit(cls, circuit: QuantumCircuit) -> CircuitFingerprint:
        """Generate a fingerprint from a QuantumCircuit (converts to IR first)."""
        ir = circuit.to_ir()
        return cls.fingerprint_ir(ir)


# ── Convenience functions ─────────────────────────────────────────────

def fingerprint(circuit_or_ir: Any) -> CircuitFingerprint:
    """
    Generate a fingerprint from either a QuantumCircuit or QuantumIR.

    Example::

        from zuntenium.circuit.fingerprint import fingerprint

        fp = fingerprint(my_circuit)
        print(fp.structural_hash)
        print(fp.gate_signature)
    """
    from zuntenium.circuit.ir.quantum_ir import QuantumIR

    if isinstance(circuit_or_ir, QuantumIR):
        return CircuitFingerprinter.fingerprint_ir(circuit_or_ir)
    else:
        # Assume it's a QuantumCircuit
        return CircuitFingerprinter.fingerprint_circuit(circuit_or_ir)


def circuits_equivalent(a: Any, b: Any) -> bool:
    """Check if two circuits have identical structural fingerprints."""
    return fingerprint(a).matches(fingerprint(b))


def circuits_similar(a: Any, b: Any, threshold: float = 0.9) -> bool:
    """Check if two circuits are structurally similar."""
    return fingerprint(a).similar_to(fingerprint(b), threshold)


__all__ = [
    "CircuitFingerprint",
    "CircuitFingerprinter",
    "fingerprint",
    "circuits_equivalent",
    "circuits_similar",
]
