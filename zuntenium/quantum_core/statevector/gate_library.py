"""
zuntenium/quantum_core/statevector/gate_library.py
===================================================
Factory class for all standard quantum gates.

Extracted from statevector_simulator.py as part of O6 God-class decomposition.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Matrix2x2 = np.ndarray   # shape (2, 2),   dtype complex128
MatrixNxN = np.ndarray    # shape (N, N),   dtype complex128


class GateLibrary:
    """
    Factory class for all standard quantum gates.

    Every method returns a np.complex128 ndarray of the appropriate shape.
    Matrices are constructed once and cached as class-level constants where
    they are parameter-independent.

    Convention
    ----------
    Matrices are in the standard computational basis {|0⟩, |1⟩} for
    single-qubit gates, and {|00⟩,|01⟩,|10⟩,|11⟩} (little-endian) for
    two-qubit gates.
    """

    # ------------------------------------------------------------------
    # Cached parameter-free single-qubit gates
    # ------------------------------------------------------------------
    #
    # Pauli-X (bit-flip / NOT):
    #   X = [[0, 1],   maps |0⟩→|1⟩, |1⟩→|0⟩
    #        [1, 0]]
    X: Matrix2x2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    # Pauli-Y:
    #   Y = [[0, -i],  maps |0⟩→i|1⟩, |1⟩→-i|0⟩
    #        [i,  0]]
    Y: Matrix2x2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    # Pauli-Z (phase-flip):
    #   Z = [[1,  0],  maps |0⟩→|0⟩, |1⟩→-|1⟩
    #        [0, -1]]
    Z: Matrix2x2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Hadamard:
    #   H = (1/√2) [[1,  1],   maps |0⟩→|+⟩, |1⟩→|−⟩
    #               [1, -1]]
    H: Matrix2x2 = np.array(
        [[1, 1], [1, -1]], dtype=np.complex128
    ) / math.sqrt(2)

    # S gate (π/2 phase):
    #   S = [[1, 0],   |1⟩ gains phase i
    #        [0, i]]
    S: Matrix2x2 = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    # T gate (π/4 phase):
    #   T = [[1, 0],         |1⟩ gains phase e^{iπ/4}
    #        [0, e^{iπ/4}]]
    T: Matrix2x2 = np.array(
        [[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128
    )

    # S† and T† (adjoints) for gate cancellation pass
    Sdg: Matrix2x2 = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    Tdg: Matrix2x2 = np.array(
        [[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128
    )

    # ------------------------------------------------------------------
    # Cached two-qubit gates (little-endian ordering)
    #
    # CNOT: control=qubit-0 (outer index), target=qubit-1 (inner index)
    #   |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
    # ------------------------------------------------------------------

    CNOT: MatrixNxN = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]], dtype=np.complex128
    )
    # SWAP: exchanges two qubit states
    #   |00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→|11⟩
    SWAP: MatrixNxN = np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]], dtype=np.complex128
    )

    # CZ: controlled-Z
    #   |11⟩ gains phase -1, others unchanged
    CZ: MatrixNxN = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, -1]], dtype=np.complex128
    )

    # iSWAP: combines SWAP with relative phase
    #   |01⟩→i|10⟩, |10⟩→i|01⟩
    ISWAP: MatrixNxN = np.array(
        [[1, 0,  0,  0],
         [0, 0,  1j, 0],
         [0, 1j, 0,  0],
         [0, 0,  0,  1]], dtype=np.complex128
    )

    # Toffoli / CCNOT (3-qubit, 8×8)
    # Controls: qubits 0,1 (outer indices). Target: qubit 2 (inner index).
    # |110⟩ ↔ |111⟩ (indices 6 ↔ 7); all others unchanged.
    TOFFOLI: MatrixNxN = np.eye(8, dtype=np.complex128)


    @classmethod
    def _build_toffoli(cls) -> None:
        M = np.eye(8, dtype=np.complex128)
        M[6, 6] = 0; M[7, 7] = 0
        M[6, 7] = 1; M[7, 6] = 1
        cls.TOFFOLI = M

    @staticmethod
    def rx(theta: float) -> Matrix2x2:
        """
        RX(θ) = exp(−iθX/2) = [[cos(θ/2),   −i sin(θ/2)],
                                 [−i sin(θ/2), cos(θ/2)  ]]

        Physical meaning: rotation by angle θ about the X-axis on the Bloch sphere.
        Numerical note: cos and sin are IEEE-754 correctly-rounded, so no stability
        issues for any θ.
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c,       -1j * s],
             [-1j * s,  c     ]], dtype=np.complex128
        )

    @staticmethod
    def ry(theta: float) -> Matrix2x2:
        """
        RY(θ) = exp(−iθY/2) = [[cos(θ/2), −sin(θ/2)],
                                 [sin(θ/2),  cos(θ/2)]]

        All entries are real — numerically this is the cleanest rotation gate.
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c, -s],
             [s,  c]], dtype=np.complex128
        )

    @staticmethod
    def rz(theta: float) -> Matrix2x2:
        """
        RZ(θ) = exp(−iθZ/2) = [[e^{−iθ/2}, 0        ],
                                 [0,          e^{+iθ/2}]]

        Note: RZ is *diagonal* — applying it costs only 2 complex multiplications
        (two scalar mults of alternating slices), much cheaper than general matmul.
        The general apply_gate path handles this correctly, but an optimized
        diagonal path is a future micro-optimization.
        """
        half = theta / 2
        return np.array(
            [[np.exp(-1j * half), 0                ],
             [0,                   np.exp(1j * half)]], dtype=np.complex128
        )

    @staticmethod
    def phase(phi: float) -> Matrix2x2:
        """
        P(φ) = [[1, 0         ],
                [0, e^{iφ}   ]]

        Also called the Phase gate or U1(φ).  S = P(π/2), T = P(π/4), Z = P(π).
        """
        return np.array(
            [[1, 0             ],
             [0, np.exp(1j*phi)]], dtype=np.complex128
        )

    @staticmethod
    def u3(theta: float, phi: float, lam: float) -> Matrix2x2:
        """
        U3(θ,φ,λ) — the most general single-qubit gate (up to global phase).
        Defined as in IBM/OpenQASM convention:

            U3 = [[cos(θ/2),              −e^{iλ} sin(θ/2)     ],
                  [e^{iφ} sin(θ/2),        e^{i(φ+λ)} cos(θ/2)]]

        Any 2×2 unitary U (up to global phase) can be written as U3(θ,φ,λ).
        This is used by the SingleQubitUnify transpiler pass.
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c,                        -np.exp(1j * lam) * s],
             [np.exp(1j * phi) * s,      np.exp(1j * (phi + lam)) * c]],
            dtype=np.complex128
        )

    @staticmethod
    def controlled(base_matrix: Matrix2x2) -> MatrixNxN:
        """
        Construct the controlled-U gate from any single-qubit matrix U.

        Mathematical construction:
            CU = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
               = [[1, 0, 0,    0   ],
                  [0, 1, 0,    0   ],
                  [0, 0, U[0,0],U[0,1]],
                  [0, 0, U[1,0],U[1,1]]]

        This pattern extends: CCU (Toffoli-like) uses nested controlled() calls.

        Parameters
        ----------
        base_matrix : 2×2 unitary

        Returns
        -------
        4×4 controlled-U matrix in little-endian basis
        """
        U = base_matrix
        M = np.zeros((4, 4), dtype=np.complex128)
        M[0, 0] = 1.0; M[1, 1] = 1.0   # |0⟩⟨0| ⊗ I block
        M[2:4, 2:4] = U                  # |1⟩⟨1| ⊗ U block
        return M

    # ------------------------------------------------------------------
    # Gate registry (name → matrix factory)
    # ------------------------------------------------------------------

    _STATIC_GATES: Dict[str, MatrixNxN] = {}   # filled after class body

    @classmethod
    def get(cls, name: str, **params) -> MatrixNxN:
        """
        Retrieve a gate matrix by name, with optional parameters.

        Examples
        --------
        GateLibrary.get('X')
        GateLibrary.get('RX', theta=np.pi/2)
        GateLibrary.get('U3', theta=0.5, phi=0.1, lam=0.2)
        GateLibrary.get('CNOT')
        """
        name_upper = name.upper()
        if name_upper in cls._STATIC_GATES:
            return cls._STATIC_GATES[name_upper]
        param_dispatch = {
            'RX':    lambda: cls.rx(**params),
            'RY':    lambda: cls.ry(**params),
            'RZ':    lambda: cls.rz(**params),
            'PHASE': lambda: cls.phase(**params),
            'U3':    lambda: cls.u3(**params),
        }
        if name_upper in param_dispatch:
            return param_dispatch[name_upper]()
        raise ValueError(
            f"Unknown gate '{name}'. "
            f"Available static gates: {sorted(cls._STATIC_GATES.keys())}. "
            "Parameterized: RX, RY, RZ, PHASE, U3."
        )


# Populate static registry after class definition
GateLibrary._build_toffoli()
GateLibrary._STATIC_GATES = {
    'X':       GateLibrary.X,
    'Y':       GateLibrary.Y,
    'Z':       GateLibrary.Z,
    'H':       GateLibrary.H,
    'S':       GateLibrary.S,
    'T':       GateLibrary.T,
    'SDG':     GateLibrary.Sdg,
    'TDG':     GateLibrary.Tdg,
    'CNOT':    GateLibrary.CNOT,
    'CX':      GateLibrary.CNOT,
    'SWAP':    GateLibrary.SWAP,
    'CZ':      GateLibrary.CZ,
    'ISWAP':   GateLibrary.ISWAP,
    'TOFFOLI': GateLibrary.TOFFOLI,
    'CCNOT':   GateLibrary.TOFFOLI,
}
