"""
math_helpers — Bloch sphere coordinates, angle normalization, and related math.
"""
from __future__ import annotations

import math

import numpy as np


def bloch_vector(statevector: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the Bloch sphere vector (r_x, r_y, r_z) for a single-qubit state.

    Args:
        statevector: Complex array of shape (2,) representing |ψ⟩ = α|0⟩ + β|1⟩.

    Returns:
        (r_x, r_y, r_z) Bloch vector components.
    """
    psi = statevector.flatten()
    assert len(psi) == 2, "Bloch vector only defined for single-qubit states."

    rho = np.outer(psi, psi.conj())
    r_x = 2 * float(np.real(rho[0, 1]))
    r_y = 2 * float(np.imag(rho[1, 0]))
    r_z = float(np.real(rho[0, 0] - rho[1, 1]))
    return r_x, r_y, r_z


def normalize_angle(theta: float) -> float:
    """Normalize an angle to the range (-π, π]."""
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta -= 2 * math.pi
    return theta


def angles_close(a: float, b: float, tol: float = 1e-10) -> bool:
    """Return True if two angles are equivalent (mod 2π) within tolerance."""
    return abs(normalize_angle(a - b)) < tol


def spherical_to_cartesian(theta: float, phi: float) -> tuple[float, float, float]:
    """Convert spherical angles (θ,φ) on the Bloch sphere to Cartesian (x,y,z)."""
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return x, y, z
