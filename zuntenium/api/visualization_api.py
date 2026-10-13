"""
visualization_api.py — High-level visualization functions.

Bloch sphere and state city plot for single/multi-qubit state inspection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from zuntenium.quantum_core.executor.result import ExecutionResult


def plot_histogram(result: ExecutionResult | dict[str, int], **kwargs: Any) -> None:
    """Plot measurement count histogram (accepts ExecutionResult or counts dict)."""
    if isinstance(result, dict):
        from zuntenium.quantum_core.executor.result import ExecutionResult

        shots = sum(result.values())
        first_key = next(iter(result), "")
        n_qubits = len(first_key) if first_key else 0
        res = ExecutionResult(counts=result, num_qubits=n_qubits, shots=shots)
        res.plot_histogram(**kwargs)
    elif hasattr(result, "plot_histogram"):
        result.plot_histogram(**kwargs)
    else:
        raise TypeError(f"Expected ExecutionResult or dict, got {type(result).__name__}")


def plot_bloch_sphere(
    statevector: np.ndarray,
    qubit: int = 0,
    title: str = "Bloch Sphere",
    **kwargs: Any,
) -> None:
    """
    Plot the Bloch sphere representation of a single-qubit state.

    For multi-qubit statevectors, traces out all other qubits first to get
    the reduced single-qubit density matrix, then computes the Bloch vector.

    Args:
        statevector: Complex amplitude array for n qubits.
        qubit: Which qubit to visualise (default 0).
        title: Plot title.
        **kwargs: Passed to matplotlib.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        raise ImportError("Install matplotlib: pip install matplotlib") from None

    sv = np.asarray(statevector, dtype=np.complex128)
    n = int(round(np.log2(len(sv))))

    # Get reduced density matrix for the requested qubit
    from zuntenium.quantum_core.statevector.density_matrix import DensityMatrix
    dm = DensityMatrix.from_amplitudes(sv)
    if n > 1:
        dm = dm.partial_trace([qubit])
    rho = dm.matrix

    # Bloch vector: r = [tr(Xρ), tr(Yρ), tr(Zρ)]
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    rx = float(np.real(np.trace(X @ rho)))
    ry = float(np.real(np.trace(Y @ rho)))
    rz = float(np.real(np.trace(Z @ rho)))

    # Draw sphere
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(60), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.08, color="#6366f1")   # transparent sphere
    ax.plot_wireframe(xs, ys, zs, alpha=0.05, color="#6366f1", linewidth=0.3)

    # Axes
    for start, end, label in [
        ([-1.3, 0, 0], [1.3, 0, 0], "+X"),
        ([0, -1.3, 0], [0, 1.3, 0], "+Y"),
        ([0, 0, -1.3], [0, 0, 1.3], "+Z"),
    ]:
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color="gray", linewidth=0.8, linestyle="--")
        ax.text(end[0]*1.1, end[1]*1.1, end[2]*1.1, label, fontsize=9, color="gray")

    # Bloch vector arrow
    ax.quiver(0, 0, 0, rx, ry, rz, color="#f43f5e", linewidth=2.5, arrow_length_ratio=0.15)
    ax.scatter([rx], [ry], [rz], color="#f43f5e", s=50, zorder=5)

    # Labels
    ax.text(0, 0,  1.35, "|0⟩", ha="center", fontsize=11)
    ax.text(0, 0, -1.35, "|1⟩", ha="center", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_state_city(
    statevector: np.ndarray,
    title: str = "State City",
    show_phase: bool = True,
    **kwargs: Any,
) -> None:
    """
    Plot a 3D 'city' bar chart of state amplitudes.

    Each basis state gets a bar whose height = |amplitude|² and whose
    colour encodes the phase angle arg(amplitude).

    Args:
        statevector: Complex amplitude array.
        title: Plot title.
        show_phase: If True, colour bars by phase. If False, uniform colour.
    """
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        raise ImportError("Install matplotlib: pip install matplotlib") from None

    sv = np.asarray(statevector, dtype=np.complex128)
    dim = len(sv)
    n = int(round(np.log2(dim)))

    probs = np.abs(sv) ** 2
    phases = np.angle(sv)  # in [-π, π]

    labels = [format(i, f"0{n}b") for i in range(dim)]
    x = np.arange(dim)

    # Choose colours based on phase
    if show_phase:
        norm_phase = (phases + np.pi) / (2 * np.pi)  # [0, 1]
        colors = cm.hsv(norm_phase)
    else:
        colors = ["#6366f1"] * dim

    fig = plt.figure(figsize=(max(8, dim // 4), 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar(x, probs, zs=0, zdir="y", width=0.7, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(6, 9 - n))
    ax.set_xlabel("Basis state |x⟩", labelpad=10)
    ax.set_ylabel("")
    ax.set_zlabel("Probability |α|²")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.show()
