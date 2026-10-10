"""VQE/QAOA high-level API."""
from __future__ import annotations

from typing import Any

import numpy as np


def run_vqe(
    ansatz: object,
    hamiltonian: object,
    initial_params: list[float] | np.ndarray,
    backend: object | None = None,
    shots: int = 1024,
    max_iter: int = 300,
    **kwargs: Any,
) -> object:
    """
    Run VQE and return a HybridResult.

    Convenience wrapper around VQE class.
    """
    from zuntenium.hybrid import VQE
    vqe = VQE(
        ansatz=ansatz,  # type: ignore[arg-type]
        hamiltonian=hamiltonian,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        shots=shots,
        **kwargs,
    )
    return vqe.run(np.asarray(initial_params), max_iter=max_iter)


def run_qaoa(
    hamiltonian: object,
    p: int = 1,
    initial_params: list[float] | np.ndarray | None = None,
    backend: object | None = None,
    shots: int = 1024,
    max_iter: int = 300,
) -> object:
    """
    Run QAOA and return a HybridResult.

    Convenience wrapper around QAOA class.
    """
    from zuntenium.hybrid import QAOA
    qaoa = QAOA(
        hamiltonian=hamiltonian,  # type: ignore[arg-type]
        p=p,
        backend=backend,  # type: ignore[arg-type]
        shots=shots,
    )
    init = np.zeros(2 * p) if initial_params is None else np.asarray(initial_params)
    return qaoa.run(init, max_iter=max_iter)
