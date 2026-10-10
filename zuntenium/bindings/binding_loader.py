"""
binding_loader.py — Resolves and loads the _zuntenium_core C++ extension.

The compiled .so is installed into zuntenium/ (same package dir).
This module handles the import path resolution and provides a clean API
for the rest of the framework to use.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

_CPP_AVAILABLE: bool = False
_cpp_module: object | None = None


def _try_load() -> bool:
    """
    Attempt to import _zuntenium_core.

    Search order:
    1. Already in sys.modules (fastest path after first load)
    2. Standard import (works if the .so is on sys.path)
    3. Direct path from package directory (zuntenium/_zuntenium_core*.so)
    """
    global _CPP_AVAILABLE, _cpp_module

    if "zuntenium._zuntenium_core" in sys.modules:
        _cpp_module = sys.modules["zuntenium._zuntenium_core"]
        _CPP_AVAILABLE = True
        return True

    # Try standard import first
    try:
        import _zuntenium_core as _core  # type: ignore[import-untyped]
        _cpp_module = _core
        sys.modules["zuntenium._zuntenium_core"] = _core
        _CPP_AVAILABLE = True
        return True
    except ImportError:
        pass

    # Try loading from the package directory (where build_bindings.sh copies the .so)
    pkg_dir = Path(__file__).parent  # zuntenium/bindings/ → go up one
    root_dir = pkg_dir.parent        # zuntenium/

    for so in sorted(root_dir.glob("_zuntenium_core*.so")):
        spec = importlib.util.spec_from_file_location("_zuntenium_core", so)
        if spec and spec.loader:
            try:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _cpp_module = mod
                sys.modules["_zuntenium_core"] = mod
                sys.modules["zuntenium._zuntenium_core"] = mod
                _CPP_AVAILABLE = True
                return True
            except Exception:
                continue

    return False


# Run at module import
_try_load()


def is_cpp_available() -> bool:
    """Return True if the compiled _zuntenium_core C++ extension is loaded."""
    return _CPP_AVAILABLE


def load_cpp_module() -> Any:
    """
    Return the _zuntenium_core module if available, else None.

    The returned module exposes:
      - module.StateVector(num_qubits)  → C++ StateVectorCPP instance
      - module.gates.h_matrix() etc.
      - module.is_cpp_available()
    """
    return _cpp_module if _CPP_AVAILABLE else None


def get_statevector_module() -> object:
    """
    Return the best available statevector implementation module.

    Returns the C++ extension module if compiled, otherwise the NumPy fallback.
    """
    if _CPP_AVAILABLE and _cpp_module is not None:
        return _cpp_module
    from zuntenium.quantum_core.statevector import statevector_numpy
    return statevector_numpy


def warn_if_numpy_fallback() -> None:
    """Issue a performance warning if the C++ extension is not available."""
    if not _CPP_AVAILABLE:
        warnings.warn(
            "ZUNTENIUM C++ extension not found. "
            "Running on pure NumPy backend (slower for large circuits). "
            "Build with: make build-bindings",
            RuntimeWarning,
            stacklevel=3,
        )


__all__ = [
    "is_cpp_available",
    "load_cpp_module",
    "get_statevector_module",
    "warn_if_numpy_fallback",
]
