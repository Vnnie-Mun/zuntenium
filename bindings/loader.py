"""
loader.py — C++ extension loader with graceful NumPy fallback.

Attempts to import ``_zuntenium_core`` (compiled pybind11 module).
If unavailable (build not yet run), falls back to pure NumPy implementations.
"""
from __future__ import annotations

import warnings

_CPP_AVAILABLE: bool = False
_zuntenium_core: object = None


def _load_cpp_extension() -> bool:
    """Try to import the compiled C++ extension. Returns True on success."""
    global _CPP_AVAILABLE, _zuntenium_core
    try:
        import _zuntenium_core as _core  # type: ignore[import]
        _zuntenium_core = _core
        _CPP_AVAILABLE = True
        return True
    except ImportError:
        return False


# Attempt load at module import time
_load_cpp_extension()


def is_cpp_available() -> bool:
    """Return True if the compiled C++ extension is available."""
    return _CPP_AVAILABLE


def get_statevector_module() -> object:
    """
    Return the best available statevector implementation module.

    Returns the C++ extension if available, otherwise the NumPy fallback.
    """
    if _CPP_AVAILABLE:
        return _zuntenium_core
    from zuntenium.quantum_core.statevector import statevector_numpy
    return statevector_numpy


def warn_if_numpy_fallback() -> None:
    """Issue a performance warning if the C++ extension is not available."""
    if not _CPP_AVAILABLE:
        warnings.warn(
            "ZUNTENIUM C++ extension not found. "
            "Running on pure NumPy backend (slower for large circuits). "
            "Build extension with: make build-bindings",
            RuntimeWarning,
            stacklevel=3,
        )
