"""
Type stubs for the future _zuntenium_rs Rust extension module (v0.3+).

This file is a placeholder. The Rust extension will provide:
- High-performance tensor network simulation (MPS)
- GPU-accelerated gate operations via wgpu
- Memory-safe parallel execution engine
"""
from __future__ import annotations

__version__: str

def is_rust_available() -> bool:
    """Return True if the Rust extension is compiled and loadable."""
    ...
