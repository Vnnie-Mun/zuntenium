"""utils/linalg — Matrix operations and decompositions."""
from __future__ import annotations

from zuntenium.utils.linalg.decomposition import kak_decompose, zyz_decompose_sv
from zuntenium.utils.linalg.matrix_ops import (
    anticommutator,
    commutator,
    is_hermitian,
    is_unitary,
    matrix_exponential,
    partial_trace,
    tensor_product,
)

__all__ = [
    "tensor_product", "partial_trace", "matrix_exponential",
    "commutator", "anticommutator", "is_unitary", "is_hermitian",
    "zyz_decompose_sv", "kak_decompose",
]
