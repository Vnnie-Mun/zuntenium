"""utils/validation — Gate and circuit validation helpers."""
from __future__ import annotations

from zuntenium.utils.validation.gate_validator import check_dimension, check_unitarity

__all__ = ["check_unitarity", "check_dimension"]
