"""Parameter type and range validation."""
from __future__ import annotations

import math


def validate_angle(value: float, name: str = "angle") -> float:
    """Validate a rotation angle parameter. Returns the value if valid."""
    if not math.isfinite(value):
        raise ValueError(f"Parameter '{name}' must be finite, got {value}.")
    return float(value)


def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a parameter is positive."""
    if value <= 0:
        raise ValueError(f"Parameter '{name}' must be positive, got {value}.")
    return float(value)


def validate_probability(value: float, name: str = "probability") -> float:
    """Validate that a parameter is in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"Parameter '{name}' must be in [0, 1], got {value}."
        )
    return float(value)
