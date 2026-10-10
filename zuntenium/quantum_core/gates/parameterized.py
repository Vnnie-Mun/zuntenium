"""
ParameterizedGate — Gate with symbolic/runtime parameters.

Used by VQE/QAOA ansätze where gate angles are optimization variables.
Parameters can be floats (bound) or string symbols (unbound).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Union

import numpy as np

from zuntenium.quantum_core.gates.base_gate import BaseGate

Param = Union[float, str]   # float = bound, str = symbolic


class ParameterizedGate(BaseGate):
    """
    A single-qubit rotation gate with named symbolic parameters.

    Wraps an underlying gate factory, allowing parameter binding at runtime.

    Args:
        name: Gate name (e.g. "Rx", "Ry", "Rz").
        param_names: Ordered list of symbolic parameter names.
        gate_factory: Callable(params: list[float]) → BaseGate.

    Example::

        pg = ParameterizedGate("Rx", ["theta"], lambda p: RxGate(p[0]))
        bound = pg.bind({"theta": 0.5})
        bound.matrix()  # Rx(0.5) matrix
    """

    is_parametric = True

    def __init__(
        self,
        name: str,
        param_names: list[str],
        gate_factory: Callable[[list[float]], BaseGate],
    ) -> None:
        self.name = name
        self.num_qubits = 1
        self._param_names = param_names
        self._gate_factory = gate_factory
        self._bound_values: dict[str, float] = {}

    def bind(self, params: dict[str, float]) -> BaseGate:
        """
        Return a concrete gate with the given parameter values substituted.

        Args:
            params: Mapping of parameter name → float value.

        Returns:
            A concrete BaseGate with all parameters resolved.

        Raises:
            KeyError: If a required parameter is missing from ``params``.
        """
        values = [params[name] for name in self._param_names]
        return self._gate_factory(values)

    def matrix(self) -> np.ndarray:
        """Return matrix with current bound values (raises if unbound)."""
        if len(self._bound_values) < len(self._param_names):
            unbound = [n for n in self._param_names if n not in self._bound_values]
            raise ValueError(
                f"ParameterizedGate '{self.name}' has unbound parameters: {unbound}. "
                "Call .bind(params) first."
            )
        values = [self._bound_values[n] for n in self._param_names]
        return self._gate_factory(values).matrix()

    @property
    def param_names(self) -> list[str]:
        return list(self._param_names)

    @property
    def params(self) -> list[float]:
        return [self._bound_values.get(n, float("nan")) for n in self._param_names]

    def __repr__(self) -> str:
        param_str = ", ".join(
            f"{n}={self._bound_values.get(n, '?')}"
            for n in self._param_names
        )
        return f"ParameterizedGate({self.name}, [{param_str}])"
