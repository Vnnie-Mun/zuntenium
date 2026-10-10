"""IBM credential management (v0.2 stub)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class IBMCredentials:
    """
    IBM Quantum API credentials.

    v0.1: Stub. v0.2: Integrates with qiskit-ibm-runtime.

    Args:
        token: IBM Quantum API token.
        instance: Hub/group/project string, e.g. "ibm-q/open/main".
    """
    token: str = ""
    instance: str = "ibm-q/open/main"

    @classmethod
    def from_env(cls) -> IBMCredentials:
        """Load credentials from environment variables."""
        return cls(
            token=os.environ.get("IBM_QUANTUM_TOKEN", ""),
            instance=os.environ.get("IBM_QUANTUM_INSTANCE", "ibm-q/open/main"),
        )

    def is_valid(self) -> bool:
        return bool(self.token)
