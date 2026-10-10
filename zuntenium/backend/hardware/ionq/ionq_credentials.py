"""IonQ credential management (v0.2 stub)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class IonQCredentials:
    """IonQ API credentials."""
    api_key: str = ""
    base_url: str = "https://api.ionq.co/v0.3"

    @classmethod
    def from_env(cls) -> IonQCredentials:
        return cls(
            api_key=os.environ.get("IONQ_API_KEY", ""),
            base_url=os.environ.get("IONQ_BASE_URL", "https://api.ionq.co/v0.3"),
        )

    def is_valid(self) -> bool:
        return bool(self.api_key)
