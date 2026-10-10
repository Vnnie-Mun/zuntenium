"""API key and OAuth management stub (v0.5)."""
from __future__ import annotations

import os


class Auth:
    """OAuth / API key authentication for ZUNTENIUM Cloud (v0.5)."""

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or os.environ.get("ZUNTENIUM_API_KEY", "")

    def is_authenticated(self) -> bool:
        return bool(self._api_key)

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
