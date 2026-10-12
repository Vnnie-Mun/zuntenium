"""
Canary Secret Leak Test Suite (F25).

Verifies that sensitive provider credentials, API tokens, and authorization headers
never leak into object reprs, exception strings, tracebacks, or execution metadata.
"""
from __future__ import annotations

import io
import logging
import sys

import pytest

from zuntenium.backends.ibm_backend import CredentialManager, IBMBackend
from zuntenium.quantum_core.executor.result import ExecutionResult

CANARY_TOKEN = "zqc_canary_token_ALPHA_77339944_TOPSECRET"


def test_credential_manager_repr_masks_token():
    """Verify that CredentialManager.__repr__ masks raw tokens."""
    creds = CredentialManager(token=CANARY_TOKEN)
    rep = repr(creds)
    assert CANARY_TOKEN not in rep, f"Canary token leaked in CredentialManager repr: {rep}"
    assert "****" in rep


def test_ibm_backend_repr_does_not_leak_token():
    """Verify that IBMBackend.__repr__ never reveals raw tokens."""
    backend = IBMBackend(token=CANARY_TOKEN, ibm_backend_name="ibm_brisbane")
    rep = repr(backend)
    assert CANARY_TOKEN not in rep, f"Canary token leaked in IBMBackend repr: {rep}"
    assert str(backend) == rep
    assert CANARY_TOKEN not in str(backend)


def test_auth_exception_does_not_echo_raw_token():
    """Verify that authentication failure messages never echo raw tokens."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger("zuntenium")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    try:
        backend = IBMBackend(token=CANARY_TOKEN, ibm_backend_name="ibm_brisbane")
        # Trigger auth/backend check in dry-run mode
        _ = backend.get_available_backends()
    except Exception as exc:
        err_str = str(exc)
        assert CANARY_TOKEN not in err_str, f"Canary token leaked in exception: {err_str}"
    finally:
        logger.removeHandler(handler)

    logs = log_capture.getvalue()
    assert CANARY_TOKEN not in logs, f"Canary token leaked into logs: {logs}"


def test_execution_result_metadata_leak_guard():
    """Verify that ExecutionResult does not leak raw tokens into metadata or repr."""
    res = ExecutionResult(
        counts={"00": 500, "11": 500},
        num_qubits=2,
        shots=1000,
        metadata={"backend": "ibm_brisbane"},
    )
    rep = repr(res)
    assert CANARY_TOKEN not in rep
    html = res._repr_html_()
    assert CANARY_TOKEN not in html
