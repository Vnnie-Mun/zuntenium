"""
Unit tests for ZUNTENIUM IBM Quantum Backend Integration.
"""
import pytest
from zuntenium.backends.ibm_backend import (
    BackendInfo,
    BackendRegistry,
    CredentialManager,
    IBMBackend,
    IBMBackendTestSuite,
    JobHandle,
    JobStatus,
    QuantumBackend,
    SimulatorBackend,
    UnifiedQuantumExecutor,
    ZunteniumToQiskitConverter,
)


@pytest.fixture
def ibm_suite():
    return IBMBackendTestSuite()


def test_abstract_interface(ibm_suite):
    ibm_suite.test_abstract_interface()
    assert ibm_suite._f == 0


def test_simulator_backend(ibm_suite):
    ibm_suite.test_simulator_backend()
    assert ibm_suite._f == 0


def test_converter_bell(ibm_suite):
    ibm_suite.test_converter_bell()
    assert ibm_suite._f == 0


def test_converter_all_native_ibm(ibm_suite):
    ibm_suite.test_converter_all_native_ibm()
    assert ibm_suite._f == 0


def test_credential_repr_masked(ibm_suite):
    ibm_suite.test_credential_repr_masked()
    assert ibm_suite._f == 0


def test_credential_env(ibm_suite):
    ibm_suite.test_credential_env()
    assert ibm_suite._f == 0


def test_job_status_enum(ibm_suite):
    ibm_suite.test_job_status_enum()
    assert ibm_suite._f == 0


def test_job_handle_age(ibm_suite):
    ibm_suite.test_job_handle_age()
    assert ibm_suite._f == 0


def test_registry(ibm_suite):
    ibm_suite.test_registry()
    assert ibm_suite._f == 0


def test_backend_switch(ibm_suite):
    ibm_suite.test_backend_switch()
    assert ibm_suite._f == 0


def test_available_backends_mock(ibm_suite):
    ibm_suite.test_available_backends_mock()
    assert ibm_suite._f == 0


def test_backend_selection_scoring(ibm_suite):
    ibm_suite.test_backend_selection_scoring()
    assert ibm_suite._f == 0


def test_ibm_backend_mock_mode(ibm_suite):
    ibm_suite.test_ibm_backend_mock_mode()
    assert ibm_suite._f == 0


def test_ibm_fallback_execute(ibm_suite):
    ibm_suite.test_ibm_fallback_execute()
    assert ibm_suite._f == 0


def test_batch_execute_fallback(ibm_suite):
    ibm_suite.test_batch_execute_fallback()
    assert ibm_suite._f == 0


def test_unified_executor_simulator(ibm_suite):
    ibm_suite.test_unified_executor_simulator()
    assert ibm_suite._f == 0


def test_unified_executor_ibm_fallback(ibm_suite):
    ibm_suite.test_unified_executor_ibm_fallback()
    assert ibm_suite._f == 0


def test_backend_aware_vqe(ibm_suite):
    ibm_suite.test_backend_aware_vqe()
    assert ibm_suite._f == 0


def test_dry_run(ibm_suite):
    ibm_suite.test_dry_run()
    assert ibm_suite._f == 0
