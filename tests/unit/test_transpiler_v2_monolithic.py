"""
Pytest integration test for monolithic zuntenium/transpiler/transpiler_v2.py.
"""
import pytest
import zuntenium.transpiler.transpiler_v2 as tv2


def test_monolithic_transpiler_suite():
    suite = tv2.TranspilerTestSuite()
    suite.run_all()
    assert suite._failed == 0
    assert suite._passed == 43


def test_monolithic_benchmarks():
    results = tv2.run_benchmarks(verbose=False)
    assert len(results) == 7
    assert results["Bell+3H"]["g_after"] == 2
    assert results["Adversarial-10"]["g_after"] == 0
