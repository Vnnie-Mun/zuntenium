# QuFlow DEFECT LEDGER — Phase 0 Baseline

> **Engineer**: Principal Debugging Engineer
> **Baseline Date**: 2026-10-13
> **SDK Version**: 0.2.0 (beta)
> **Test Environment**: Python 3.10.12, Linux x86_64, pytest 9.1.1

---

## Summary

| Metric | Value |
|---|---|
| Unit tests passed | 289 |
| Integration tests passed | 44 |
| Smoke tests (external SDK) | 21 |
| **Total tests** | **354** |
| Total failures | **0** |
| Skipped | 2 (benchmark-only) |
| Warnings | 2 (non-blocking, cosmetic) |

---

## Phase 0 Findings

### DEF-001: Endianness Convention — Documentation Mismatch (RESOLVED)

| Field | Value |
|---|---|
| **Severity** | P2 — Confusing but correct |
| **Status** | ✅ RESOLVED |
| **Layer** | Documentation / Test Expectation |
| **Root Cause** | README previously described "big-endian" ordering, but the simulator uses **little-endian** (q0 = rightmost bit, Qiskit convention). The bitstring `'01'` means q1=0, q0=1. |
| **Evidence** | `X(q0)` on 2-qubit → `{'01': 100}`. `X(q1)` → `{'10': 100}`. Consistent across all backends. |
| **Fix** | Updated smoke test expectations. Documented the convention as little-endian in `test_quflow_external_smoke.py`. |
| **Regression Test** | `TestConventions::test_little_endian_qubit_ordering` |

### DEF-002: `rx()` API Argument Order — Developer Surprise (RESOLVED)

| Field | Value |
|---|---|
| **Severity** | P2 — API ergonomics trap |
| **Status** | ✅ RESOLVED |
| **Layer** | Public API Surface |
| **Root Cause** | `QuantumCircuit.rx()` signature is `rx(theta, qubit)` (theta first), which differs from the `h(qubit)`, `x(qubit)` pattern. Developers instinctively write `rx(qubit, theta=π)` which fails with "got multiple values for argument 'theta'". |
| **Evidence** | `inspect.signature(QuantumCircuit.rx)` → `(self, theta: 'float', qubit: 'int')` |
| **Fix** | Test corrected to use `rx(math.pi, 0)`. Documented in smoke tests. Consider adding qubit-first overload in future minor version. |
| **Regression Test** | `TestConventions::test_rotation_angle_radians` |

### DEF-003: `plt.show()` Warning in Non-Interactive Backends (KNOWN)

| Field | Value |
|---|---|
| **Severity** | P3 — Cosmetic warning |
| **Status** | ⚠️ KNOWN — non-blocking |
| **Layer** | Visualization / `plot_histogram()` |
| **Root Cause** | `plot_histogram()` calls `plt.show()` unconditionally, which emits `UserWarning: FigureCanvasAgg is non-interactive` in headless/test environments. |
| **Evidence** | Warning appears in pytest output during `test_plot_histogram_dict_and_result`. |
| **Suggested Fix** | Guard `plt.show()` behind `matplotlib.is_interactive()` check or add a `show=True` parameter. |

### DEF-004: PyTorch Optional Dependency Warning (KNOWN)

| Field | Value |
|---|---|
| **Severity** | P4 — Informational |
| **Status** | ⚠️ KNOWN — by design |
| **Layer** | Transpiler / AI Module |
| **Root Cause** | `zuntenium.transpiler.ai` emits UserWarning when PyTorch is not installed. This is expected for CPU-only environments. |
| **Evidence** | `UserWarning: PyTorch not found. Phase 2 RL will use the NumPy fallback policy.` |
| **Resolution** | Working as designed. Warning helps users know they can install torch for better performance. |

---

## Backends Verified

| Backend | Alias | Status | Notes |
|---|---|---|---|
| StatevectorSimulator | `statevector` | ✅ Pass | Default backend |
| DensityMatrixSimulator | `density_matrix`, `densitymatrix` | ✅ Pass | Mixed-state support |
| StabilizerSimulator | `stabilizer`, `clifford` | ✅ Pass | 10,000+ qubit Clifford |
| TensorNetworkBackend | `tensor_network`, `mps`, `tensornetwork` | ✅ Pass | MPS with bond truncation |
| GPUStatevectorSimulator | `gpu`, `gpu_statevector` | ✅ Pass | CuPy w/ NumPy fallback |

---

## Cross-Backend Differential Tests

| Circuit | statevector | density_matrix | stabilizer | Result |
|---|---|---|---|---|
| Bell state (H, CX) | `{00, 11}` | `{00, 11}` | `{00, 11}` | ✅ Consistent |
| 3-qubit GHZ | `{000, 111}` | `{000, 111}` | `{000, 111}` | ✅ Consistent |
| Identity (no-op) | `{0: 100}` | `{0: 100}` | `{0: 100}` | ✅ Consistent |
| H·H cancellation | `{0: 100}` | `{0: 100}` | `{0: 100}` | ✅ Consistent |

---

## Next Steps (Phase 1–6)

- [ ] Phase 1: Add `plt.show()` guard for headless environments (DEF-003)
- [ ] Phase 2: Consider adding `rx(qubit, angle)` overload for ergonomics (DEF-002)
- [ ] Phase 3: Document little-endian convention prominently in SDK guides
- [ ] Phase 4: Add IDE-specific integration tests (VS Code, PyCharm, Jupyter)
- [ ] Phase 5: Performance regression baseline with `pytest-benchmark`
- [ ] Phase 6: Hardware backend integration tests (IBM, IonQ stubs)
