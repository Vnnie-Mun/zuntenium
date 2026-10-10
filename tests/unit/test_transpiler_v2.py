"""
Transpiler v2 — pytest test suite (33 tests covering all audit fixes).
"""
import math
import timeit
import numpy as np
import pytest

from zuntenium.transpiler.v2.core import Gate, Circuit, normalize_angle, _gate_matrix
from zuntenium.transpiler.v2.passes import (
    RedundantGateCancellationPass, RotationMergingPass,
    CommutationReorderingPass, SingleQubitFusionPass,
    DepthOptimizationPass, DeadGateEliminationPass, _is_inverse,
)
from zuntenium.transpiler.v2.engine import Transpiler, _build_random


def _unitary_equiv(before: Circuit, after: Circuit, tol=1e-6) -> bool:
    """Check circuits produce equivalent statevectors (up to global phase)."""
    def run(circ):
        n = circ.num_qubits; sv = np.zeros(2**n, dtype=np.complex128); sv[0] = 1.0
        for g in circ.gates:
            if g.is_measurement: continue
            M = _gate_matrix(g); qubits = list(g.qubits)
            if len(qubits) == 1:
                q = qubits[0]; new_sv = np.zeros_like(sv)
                for idx in range(2**n):
                    bit = (idx >> q) & 1
                    idx0 = idx & ~(1 << q); idx1 = idx | (1 << q)
                    new_sv[idx] = M[bit, 0] * sv[idx0] + M[bit, 1] * sv[idx1]
                sv = new_sv
            elif len(qubits) == 2:
                q0, q1 = qubits[0], qubits[1]; new_sv = np.zeros_like(sv)
                for idx in range(2**n):
                    b0 = (idx >> q0) & 1; b1 = (idx >> q1) & 1
                    row = b0 * 2 + b1
                    for cb0 in range(2):
                        for cb1 in range(2):
                            col = cb0 * 2 + cb1
                            src = (idx & ~(1<<q0) & ~(1<<q1)) | (cb0<<q0) | (cb1<<q1)
                            new_sv[idx] += M[row, col] * sv[src]
                sv = new_sv
        return sv
    sv_b, sv_a = run(before), run(after)
    return math.isclose(abs(np.vdot(sv_b, sv_a)), 1.0, abs_tol=tol)


# ═══════════════════════════════════════════════════════════
# Original tests
# ═══════════════════════════════════════════════════════════

class TestBasicTranspiler:
    def setup_method(self):
        self.t = Transpiler()

    def test_empty_circuit(self):
        c = Circuit(3); opt, _ = self.t.transpile(c)
        assert opt.num_gates == 0 and opt.depth() == 0

    def test_single_gate(self):
        c = Circuit(1); c.add_gate(Gate('H',(0,)))
        opt, _ = self.t.transpile(c); assert opt.num_gates == 1

    def test_pauli_cancellation_adjacent(self):
        c = Circuit(1)
        for _ in range(4): c.add_gate(Gate('X',(0,)))
        opt, _ = self.t.transpile(c)
        assert opt.num_gates == 0
        assert _unitary_equiv(c, opt)

    def test_pauli_cancellation_3x(self):
        c = Circuit(1)
        for _ in range(3): c.add_gate(Gate('X',(0,)))
        opt, _ = self.t.transpile(c); assert opt.num_gates <= 1

    def test_hadamard_pair(self):
        c = Circuit(2)
        c.add_gate(Gate('H',(0,))); c.add_gate(Gate('H',(0,))); c.add_gate(Gate('H',(1,)))
        opt, _ = self.t.transpile(c); assert opt.num_gates == 1

    def test_rotation_fusion(self):
        c = Circuit(1)
        c.add_gate(Gate('RZ',(0,), params=(0.3,)))
        c.add_gate(Gate('RZ',(0,), params=(0.7,)))
        c.add_gate(Gate('RZ',(0,), params=(-0.5,)))
        opt, _ = self.t.transpile(c)
        assert opt.num_gates == 1
        assert math.isclose(opt.gates[0].params[0], 0.5, abs_tol=1e-6)

    def test_rotation_cancel(self):
        c = Circuit(1)
        c.add_gate(Gate('RX',(0,), params=(1.2,)))
        c.add_gate(Gate('RX',(0,), params=(-1.2,)))
        opt, _ = self.t.transpile(c)
        assert opt.num_gates == 0
        assert _unitary_equiv(c, opt)

    def test_s_sdg_cancel(self):
        c = Circuit(1)
        c.add_gate(Gate('S',(0,))); c.add_gate(Gate('SDG',(0,)))
        opt, _ = self.t.transpile(c); assert opt.num_gates == 0

    def test_s_fusion(self):
        c = Circuit(1)
        c.add_gate(Gate('S',(0,))); c.add_gate(Gate('S',(0,)))
        opt, _ = self.t.transpile(c); assert opt.num_gates <= 1

    def test_bell_state(self):
        c = Circuit(2)
        c.add_gate(Gate('H',(0,))); c.add_gate(Gate('H',(0,))); c.add_gate(Gate('H',(0,)))
        c.add_gate(Gate('CNOT',(1,), controls=(0,)))
        opt, _ = self.t.transpile(c)
        assert opt.num_gates == 2
        assert _unitary_equiv(c, opt)

    def test_commutation_reordering(self):
        c = Circuit(2)
        c.add_gate(Gate('Z',(0,)))
        c.add_gate(Gate('CNOT',(1,), controls=(0,)))
        c.add_gate(Gate('Z',(0,)))
        opt, _ = self.t.transpile(c)
        assert _unitary_equiv(c, opt)

    def test_single_qubit_fusion(self):
        c = Circuit(1)
        c.add_gate(Gate('H',(0,))); c.add_gate(Gate('T',(0,))); c.add_gate(Gate('H',(0,)))
        opt, _ = self.t.transpile(c)
        assert opt.num_gates <= 1
        assert _unitary_equiv(c, opt)

    def test_dead_gate_elimination(self):
        c = Circuit(2)
        c.add_gate(Gate('H',(1,))); c.add_gate(Gate('X',(0,))); c.add_gate(Gate('MEASURE',(0,)))
        opt, _ = self.t.transpile(c)
        non_meas = [g for g in opt.gates if not g.is_measurement]
        assert all(0 in g.qubits for g in non_meas)

    def test_20_qubit_circuit(self):
        n = 20; c = Circuit(n)
        for q in range(n):
            c.add_gate(Gate('H',(q,))); c.add_gate(Gate('H',(q,)))
            c.add_gate(Gate('RZ',(q,), params=(0.5,))); c.add_gate(Gate('RZ',(q,), params=(-0.5,)))
        opt, r = self.t.transpile(c)
        assert opt.num_gates == 0
        assert r.total_time_ms < 500

    def test_angle_normalization(self):
        c = Circuit(1); c.add_gate(Gate('RZ',(0,), params=(2*math.pi + 0.5,)))
        opt, _ = self.t.transpile(c)
        if opt.num_gates > 0:
            assert -math.pi < opt.gates[0].params[0] <= math.pi

    def test_only_measurements(self):
        c = Circuit(3)
        for q in range(3): c.add_gate(Gate('MEASURE',(q,)))
        opt, _ = self.t.transpile(c); assert opt.num_gates == 3

    def test_empty_passes_list(self):
        c = Circuit(2); c.add_gate(Gate('H',(0,)))
        opt, _ = Transpiler(passes=[]).transpile(c); assert opt.num_gates == 1


# ═══════════════════════════════════════════════════════════
# Audit fix tests
# ═══════════════════════════════════════════════════════════

class TestAuditFixes:
    def setup_method(self):
        self.t = Transpiler()

    def test_zyz_roundtrip_X(self):
        """CRITICAL-1: ZYZ must round-trip X gate."""
        c = Circuit(1); c.add_gate(Gate('X',(0,)))
        opt, _ = self.t.transpile(c)
        assert _unitary_equiv(c, opt)

    def test_zyz_roundtrip_T(self):
        """CRITICAL-1: ZYZ must round-trip T gate."""
        c = Circuit(1); c.add_gate(Gate('T',(0,)))
        opt, _ = self.t.transpile(c)
        assert _unitary_equiv(c, opt)

    def test_zyz_roundtrip_SDG(self):
        """CRITICAL-1: ZYZ must round-trip SDG gate."""
        c = Circuit(1); c.add_gate(Gate('SDG',(0,)))
        opt, _ = self.t.transpile(c)
        assert _unitary_equiv(c, opt)

    def test_zyz_roundtrip_HT(self):
        """CRITICAL-1: H·T fusion must be unitary-equivalent."""
        c = Circuit(1); c.add_gate(Gate('H',(0,))); c.add_gate(Gate('T',(0,)))
        opt, _ = self.t.transpile(c)
        assert _unitary_equiv(c, opt)

    def test_nan_param_rejected(self):
        """HIGH-2: NaN params must raise ValueError."""
        with pytest.raises(ValueError): Gate('RX', (0,), params=(float('nan'),))

    def test_inf_param_rejected(self):
        """HIGH-2: Inf params must raise ValueError."""
        with pytest.raises(ValueError): Gate('RZ', (0,), params=(float('inf'),))

    def test_nan_angle_normalization(self):
        """HIGH-2: normalize_angle rejects NaN."""
        with pytest.raises(ValueError): normalize_angle(float('nan'))

    def test_commutation_pass_scaling(self):
        """HIGH-1: CommutationReorderingPass must be roughly O(N)."""
        p = CommutationReorderingPass()
        c50 = _build_random(4, 50); t50 = timeit.timeit(lambda: p._apply(c50), number=3)/3
        c200 = _build_random(4, 200); t200 = timeit.timeit(lambda: p._apply(c200), number=3)/3
        ratio = t200 / max(t50, 1e-6)
        assert ratio < 12, f"Scaling ratio {ratio:.1f}x too high (expect <12 for O(N))"

    def test_fuse_nonadjacent_cancel(self):
        """MEDIUM-1: S·S fusion works when separated by commuting gate."""
        c = Circuit(2)
        c.add_gate(Gate('S',(0,))); c.add_gate(Gate('X',(1,))); c.add_gate(Gate('S',(0,)))
        opt, _ = self.t.transpile(c)
        q0_gates = [g for g in opt.gates if 0 in g.qubits]
        assert len(q0_gates) <= 1
        assert _unitary_equiv(c, opt)

    def test_cnot_control_deadgate(self):
        """Dead gate: X(q0)->CNOT(q0,q1)->MEASURE(q1) must NOT eliminate X."""
        c = Circuit(2)
        c.add_gate(Gate('X',(0,))); c.add_gate(Gate('CNOT',(1,), controls=(0,)))
        c.add_gate(Gate('MEASURE',(1,)))
        opt, _ = self.t.transpile(c)
        non_meas = [g for g in opt.gates if not g.is_measurement]
        assert len(non_meas) == 2

    def test_batch_remove(self):
        """LOW-2: batch_remove works correctly."""
        c = Circuit(2)
        c.add_gate(Gate('H',(0,))); c.add_gate(Gate('X',(1,))); c.add_gate(Gate('Z',(0,)))
        removed = c.batch_remove({0, 2})
        assert c.num_gates == 1
        assert c.gates[0].name == 'X'
        assert len(removed) == 2

    def test_rotation_merging_commuting_window(self):
        """MEDIUM-2: Commuting window gates preserved after merge."""
        c = Circuit(2)
        c.add_gate(Gate('RZ',(0,), params=(0.3,)))
        c.add_gate(Gate('X',(1,)))  # commutes with RZ(q0)
        c.add_gate(Gate('RZ',(0,), params=(0.2,)))
        opt, _ = self.t.transpile(c)
        assert _unitary_equiv(c, opt)
        q1_gates = [g for g in opt.gates if 1 in g.qubits]
        assert len(q1_gates) >= 1  # X(q1) must survive

    def test_adversarial_full_cancel(self):
        """All redundant gates cancel to zero."""
        c = Circuit(5)
        for q in range(5):
            c.add_gate(Gate('H',(q,))); c.add_gate(Gate('H',(q,)))
            c.add_gate(Gate('T',(q,))); c.add_gate(Gate('TDG',(q,)))
            c.add_gate(Gate('X',(q,))); c.add_gate(Gate('X',(q,)))
        opt, _ = self.t.transpile(c)
        assert opt.num_gates == 0

    def test_gate_copy_independence(self):
        """Gate copies are independent."""
        g = Gate('RZ', (0,), params=(0.5,))
        g2 = g.copy()
        g2.add_history("modified")
        assert 'history' not in g.metadata or 'modified' not in g.metadata.get('history', [])

    def test_circuit_from_gate_list(self):
        """from_gate_list round-trips correctly."""
        gates = [Gate('H',(0,)), Gate('CNOT',(1,), controls=(0,))]
        c = Circuit.from_gate_list(2, gates)
        assert c.num_gates == 2; assert c.num_qubits == 2

    def test_commutation_expose_cancellation(self):
        """LOW-1: CommutationReorderingPass Phase B exposes cancellation pairs."""
        p = CommutationReorderingPass()
        c = Circuit(2)
        c.add_gate(Gate('X', (0,)))
        c.add_gate(Gate('Z', (1,)))  # commutes with X(0)
        c.add_gate(Gate('X', (0,)))
        reordered = p._apply(c)
        canceller = RedundantGateCancellationPass()
        opt = canceller._apply(reordered)
        assert opt.num_gates == 1
        assert opt.gates[0].name == 'Z'
        assert _unitary_equiv(c, opt)

