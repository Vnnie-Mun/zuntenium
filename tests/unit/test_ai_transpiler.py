"""
Comprehensive tests for the Zuntenium AI Transpiler.

Validates all Phase 1 rule-based passes, the RL environment,
state encoder, reward function, and end-to-end integration.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from zuntenium.transpiler.ai.ir import (
    CircuitDAG,
    Gate,
    HardwareConfig,
    TranspilerResult,
    circuit_to_dag,
    dag_to_circuit,
)
from zuntenium.transpiler.ai.passes import (
    CommutativeReorderingPass,
    PauliCancellationPass,
    RotationFusionPass,
    RuleEngine,
    SingleQubitUnifyPass,
    SWAPRoutingPass,
    gates_commute,
)
from zuntenium.transpiler.ai.rl import (
    CircuitOptimizationEnv,
    CircuitStateEncoder,
    ReplayBuffer,
    RewardFunction,
    RLAgent,
    TransformAction,
)
from zuntenium.transpiler.ai.transpiler import (
    AITranspiler,
    DatasetGenerator,
    TranspilerAwareHybridSolver,
)


# ═══════════════════════════════════════════════════════════════════════════
# Gate IR tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGate:
    def test_hash_and_eq(self) -> None:
        g1 = Gate("H", (0,))
        g2 = Gate("H", (0,))
        assert g1 == g2
        assert hash(g1) == hash(g2)
        assert g1 != Gate("X", (0,))

    def test_is_single_qubit(self) -> None:
        assert Gate("H", (0,)).is_single_qubit()
        assert not Gate("CNOT", (0, 1)).is_single_qubit()

    def test_is_parametric(self) -> None:
        assert Gate("RZ", (0,), (0.5,)).is_parametric()
        assert not Gate("H", (0,)).is_parametric()

    def test_repr(self) -> None:
        r = repr(Gate("RZ", (0,), (0.5,)))
        assert "RZ" in r and "0.5" in r


class TestHardwareConfig:
    def test_fully_connected(self) -> None:
        hw = HardwareConfig.fully_connected(4)
        assert hw.are_connected(0, 3)
        assert hw.shortest_swap_path(0, 3) == []

    def test_linear_chain(self) -> None:
        hw = HardwareConfig.linear_chain(4)
        assert hw.are_connected(0, 1)
        assert not hw.are_connected(0, 2)

    def test_shortest_swap_path(self) -> None:
        hw = HardwareConfig.linear_chain(4)
        path = hw.shortest_swap_path(0, 3)
        assert len(path) == 2  # SWAP 0-1, SWAP 1-2


class TestCircuitDAG:
    def test_empty(self) -> None:
        dag = CircuitDAG()
        assert dag.depth() == 0
        assert dag.gate_count() == 0
        assert dag.to_gate_list() == []

    def test_single_gate(self) -> None:
        dag = CircuitDAG()
        dag.add_gate(Gate("H", (0,)))
        assert dag.depth() == 1
        assert dag.gate_count() == 1

    def test_depth_parallel(self) -> None:
        """Two gates on different qubits → depth 1."""
        dag = circuit_to_dag([Gate("H", (0,)), Gate("X", (1,))])
        assert dag.depth() == 1

    def test_depth_serial(self) -> None:
        """Two gates on same qubit → depth 2."""
        dag = circuit_to_dag([Gate("H", (0,)), Gate("X", (0,))])
        assert dag.depth() == 2

    def test_roundtrip(self) -> None:
        gates = [Gate("H", (0,)), Gate("CNOT", (0, 1)), Gate("X", (1,))]
        dag = circuit_to_dag(gates)
        result = dag_to_circuit(dag)
        assert len(result) == 3
        assert result[0].name == "H"
        assert result[2].name == "X"

    def test_copy(self) -> None:
        dag = circuit_to_dag([Gate("H", (0,))])
        copy = dag.copy()
        assert copy.gate_count() == 1
        copy.add_gate(Gate("X", (0,)))
        assert dag.gate_count() == 1  # original unchanged


# ═══════════════════════════════════════════════════════════════════════════
# Rule-based pass tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCommutation:
    def test_disjoint_commute(self) -> None:
        assert gates_commute(Gate("H", (0,)), Gate("X", (1,)))

    def test_same_qubit_hh(self) -> None:
        assert gates_commute(Gate("H", (0,)), Gate("H", (0,)))

    def test_same_qubit_no(self) -> None:
        assert not gates_commute(Gate("H", (0,)), Gate("X", (0,)))

    def test_cnot_z_control(self) -> None:
        assert gates_commute(Gate("CNOT", (0, 1)), Gate("Z", (0,)))

    def test_cnot_x_target(self) -> None:
        assert gates_commute(Gate("CNOT", (0, 1)), Gate("X", (1,)))


class TestPauliCancellation:
    def test_hh_cancel(self) -> None:
        r = PauliCancellationPass().run([Gate("H", (0,)), Gate("H", (0,))])
        assert r == []

    def test_xx_cancel(self) -> None:
        r = PauliCancellationPass().run([Gate("X", (0,)), Gate("X", (0,))])
        assert r == []

    def test_cnot_cnot(self) -> None:
        g = [Gate("CNOT", (0, 1)), Gate("CNOT", (0, 1))]
        r = PauliCancellationPass().run(g)
        assert r == []

    def test_s_sdg(self) -> None:
        r = PauliCancellationPass().run([Gate("S", (0,)), Gate("SDG", (0,))])
        assert r == []

    def test_rz_neg(self) -> None:
        r = PauliCancellationPass().run([
            Gate("RZ", (0,), (0.5,)),
            Gate("RZ", (0,), (-0.5,)),
        ])
        assert r == []

    def test_no_false_cancel(self) -> None:
        r = PauliCancellationPass().run([Gate("H", (0,)), Gate("X", (0,))])
        assert len(r) == 2


class TestRotationFusion:
    def test_two_rz(self) -> None:
        r = RotationFusionPass().run([
            Gate("RZ", (0,), (0.3,)),
            Gate("RZ", (0,), (0.7,)),
        ])
        assert len(r) == 1
        assert r[0].name == "RZ"
        assert math.isclose(r[0].params[0], 1.0, abs_tol=1e-9)

    def test_cancel_to_zero(self) -> None:
        r = RotationFusionPass().run([
            Gate("RX", (0,), (1.0,)),
            Gate("RX", (0,), (-1.0,)),
        ])
        assert r == []

    def test_three_rz(self) -> None:
        r = RotationFusionPass().run([
            Gate("RZ", (0,), (0.3,)),
            Gate("RZ", (0,), (0.7,)),
            Gate("RZ", (0,), (-0.5,)),
        ])
        assert len(r) == 1
        exp = (0.5 + math.pi) % (2 * math.pi) - math.pi
        assert math.isclose(r[0].params[0], exp, abs_tol=1e-9)


class TestSingleQubitUnify:
    def test_two_h_cancel(self) -> None:
        r = SingleQubitUnifyPass().run([Gate("H", (0,)), Gate("H", (0,))])
        assert r == []  # H·H = I

    def test_fold_to_u3(self) -> None:
        r = SingleQubitUnifyPass().run([Gate("H", (0,)), Gate("S", (0,))])
        assert len(r) == 1
        assert r[0].name == "U3"


class TestSWAPRouting:
    def test_no_hw(self) -> None:
        g = [Gate("CNOT", (0, 1))]
        r = SWAPRoutingPass().run(g, hw=None)
        assert r == g

    def test_connected_no_swap(self) -> None:
        hw = HardwareConfig.linear_chain(4)
        r = SWAPRoutingPass().run([Gate("CNOT", (0, 1))], hw)
        assert all(g.name != "SWAP" for g in r)

    def test_disconnected_inserts_swap(self) -> None:
        hw = HardwareConfig.linear_chain(4)
        r = SWAPRoutingPass().run([Gate("CNOT", (0, 3))], hw)
        swaps = [g for g in r if g.name == "SWAP"]
        assert len(swaps) == 2


class TestCommutativeReordering:
    def test_no_change_needed(self) -> None:
        g = [Gate("H", (0,)), Gate("X", (1,))]
        r = CommutativeReorderingPass().run(g)
        assert len(r) == 2


# ═══════════════════════════════════════════════════════════════════════════
# RL component tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCircuitStateEncoder:
    def test_dim(self) -> None:
        assert CircuitStateEncoder.state_dim() == 1354

    def test_encode_empty(self) -> None:
        v = CircuitStateEncoder.encode([])
        assert v.shape == (1354,)
        assert v.dtype == np.float32

    def test_encode_gates(self) -> None:
        v = CircuitStateEncoder.encode([Gate("H", (0,)), Gate("CNOT", (0, 1))])
        assert v.shape == (1354,)
        assert v[3] == 1.0  # H is index 3 in GATE_TYPES


class TestRewardFunction:
    def test_improvement_positive(self) -> None:
        rf = RewardFunction()
        old = [Gate("H", (0,)), Gate("H", (0,))]
        new: list[Gate] = []
        r = rf.compute(old, new)
        assert r > 0

    def test_no_change_zero(self) -> None:
        rf = RewardFunction()
        g = [Gate("H", (0,))]
        r = rf.compute(g, g)
        assert math.isclose(r, 0.0, abs_tol=0.01)

    def test_fidelity_monotone(self) -> None:
        rf = RewardFunction()
        f1 = rf.fidelity([Gate("H", (0,))])
        f5 = rf.fidelity([Gate("H", (i,)) for i in range(5)])
        assert f1 > f5


class TestReplayBuffer:
    def test_push_and_len(self) -> None:
        buf = ReplayBuffer(capacity=10)
        s = np.zeros(4, dtype=np.float32)
        buf.push(s, 0, 1.0, s, False)
        assert len(buf) == 1

    def test_overflow(self) -> None:
        buf = ReplayBuffer(capacity=3)
        s = np.zeros(2, dtype=np.float32)
        for i in range(5):
            buf.push(s, i, float(i), s, False)
        assert len(buf) == 3

    def test_clear(self) -> None:
        buf = ReplayBuffer()
        buf.push(np.zeros(2), 0, 0, np.zeros(2), False)
        buf.clear()
        assert len(buf) == 0


class TestCircuitOptimizationEnv:
    def test_reset_returns_state(self) -> None:
        env = CircuitOptimizationEnv([Gate("H", (0,))])
        s = env.reset()
        assert s.shape == (1354,)

    def test_step_returns_tuple(self) -> None:
        env = CircuitOptimizationEnv([Gate("H", (0,)), Gate("H", (0,))])
        s = env.reset()
        ns, r, done = env.step(1)  # CancelPauliPair
        assert ns.shape == (1354,)
        assert isinstance(r, float)
        assert isinstance(done, bool)


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRuleEngine:
    def test_adversarial(self) -> None:
        adv = DatasetGenerator.adversarial_circuit(3)
        engine = RuleEngine()
        opt, log = engine.optimize(adv)
        assert len(opt) == 0
        assert len(log) > 0

    def test_empty(self) -> None:
        opt, log = RuleEngine().optimize([])
        assert opt == []

    def test_single_gate(self) -> None:
        opt, _ = RuleEngine().optimize([Gate("H", (0,))])
        assert len(opt) == 1


class TestAITranspiler:
    def test_phase1_adversarial(self) -> None:
        t = AITranspiler(use_ai=False)
        adv = DatasetGenerator.adversarial_circuit(3)
        r = t.optimize(adv)
        assert r.optimized_gate_count == 0
        assert r.gate_reduction_pct == 100.0
        assert r.phase == "rule_based"

    def test_phase1_bell(self) -> None:
        t = AITranspiler(use_ai=False)
        bell = [
            Gate("H", (0,)), Gate("H", (0,)),
            Gate("H", (0,)), Gate("CNOT", (0, 1)),
        ]
        r = t.optimize(bell)
        assert r.optimized_gate_count == 2  # H + CNOT
        assert r.gate_reduction_pct == 50.0

    def test_phase1_rotation_fusion(self) -> None:
        t = AITranspiler(use_ai=False)
        rots = [
            Gate("RZ", (0,), (0.3,)),
            Gate("RZ", (0,), (0.7,)),
            Gate("RZ", (0,), (-0.5,)),
        ]
        r = t.optimize(rots)
        assert len(r.circuit) == 1

    def test_swap_routing(self) -> None:
        t = AITranspiler(use_ai=False)
        hw = HardwareConfig.linear_chain(4)
        r = t.optimize([Gate("H", (0,)), Gate("CNOT", (0, 3))], hw=hw)
        swaps = [g for g in r.circuit if g.name == "SWAP"]
        assert len(swaps) == 2

    def test_transpiler_result_summary(self) -> None:
        r = TranspilerResult(
            circuit=[], original_depth=10, optimized_depth=5,
            original_gate_count=20, optimized_gate_count=10,
            estimated_fidelity=0.99, optimization_time_ms=1.0,
            phase="test",
        )
        s = r.summary()
        assert "50.0%↓" in s

    def test_empty_circuit(self) -> None:
        t = AITranspiler()
        r = t.optimize([])
        assert r.optimized_gate_count == 0


class TestDatasetGenerator:
    def test_random(self) -> None:
        g = DatasetGenerator.random_circuit(4, 20)
        assert len(g) == 20

    def test_adversarial(self) -> None:
        g = DatasetGenerator.adversarial_circuit(3)
        assert len(g) == 24  # 8 gates per qubit × 3

    def test_qaoa(self) -> None:
        g = DatasetGenerator.qaoa_circuit(4, 2, [0.1, 0.2], [0.3, 0.4])
        assert len(g) > 0

    def test_ghz(self) -> None:
        g = DatasetGenerator.ghz_circuit(4)
        assert len(g) == 4  # H + 3 CNOTs


class TestTranspilerAwareHybridSolver:
    def test_no_simulator(self) -> None:
        t = AITranspiler(use_ai=False)
        solver = TranspilerAwareHybridSolver(t, simulator=None)
        out = solver.execute_circuit([Gate("H", (0,))])
        assert "transpiler_result" in out
        assert "HybridSolver" in solver.stats()
