"""Unit tests for TranspilerPipeline and optimization passes."""
import math

from zuntenium.circuit.circuit import QuantumCircuit
from zuntenium.transpiler.passes.optimization.constant_folding import ConstantFoldingPass
from zuntenium.transpiler.passes.optimization.gate_cancellation import GateCancellationPass
from zuntenium.transpiler.passes.optimization.rotation_merging import RotationMergingPass
from zuntenium.transpiler.pipeline import TranspilerPipeline


def make_ir(qc: QuantumCircuit) -> object:
    return qc.to_ir()


class TestGateCancellationPass:
    """Tests for H·H = I and similar self-inverse cancellations."""

    def test_cancels_hh(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0).h(0)
        ir = make_ir(qc)
        result = GateCancellationPass().run(ir)
        assert result.num_gates() == 0, "H·H should cancel to identity"

    def test_cancels_xx(self) -> None:
        qc = QuantumCircuit(1)
        qc.x(0).x(0)
        ir = make_ir(qc)
        result = GateCancellationPass().run(ir)
        assert result.num_gates() == 0

    def test_cancels_cnot_pair(self) -> None:
        qc = QuantumCircuit(2)
        qc.cx(0, 1).cx(0, 1)
        ir = make_ir(qc)
        result = GateCancellationPass().run(ir)
        assert result.num_gates() == 0

    def test_does_not_cancel_different_gates(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0).x(0)
        ir = make_ir(qc)
        result = GateCancellationPass().run(ir)
        assert result.num_gates() == 2

    def test_cancels_multiple_pairs(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0).h(0).x(0).x(0).z(0).z(0)
        ir = make_ir(qc)
        result = GateCancellationPass().run(ir)
        assert result.num_gates() == 0

    def test_is_idempotent(self) -> None:
        """Running pass twice should have no extra effect."""
        qc = QuantumCircuit(1)
        qc.h(0).h(0).x(0)
        ir = make_ir(qc)
        p = GateCancellationPass()
        r1 = p.run(ir)
        r2 = p.run(r1)
        assert r1.num_gates() == r2.num_gates()


class TestRotationMergingPass:
    """Tests for Rz(a)·Rz(b) → Rz(a+b)."""

    def test_merges_two_rz(self) -> None:
        qc = QuantumCircuit(1)
        qc.rz(0.5, 0).rz(0.3, 0)
        ir = make_ir(qc)
        result = RotationMergingPass().run(ir)
        assert result.num_gates() == 1
        node = result.instructions[0]
        assert node.gate is not None
        assert abs(node.params[0] - 0.8) < 1e-9

    def test_drops_zero_angle_rotation(self) -> None:
        qc = QuantumCircuit(1)
        qc.rz(math.pi, 0).rz(-math.pi, 0)
        ir = make_ir(qc)
        result = RotationMergingPass(tol=1e-10).run(ir)
        assert result.num_gates() == 0

    def test_does_not_merge_different_axes(self) -> None:
        qc = QuantumCircuit(1)
        qc.rx(0.5, 0).rz(0.5, 0)
        ir = make_ir(qc)
        result = RotationMergingPass().run(ir)
        assert result.num_gates() == 2


class TestConstantFoldingPass:
    """Tests for known-angle gate replacement."""

    def test_rx_pi_folds_to_x(self) -> None:
        qc = QuantumCircuit(1)
        qc.rx(math.pi, 0)
        ir = make_ir(qc)
        result = ConstantFoldingPass().run(ir)
        assert result.num_gates() == 1
        assert result.instructions[0].gate_name() == "X"

    def test_rz_pi2_folds_to_s(self) -> None:
        qc = QuantumCircuit(1)
        qc.rz(math.pi / 2, 0)
        ir = make_ir(qc)
        result = ConstantFoldingPass().run(ir)
        assert result.num_gates() == 1
        assert result.instructions[0].gate_name() == "S"

    def test_rz_zero_is_dropped(self) -> None:
        qc = QuantumCircuit(1)
        qc.rz(0.0, 0)
        ir = make_ir(qc)
        result = ConstantFoldingPass().run(ir)
        assert result.num_gates() == 0


class TestTranspilerPipeline:
    """Tests for the full transpiler pipeline."""

    def test_level_0_is_passthrough(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0).h(0)
        ir = make_ir(qc)
        pipeline = TranspilerPipeline(level=0)
        result = pipeline.run(ir)
        assert result.num_gates() == 2   # not cancelled

    def test_level_1_cancels_hh(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0).h(0)
        pipeline = TranspilerPipeline(level=1)
        result = pipeline.run(make_ir(qc))
        assert result.num_gates() == 0

    def test_level_2_pipeline(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).h(0).rz(0.3, 1).rz(0.2, 1)
        pipeline = TranspilerPipeline(level=2)
        result = pipeline.run(make_ir(qc))
        assert result.num_gates() <= 1

    def test_add_pass(self) -> None:
        pipeline = TranspilerPipeline(level=0)
        pipeline.add_pass(GateCancellationPass())
        assert any(p.name == "GateCancellation" for p in pipeline.passes)

    def test_remove_pass(self) -> None:
        pipeline = TranspilerPipeline(level=1)
        removed = pipeline.remove_pass("GateCancellation")
        assert removed
        assert all(p.name != "GateCancellation" for p in pipeline.passes)
