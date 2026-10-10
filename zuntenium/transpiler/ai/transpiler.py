"""
AITranspiler — unified entry point, dataset generator, and HybridSolver bridge.

Usage::

    # Phase 1 (rule-based):
    t = AITranspiler()
    r = t.optimize(gates)
    print(r.summary())

    # Phase 2 (RL):
    t = AITranspiler(use_ai=True)
    t.pretrain(num_episodes=200)
    r = t.optimize(gates, shots=5)

    # Online learning feedback:
    t.feedback(r, real_fidelity=0.93)

Author:  ZUNTENIUM Core Team
License: Apache 2.0
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from zuntenium.transpiler.ai.ir import (
    Gate,
    HardwareConfig,
    TranspilerResult,
    circuit_to_dag,
    circuit_to_zuntenium_format,
)
from zuntenium.transpiler.ai.passes import RuleEngine
from zuntenium.transpiler.ai.rl import (
    CircuitOptimizationEnv,
    CircuitStateEncoder,
    ReplayBuffer,
    RewardFunction,
    RLAgent,
)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Generator
# ═══════════════════════════════════════════════════════════════════════════

class DatasetGenerator:
    """Synthetic circuit dataset for RL pre-training and benchmarking."""

    POOL_1Q = ["X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"]
    POOL_2Q = ["CNOT", "CZ", "SWAP"]

    @staticmethod
    def random_circuit(
        n: int,
        d: int,
        p2q: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Gate]:
        """Generate a random circuit with *n* qubits and *d* gates."""
        if rng is None:
            rng = np.random.default_rng()
        gates: List[Gate] = []
        for _ in range(d):
            q = int(rng.integers(0, n))
            if n > 1 and rng.random() < p2q:
                q2 = int(rng.integers(0, n))
                while q2 == q:
                    q2 = int(rng.integers(0, n))
                nm = str(rng.choice(DatasetGenerator.POOL_2Q))
                gates.append(Gate(nm, (q, q2)))
            else:
                nm = str(rng.choice(DatasetGenerator.POOL_1Q))
                params = (
                    (float(rng.uniform(0, 2 * math.pi)),)
                    if nm in ("RX", "RY", "RZ")
                    else ()
                )
                gates.append(Gate(nm, (q,), params))
        return gates

    @staticmethod
    def adversarial_circuit(n: int) -> List[Gate]:
        """Circuit packed with cancellation / fusion opportunities."""
        gates: List[Gate] = []
        for q in range(n):
            gates += [
                Gate("H", (q,)), Gate("H", (q,)),
                Gate("RZ", (q,), (0.5,)), Gate("RZ", (q,), (-0.5,)),
                Gate("T", (q,)), Gate("TDG", (q,)),
                Gate("X", (q,)), Gate("X", (q,)),
            ]
        return gates

    @staticmethod
    def qaoa_circuit(
        n: int,
        p: int,
        gammas: List[float],
        betas: List[float],
    ) -> List[Gate]:
        """Standard QAOA circuit for MaxCut on a linear graph."""
        gates: List[Gate] = []
        for q in range(n):
            gates.append(Gate("H", (q,)))
        for layer in range(p):
            for i in range(n - 1):
                gates += [
                    Gate("CNOT", (i, i + 1)),
                    Gate("RZ", (i + 1,), (2 * gammas[layer],)),
                    Gate("CNOT", (i, i + 1)),
                ]
            for q in range(n):
                gates.append(Gate("RX", (q,), (2 * betas[layer],)))
        return gates

    @staticmethod
    def ghz_circuit(n: int) -> List[Gate]:
        """GHZ state preparation: H(0), CNOT(0,i) for i=1..n-1."""
        return [Gate("H", (0,))] + [Gate("CNOT", (0, i)) for i in range(1, n)]


# ═══════════════════════════════════════════════════════════════════════════
# AITranspiler
# ═══════════════════════════════════════════════════════════════════════════

class AITranspiler:
    """ZUNTENIUM AI Transpiler — Phase 1 (rule-based) + Phase 2 (RL/PPO).

    Args:
        use_ai:    Enable Phase 2 RL agent.
        hw:        Default hardware config (can be overridden per call).
        reward_fn: Custom reward function (default ``RewardFunction()``).
        rl_agent:  Inject a pre-built ``RLAgent`` instance.
    """

    def __init__(
        self,
        use_ai: bool = False,
        hw: Optional[HardwareConfig] = None,
        reward_fn: Optional[RewardFunction] = None,
        rl_agent: Optional[RLAgent] = None,
    ) -> None:
        self.use_ai = use_ai
        self.hw = hw
        self._re = RuleEngine()
        self._rf = reward_fn or RewardFunction()
        self._buf = ReplayBuffer(4096)
        self._agent: Optional[RLAgent] = (
            (rl_agent or RLAgent(CircuitStateEncoder.state_dim()))
            if use_ai
            else None
        )
        self._stats: List[Dict[str, float]] = []

    def optimize(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig] = None,
        shots: int = 1,
    ) -> TranspilerResult:
        """Run the full optimisation pipeline on *gates*.

        Args:
            gates:  Input circuit as a gate list.
            hw:     Hardware config (overrides constructor default).
            shots:  Number of RL episodes (Phase 2 only).

        Returns:
            ``TranspilerResult`` with optimised circuit and metrics.
        """
        hw_eff = hw or self.hw
        t0 = time.perf_counter()
        dag0 = circuit_to_dag(gates)
        d0 = dag0.depth()
        g0 = len(gates)

        opt, log = self._re.optimize(gates, hw_eff)
        phase = "rule_based"

        if self.use_ai and self._agent:
            opt, rl_log = self._rl_optimize(opt, hw_eff, shots)
            log += rl_log
            phase = "rule_based+rl"

        return TranspilerResult(
            circuit=opt,
            original_depth=d0,
            optimized_depth=circuit_to_dag(opt).depth(),
            original_gate_count=g0,
            optimized_gate_count=len(opt),
            estimated_fidelity=self._rf.fidelity(opt),
            optimization_time_ms=(time.perf_counter() - t0) * 1000,
            phase=phase,
            pass_log=log,
        )

    def _rl_optimize(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig],
        shots: int,
    ) -> tuple[List[Gate], List[str]]:
        best = list(gates)
        best_n = len(gates)
        log: List[str] = []
        for ep in range(shots):
            env = CircuitOptimizationEnv(
                gates, self._rf, max_steps=40, hw=hw
            )
            s = env.reset()
            ep_r = 0.0
            while not env._done:
                a, lp, v = self._agent.get_action(s)  # type: ignore[union-attr]
                ns, r, done = env.step(a)
                self._buf.push(s, a, r, ns, done, lp, v)
                s = ns
                ep_r += r
            cur = env._cur
            if len(cur) < best_n:
                best = cur
                best_n = len(cur)
                log.append(f"RL ep{ep}: {best_n} gates (r={ep_r:.2f})")
        if len(self._buf) >= 64:
            m = self._agent.train_step(self._buf)  # type: ignore[union-attr]
            self._stats.append(m)
            self._buf.clear()
        return best, log

    def pretrain(
        self,
        num_episodes: int = 200,
        n_range: tuple[int, int] = (4, 12),
        d_range: tuple[int, int] = (10, 40),
        verbose: bool = True,
    ) -> None:
        """Pre-train the RL agent on synthetic circuits."""
        if not self.use_ai or not self._agent:
            return
        rng = np.random.default_rng(0)
        total = 0.0
        for ep in range(num_episodes):
            n = int(rng.integers(*n_range))
            d = int(rng.integers(*d_range))
            if ep % 3 == 2:
                gates = DatasetGenerator.adversarial_circuit(min(n, 5))
            else:
                gates = DatasetGenerator.random_circuit(n, d, rng=rng)
            opt, _ = self._re.optimize(gates)
            env = CircuitOptimizationEnv(opt, self._rf, max_steps=25)
            s = env.reset()
            ep_r = 0.0
            while not env._done:
                a, lp, v = self._agent.get_action(s)
                ns, r, done = env.step(a)
                self._buf.push(s, a, r, ns, done, lp, v)
                s = ns
                ep_r += r
            total += ep_r
            if len(self._buf) >= 256:
                m = self._agent.train_step(self._buf)
                self._stats.append(m)
                self._buf.clear()
            if verbose and (ep + 1) % 50 == 0:
                print(
                    f"  Pretrain {ep + 1}/{num_episodes}: "
                    f"avg_reward={total / (ep + 1):.3f}"
                )

    def feedback(self, result: TranspilerResult, real_fidelity: float) -> None:
        """Inject real-world fidelity feedback for online RL calibration."""
        if not self.use_ai or not self._agent:
            return
        bonus = math.log(
            max(real_fidelity, 1e-8) / max(result.estimated_fidelity, 1e-8)
        )
        s = CircuitStateEncoder.encode(
            result.circuit, self._rf.fidelity(result.circuit)
        )
        self._buf.push(s, 0, bonus * 2.0, s.copy(), True)


# ═══════════════════════════════════════════════════════════════════════════
# TranspilerAwareHybridSolver
# ═══════════════════════════════════════════════════════════════════════════

class TranspilerAwareHybridSolver:
    """Wraps any circuit execution with automatic transpilation + feedback.

    Workflow::

        gates → AITranspiler.optimize() → StatevectorSimulator.run() → feedback()
    """

    def __init__(
        self,
        transpiler: AITranspiler,
        simulator: Any = None,
    ) -> None:
        self.transpiler = transpiler
        self.sim = simulator
        self._calls = 0
        self._saved = 0

    def execute_circuit(
        self,
        gates: List[Gate],
        shots: int = 1024,
        hw: Optional[HardwareConfig] = None,
    ) -> Dict[str, Any]:
        """Optimise and execute *gates*, returning results + metrics."""
        result = self.transpiler.optimize(gates, hw=hw)
        self._calls += 1
        self._saved += result.original_gate_count - result.optimized_gate_count
        out: Dict[str, Any] = {"transpiler_result": result}
        if self.sim is not None:
            try:
                qc = circuit_to_zuntenium_format(result.circuit)
                qc.run(self.sim)
                out["counts"] = self.sim.sample(shots)
                out["statevector"] = self.sim.get_statevector()
                n = len(out["counts"])
                mx = 2 ** self.sim.num_qubits
                self.transpiler.feedback(
                    result, real_fidelity=min(1.0, n / max(mx * 0.1, 1))
                )
            except Exception as e:
                out["error"] = str(e)
        return out

    def stats(self) -> str:
        """Return a summary string of transpiler savings."""
        return (
            f"HybridSolver: {self._calls} circuits, "
            f"{self._saved} gates saved total."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

def run_demo() -> None:
    """Full demonstration of all transpiler capabilities."""
    print("\n" + "═" * 65)
    print("  ZUNTENIUM AI TRANSPILER — Complete Demo")
    print("═" * 65)

    transpiler = AITranspiler(use_ai=False)

    # 1. Adversarial circuit (all cancellable)
    print("\n── Demo 1: Adversarial circuit (Phase 1 rule-based) ──")
    adv = DatasetGenerator.adversarial_circuit(3)
    r1 = transpiler.optimize(adv)
    print(f"  Before: {len(adv)} gates")
    print(f"  After:  {len(r1.circuit)} gates")
    print(f"  {r1.summary()}")
    for e in r1.pass_log:
        print(f"    {e}")

    # 2. Bell state circuit
    print("\n── Demo 2: Bell state ──")
    bell = [Gate("H", (0,)), Gate("H", (0,)), Gate("H", (0,)), Gate("CNOT", (0, 1))]
    r2 = transpiler.optimize(bell)
    print(f"  {r2.summary()}")

    # 3. Rotation fusion
    print("\n── Demo 3: Rotation fusion ──")
    rots = [
        Gate("RZ", (0,), (0.3,)),
        Gate("RZ", (0,), (0.7,)),
        Gate("RZ", (0,), (-0.5,)),
    ]
    r3 = transpiler.optimize(rots)
    exp_a = (0.5 + math.pi) % (2 * math.pi) - math.pi
    got_a = r3.circuit[0].params[0] if r3.circuit else None
    print(f"  3×RZ fused to: {r3.circuit}")
    print(
        f"  Expected angle {exp_a:.6f}, got {got_a:.6f}, "
        f"match={math.isclose(exp_a, got_a, abs_tol=1e-6) if got_a else 'N/A'}"
    )

    # 4. SWAP routing
    print("\n── Demo 4: SWAP routing on linear chain ──")
    hw = HardwareConfig.linear_chain(4)
    r4 = transpiler.optimize([Gate("H", (0,)), Gate("CNOT", (0, 3))], hw=hw)
    print(f"  After routing: {[str(g) for g in r4.circuit]}")
    print(f"  {r4.summary()}")

    # 5. QAOA circuit
    print("\n── Demo 5: QAOA(n=4, p=2) ──")
    qaoa = DatasetGenerator.qaoa_circuit(4, 2, [0.4, 0.8], [0.3, 0.6])
    r5 = transpiler.optimize(qaoa)
    print(f"  {r5.summary()}")

    # 6. Phase 2 RL pre-training
    print("\n── Demo 6: Phase 2 RL (pre-train 20 episodes) ──")
    rl_t = AITranspiler(use_ai=True)
    rl_t.pretrain(num_episodes=20, verbose=True)
    circ = DatasetGenerator.random_circuit(4, 20, rng=np.random.default_rng(7))
    r6p1 = AITranspiler(use_ai=False).optimize(circ)
    r6rl = rl_t.optimize(circ, shots=3)
    print(f"  Phase 1: {r6p1.summary()}")
    print(f"  RL:      {r6rl.summary()}")

    # 7. HybridSolver integration
    print("\n── Demo 7: HybridSolver integration ──")
    sim = None
    try:
        from zuntenium.quantum_core.statevector import StatevectorSimulator
        sim = StatevectorSimulator(4, seed=1)
    except ImportError:
        pass
    solver = TranspilerAwareHybridSolver(transpiler, sim)
    vqe = DatasetGenerator.qaoa_circuit(4, 1, [0.5], [0.5])
    out = solver.execute_circuit(vqe, shots=512)
    tr = out["transpiler_result"]
    print(
        f"  VQE: {tr.original_gate_count}→{tr.optimized_gate_count} "
        f"gates ({tr.gate_reduction_pct:.1f}%↓)"
    )
    if "counts" in out:
        top4 = dict(list(out["counts"].items())[:4])
        print(f"  Counts (top 4): {top4}")
    print(f"  {solver.stats()}")

    print("\n" + "═" * 65)
    print("  All demos completed. ✓")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
