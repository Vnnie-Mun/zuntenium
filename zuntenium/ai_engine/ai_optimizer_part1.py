"""AI Optimization Engine Part 1: feature engineering and SQLite history."""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zuntenium.execution.adaptive_engine import CircuitAnalyzer, CircuitFeatures
from zuntenium.transpiler.v2 import Circuit

logger = logging.getLogger("zuntenium.ai_engine")


@dataclass
class RuntimeContext:
    hardware_available: bool = True
    queue_time_estimate_s: float = -1.0
    current_system_load: float = 0.0
    noise_model_requested: bool = False
    max_execution_budget_ms: float = 10000.0
    preferred_backend: Optional[str] = None


@dataclass
class HistoricalContext:
    avg_exec_time_ms_last10: float = -1.0
    success_rate_last10: float = 1.0
    last_backend: str = "none"
    last_fidelity: float = -1.0
    num_executions_total: int = 0

    @classmethod
    def no_history(cls) -> "HistoricalContext":
        return cls()


@dataclass
class FeatureVector:
    array: np.ndarray
    names: List[str]
    circuit_sig: str
    raw_circuit: CircuitFeatures

    NAMES = [
        "num_qubits_norm", "depth_norm", "gate_count_norm", "clifford_only",
        "entanglement_score", "parametric", "measurement_count_norm",
        "connectivity_density", "memory_mb_norm", "hardware_available",
        "queue_time_norm", "system_load", "noise_requested", "budget_norm",
        "has_preference", "avg_exec_time_norm", "success_rate", "last_backend_enc",
        "last_fidelity", "exec_count_norm", "ent_x_qubits", "depth_x_parametric",
        "hist_success_x_hw", "memory_x_entanglement",
    ]

    @property
    def dim(self) -> int: return len(self.array)
    def explain(self):
        return "\n".join(["FeatureVector:"] + [f"  {name:<28}: {value:>8.4f}" for name, value in zip(self.names, self.array)])
    def to_dict(self): return dict(zip(self.names, self.array.tolist()))


class FeatureEngineer:
    _BACKEND_ENC = {"statevector": 0.0, "stabilizer": .2, "densitymatrix": .4, "tensornetwork": .6, "hardware": .8, "ibm": .8, "none": -.1}

    def extract(self, circuit: Circuit, runtime: Optional[RuntimeContext] = None, history: Optional[HistoricalContext] = None) -> FeatureVector:
        runtime = runtime or RuntimeContext(); history = history or HistoricalContext.no_history(); features = CircuitAnalyzer.analyze(circuit)
        values = np.zeros(24, dtype=np.float32)
        values[:9] = [np.clip(features.num_qubits / 20, 0, 1), np.clip(features.depth / 100, 0, 1), np.clip(features.gate_count / 200, 0, 1), float(features.clifford_only), features.entanglement_score, float(features.parametric), np.clip(features.measurement_count / 10, 0, 1), features.connectivity_density, np.clip(math.log2(features.estimated_memory_mb + 1) / 14, 0, 1)]
        values[9:15] = [float(runtime.hardware_available), np.clip(runtime.queue_time_estimate_s / 3600, 0, 1) if runtime.queue_time_estimate_s >= 0 else .5, np.clip(runtime.current_system_load, 0, 1), float(runtime.noise_model_requested), np.clip(runtime.max_execution_budget_ms / 10000, 0, 1), float(runtime.preferred_backend is not None)]
        values[15:20] = [np.clip(history.avg_exec_time_ms_last10 / 5000, 0, 1) if history.avg_exec_time_ms_last10 >= 0 else .5, np.clip(history.success_rate_last10, 0, 1), self._BACKEND_ENC.get(history.last_backend.lower(), -.1), np.clip(history.last_fidelity, 0, 1) if history.last_fidelity >= 0 else .5, np.clip(math.log1p(history.num_executions_total) / 7, 0, 1)]
        values[20:] = [values[4] * values[0], values[1] * values[5], values[16] * values[9], values[8] * values[4]]
        return FeatureVector(values, list(FeatureVector.NAMES), self._circuit_signature(features), features)

    @staticmethod
    def _circuit_signature(features):
        return f"n{features.num_qubits // 4}_d{min(features.depth // 20, 5)}_c{int(features.clifford_only)}_p{int(features.parametric)}"


BACKEND_LABELS = ["statevector", "stabilizer", "densitymatrix", "tensornetwork", "hardware"]
TRANSPILE_LABELS = ["rule_only", "rule_rl", "rule_rl_hw"]


@dataclass
class ExecutionRecord:
    timestamp: str
    circuit_sig: str
    features_json: str
    backend: str
    transpile_strat: str
    exec_time_ms: float
    fidelity: float
    success: bool
    reward: float
    error_msg: str = ""


class ExecutionLogger:
    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        circuit_sig TEXT NOT NULL, features_json TEXT NOT NULL,
        backend TEXT NOT NULL, transpile_strat TEXT NOT NULL,
        exec_time_ms REAL NOT NULL, fidelity REAL NOT NULL,
        success INTEGER NOT NULL, reward REAL NOT NULL, error_msg TEXT DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_sig ON executions (circuit_sig);
    CREATE INDEX IF NOT EXISTS idx_ts ON executions (timestamp);
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            try:
                directory = Path.home() / ".zuntenium"
                directory.mkdir(parents=True, exist_ok=True)
                db_path = str(directory / "execution_history.db")
            except OSError:
                import tempfile
                directory = Path(tempfile.gettempdir()) / "zuntenium"
                directory.mkdir(parents=True, exist_ok=True)
                db_path = str(directory / "execution_history.db")
        self._db_path = db_path
        self._init_db()
    def _init_db(self):
        with sqlite3.connect(self._db_path) as connection:
            connection.executescript(self._CREATE_SQL); connection.execute("PRAGMA journal_mode=WAL")
    def _connect(self):
        connection = sqlite3.connect(self._db_path); connection.row_factory = sqlite3.Row; return connection
    def log(self, record: ExecutionRecord):
        try:
            with self._connect() as connection:
                connection.execute("INSERT INTO executions (timestamp,circuit_sig,features_json,backend,transpile_strat,exec_time_ms,fidelity,success,reward,error_msg) VALUES (?,?,?,?,?,?,?,?,?,?)", (record.timestamp, record.circuit_sig, record.features_json, record.backend, record.transpile_strat, record.exec_time_ms, record.fidelity, int(record.success), record.reward, record.error_msg))
        except Exception as error: logger.warning("ExecutionLogger.log failed: %s", error)
    def log_execution(self, features, backend, strat, exec_ms, fidelity, success, error=""):
        reward = compute_reward(exec_ms, fidelity, success)
        self.log(ExecutionRecord(datetime.now(timezone.utc).isoformat(), features.circuit_sig, json.dumps(features.to_dict()), backend, strat, exec_ms, fidelity, success, reward, error))
    def get_history(self, circuit_sig, n=10):
        with self._connect() as connection: rows = connection.execute("SELECT * FROM executions WHERE circuit_sig=? ORDER BY timestamp DESC LIMIT ?", (circuit_sig, n)).fetchall()
        return [dict(row) for row in rows]
    def count_all(self):
        with self._connect() as connection: return connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
    def build_historical_context(self, circuit_sig):
        rows = self.get_history(circuit_sig)
        if not rows: return HistoricalContext.no_history()
        successful_times = [row["exec_time_ms"] for row in rows if row["success"]]; fidelities = [row["fidelity"] for row in rows if row["success"]]
        return HistoricalContext(float(np.mean(successful_times)) if successful_times else -1, sum(row["success"] for row in rows) / len(rows), rows[0]["backend"], float(np.mean(fidelities)) if fidelities else -1, self.count_all())
    def get_training_data(self, min_records=10):
        with self._connect() as connection: rows = connection.execute("SELECT features_json,backend,transpile_strat,reward FROM executions").fetchall()
        if len(rows) < min_records: return None
        data = []
        for row in rows:
            try:
                values = np.array(list(json.loads(row["features_json"]).values()), dtype=np.float32)
                if len(values) == 24: data.append((values, BACKEND_LABELS.index(row["backend"]) if row["backend"] in BACKEND_LABELS else 0, TRANSPILE_LABELS.index(row["transpile_strat"]) if row["transpile_strat"] in TRANSPILE_LABELS else 0, float(row["reward"])))
            except (ValueError, TypeError, json.JSONDecodeError): continue
        if len(data) < min_records: return None
        values, backends, strategies, rewards = zip(*data)
        return np.stack(values), np.array(backends, dtype=np.int64), np.array(strategies, dtype=np.int64), np.array(rewards, dtype=np.float32)
    def cleanup(self, keep_days=30):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        with self._connect() as connection: return connection.execute("DELETE FROM executions WHERE timestamp < ?", (cutoff,)).rowcount
    def stats(self):
        with self._connect() as connection: rows = connection.execute("SELECT backend,COUNT(*) cnt,AVG(exec_time_ms) avg_ms,AVG(fidelity) avg_fid,AVG(success) success_rate,AVG(reward) avg_reward FROM executions GROUP BY backend").fetchall()
        return {"total": self.count_all(), "by_backend": [dict(row) for row in rows]}


def compute_reward(exec_ms, fidelity, success, max_ms=10000.0, w_speed=.4, w_fidelity=.5, w_success=.1):
    speed = 1 - min(exec_ms, max_ms) / max_ms; return float(np.clip(w_speed * speed + w_fidelity * fidelity + w_success * (1.0 if success else -5.0), -1, 1))


class Part1TestSuite:
    def __init__(self):
        self._p = 0
        self._f = 0

    def _a(self, cond: bool, msg: str):
        if cond:
            self._p += 1
            print(f"  ✓  {msg}")
        else:
            self._f += 1
            print(f"  ✗  FAIL: {msg}")

    def test_feature_engineer_basic(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        fe = FeatureEngineer()
        fv = fe.extract(_build_qaoa(4, 1))
        self._a(fv.array.shape == (24,), f"dim=24 (got {fv.array.shape})")
        self._a(bool(np.all(np.isfinite(fv.array))), "all finite")
        self._a(fv.array.dtype == np.float32, "float32")

    def test_feature_range(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa, _build_ghz
        fe = FeatureEngineer()
        for c in [_build_qaoa(4, 1), _build_ghz(3)]:
            fv = fe.extract(c)
            self._a(float(np.min(fv.array)) >= -0.2, f"min≥−0.2 ({float(np.min(fv.array)):.3f})")
            self._a(float(np.max(fv.array)) <= 1.01, f"max≤1 ({float(np.max(fv.array)):.3f})")

    def test_feature_with_context(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        fe = FeatureEngineer()
        rt = RuntimeContext(
            hardware_available=True,
            queue_time_estimate_s=300.0,
            noise_model_requested=True,
        )
        hs = HistoricalContext(
            avg_exec_time_ms_last10=500.0,
            success_rate_last10=0.9,
            last_backend="statevector",
            last_fidelity=0.99,
            num_executions_total=25,
        )
        fv = fe.extract(_build_qaoa(4, 1), rt, hs)
        self._a(fv.array[9] == 1.0, "hardware_available=1.0")
        self._a(fv.array[12] == 1.0, "noise_requested=1.0")
        self._a(fv.array[16] > 0.8, f"success_rate≈0.9 ({fv.array[16]:.3f})")

    def test_circuit_signature(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        fe = FeatureEngineer()
        sig = fe.extract(_build_qaoa(4, 1)).circuit_sig
        self._a(isinstance(sig, str) and len(sig) > 0, f"sig non-empty: {sig}")

    def test_feature_names_count(self):
        self._a(len(FeatureVector.NAMES) == 24, f"24 feature names (got {len(FeatureVector.NAMES)})")

    def test_logger_log_query(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            from zuntenium.transpiler.v2.engine import _build_qaoa
            fe = FeatureEngineer()
            fv = fe.extract(_build_qaoa(4, 1))
            log = ExecutionLogger(db_path)
            for i in range(5):
                log.log_execution(
                    fv, "statevector", "rule_only",
                    exec_ms=10.0 + i, fidelity=0.99, success=True,
                )
            total = log.count_all()
            self._a(total == 5, f"logged 5 records (got {total})")
            hist = log.get_history(fv.circuit_sig, n=3)
            self._a(len(hist) == 3, f"get_history returns 3 (got {len(hist)})")
            hc = log.build_historical_context(fv.circuit_sig)
            self._a(hc.success_rate_last10 == 1.0, "success_rate=1.0")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_logger_training_data(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            from zuntenium.transpiler.v2.engine import _build_qaoa, _build_ghz, _build_random
            fe = FeatureEngineer()
            log = ExecutionLogger(db_path)
            circs = [_build_qaoa(4, 1), _build_ghz(3), _build_random(4, 20)]
            backends = ["statevector", "stabilizer", "tensornetwork"]
            for c, b in zip(circs * 5, backends * 5):
                fv = fe.extract(c)
                log.log_execution(
                    fv, b, "rule_only",
                    exec_ms=10.0, fidelity=0.99, success=True,
                )
            data = log.get_training_data(min_records=5)
            self._a(data is not None, "get_training_data returns data")
            if data:
                X, yb, ys, rewards = data
                self._a(X.shape[1] == 24, f"X has 24 features (got {X.shape[1]})")
                self._a(len(set(yb.tolist())) > 1, "multiple backend classes")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_reward_function(self):
        r1 = compute_reward(100.0, 1.0, True)
        r2 = compute_reward(9000.0, 0.5, False)
        r3 = compute_reward(5000.0, 0.95, True)
        self._a(r1 > r3 > r2, f"ordering: {r1:.3f}>{r3:.3f}>{r2:.3f}")
        self._a(-1 <= r1 <= 1, "r1 in [-1,1]")
        self._a(-1 <= r2 <= 1, "r2 in [-1,1]")

    def test_explain_output(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        fe = FeatureEngineer()
        fv = fe.extract(_build_qaoa(4, 1))
        txt = fv.explain()
        self._a("num_qubits_norm" in txt, "explain() has feature names")
        self._a("FeatureVector" in txt, "explain() has header")

    def run_all(self):
        print("\n" + "═" * 60)
        print("  AI Optimization Engine Part 1 — Test Suite")
        print("═" * 60)
        tests = [m for m in sorted(dir(self)) if m.startswith('test_')]
        for t in tests:
            print(f"\n[{t}]")
            try:
                getattr(self, t)()
            except Exception as e:
                self._f += 1
                print(f"  ✗  EXCEPTION: {e}")
        total = self._p + self._f
        print(f"\n{'═' * 60}")
        print(f"  Results: {self._p}/{total} passed"
              + (f"  ({self._f} FAILED)" if self._f else "  ✓ all pass"))
        print("═" * 60)


__all__ = [
    "RuntimeContext",
    "HistoricalContext",
    "FeatureVector",
    "FeatureEngineer",
    "BACKEND_LABELS",
    "TRANSPILE_LABELS",
    "ExecutionRecord",
    "ExecutionLogger",
    "compute_reward",
    "Part1TestSuite",
]

if __name__ == "__main__":
    Part1TestSuite().run_all()

