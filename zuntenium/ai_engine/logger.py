# zuntenium/ai_engine/logger.py
"""
ExecutionLogger — Persistent execution history for the AI Optimization Engine.

Uses SQLite (stdlib) for zero-dependency, embedded, production-grade storage.

Schema
------
  executions: one row per circuit execution
  training_cache: precomputed training batches (optional cache)

All writes are atomic via WAL mode.  Reads are read-only transactions.
Thread-safe via connection-per-thread pattern.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS executions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               REAL    NOT NULL,
    datetime_utc            TEXT    NOT NULL,

    -- Circuit identity
    circuit_name            TEXT,
    num_qubits              INTEGER,

    -- FeatureVector (JSON blob for forward-compatibility)
    features_json           TEXT    NOT NULL,

    -- AIDecision
    decision_backend        TEXT    NOT NULL,
    decision_strategy       TEXT    NOT NULL,
    decision_params_json    TEXT,
    decision_confidence     REAL,
    decision_reason         TEXT,
    decision_source         TEXT,
    fallback_used           INTEGER DEFAULT 0,   -- BOOL as int

    -- Execution outcome
    execution_time_s        REAL,
    fidelity                REAL,
    success                 INTEGER DEFAULT 1,   -- BOOL
    error_message           TEXT,

    -- RL reward signal
    reward                  REAL
);

CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON executions(timestamp);
CREATE INDEX IF NOT EXISTS idx_executions_num_qubits ON executions(num_qubits);
CREATE INDEX IF NOT EXISTS idx_executions_backend    ON executions(decision_backend);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# ---------------------------------------------------------------------------
# ExecutionLogger
# ---------------------------------------------------------------------------

class ExecutionLogger:
    """
    Thread-safe, persistent logger for Zuntenium AI Engine execution history.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
        Default: ``{package_root}/data/execution_history.db``
    max_rows : int
        Maximum rows retained before auto-compaction (default 100_000).
    """

    _LOCAL = threading.local()   # thread-local connection cache

    def __init__(
        self,
        db_path: str | Path | None = None,
        max_rows: int = 100_000,
        async_writes: bool = False,
        max_queue_size: int = 10_000,
    ) -> None:
        if db_path is None:
            here = Path(__file__).resolve().parent.parent
            db_path = here / "data" / "execution_history.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.async_writes = async_writes
        self._local = threading.local()
        self._init_db()

        self._write_queue: queue.Queue[tuple[str, tuple[Any, ...]] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        if self.async_writes:
            self._write_queue = queue.Queue(maxsize=max_queue_size)
            self._writer_thread = threading.Thread(
                target=self._async_worker, daemon=True, name="ExecutionLogger-AsyncWriter"
            )
            self._writer_thread.start()

        _log.info(
            "ExecutionLogger initialised at %s (async_writes=%s)",
            self.db_path, self.async_writes
        )

    def _async_worker(self) -> None:
        """Background thread consuming records from the queue and batch-inserting into SQLite."""
        while True:
            if self._write_queue is None:
                break
            item = self._write_queue.get()
            if item is None:
                self._write_queue.task_done()
                break
            try:
                sql, params = item
                with self._conn() as conn:
                    conn.execute(sql, params)
                    conn.commit()
            except Exception as e:
                _log.warning("AsyncExecutionLogger write failed: %s", e)
            finally:
                self._write_queue.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until all pending writes in the async queue are committed."""
        if self.async_writes and self._write_queue is not None:
            self._write_queue.join()

    def close(self) -> None:
        """Flush and cleanly shut down the background writer thread."""
        if self.async_writes and self._write_queue is not None:
            self._write_queue.put(None)
            if self._writer_thread is not None and self._writer_thread.is_alive():
                self._writer_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a per-thread connection with WAL mode enabled."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        yield conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                ("schema_version", "1.0"),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_execution(
        self,
        *,
        circuit_name: str = "unknown",
        num_qubits: int = 0,
        features,                  # FeatureVector
        decision,                  # AIDecision
        execution_time_s: float = 0.0,
        fidelity: float = 1.0,
        success: bool = True,
        error_message: str | None = None,
        reward: float | None = None,
    ) -> int:
        """
        Persist one execution record.

        Returns
        -------
        int : row id of the inserted record
        """
        now = time.time()
        dt_utc = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        # Auto-compute reward if not provided
        if reward is None:
            reward = self._compute_reward(
                execution_time_s=execution_time_s,
                fidelity=fidelity,
                success=success,
                fallback_used=getattr(decision, "fallback_used", False),
            )

        features_json = json.dumps(features.to_dict())
        params_json   = json.dumps(getattr(decision, "params", {}))

        insert_sql = """
        INSERT INTO executions (
            timestamp, datetime_utc,
            circuit_name, num_qubits,
            features_json,
            decision_backend, decision_strategy,
            decision_params_json, decision_confidence,
            decision_reason, decision_source, fallback_used,
            execution_time_s, fidelity, success, error_message,
            reward
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        insert_args = (
            now, dt_utc,
            circuit_name, num_qubits,
            features_json,
            decision.backend.value, decision.transpilation_strategy.value,
            params_json, float(decision.confidence),
            decision.reason, decision.source.value, int(decision.fallback_used),
            float(execution_time_s), float(fidelity),
            int(success), error_message,
            float(reward),
        )

        if self.async_writes and self._write_queue is not None:
            self._write_queue.put((insert_sql, insert_args))
            return -1

        with self._conn() as conn:
            cur = conn.execute(insert_sql, insert_args)
            conn.commit()
            row_id = cur.lastrowid

        # Async compaction check
        self._maybe_compact_async()
        return row_id

    # ------------------------------------------------------------------
    # Query helpers (used by FeatureExtractor for historical features)
    # ------------------------------------------------------------------

    def query_stats(self, num_qubits: int | None = None) -> dict[str, Any]:
        """
        Return aggregated stats for the FeatureExtractor.

        Parameters
        ----------
        num_qubits : optional filter — only consider circuits with this qubit count
        """
        with self._conn() as conn:
            where = "WHERE num_qubits = ?" if num_qubits is not None else ""
            args  = (num_qubits,) if num_qubits is not None else ()

            # Last 10 execution times
            rows = conn.execute(
                f"""
                SELECT execution_time_s, success, decision_backend
                FROM executions {where}
                ORDER BY timestamp DESC LIMIT 20
                """,
                args,
            ).fetchall()

        if not rows:
            return {
                "avg_exec_time_last_10": 0.0,
                "success_rate_last_20": 1.0,
                "prev_backend": "statevector",
                "total_executions": 0,
            }

        times = [r["execution_time_s"] for r in rows[:10] if r["execution_time_s"] is not None]
        successes = [r["success"] for r in rows]
        prev_backend = rows[0]["decision_backend"] if rows else "statevector"

        with self._conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) as cnt FROM executions {where}", args
            ).fetchone()["cnt"]

        return {
            "avg_exec_time_last_10": float(np.mean(times)) if times else 0.0,
            "success_rate_last_20": float(np.mean(successes)) if successes else 1.0,
            "prev_backend": prev_backend,
            "total_executions": int(total),
        }

    def get_training_data(
        self,
        limit: int = 10_000,
        min_reward: float | None = None,
    ) -> list[dict]:
        """
        Return rows suitable for (re)training the supervised model.

        Returns list of dicts with keys:
            features_array (np.ndarray), label (str backend), reward (float)
        """
        from .features import FeatureVector

        where_clauses = ["success = 1"]
        args: list[Any] = []
        if min_reward is not None:
            where_clauses.append("reward >= ?")
            args.append(min_reward)
        where = "WHERE " + " AND ".join(where_clauses)

        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT features_json, decision_backend, decision_strategy, reward
                FROM executions {where}
                ORDER BY timestamp DESC LIMIT ?
                """,
                (*args, limit),
            ).fetchall()

        result = []
        for row in rows:
            try:
                fdict = json.loads(row["features_json"])
                fv = FeatureVector.from_dict(fdict)
                result.append({
                    "features_array": fv.to_array(),
                    "label":    row["decision_backend"],
                    "strategy": row["decision_strategy"],   # FIX BUG-003: include strategy
                    "reward":   float(row["reward"]),
                })
            except Exception as exc:
                _log.debug("Skipping malformed row: %s", exc)
        return result

    def get_rl_experiences(self, limit: int = 5_000) -> list[dict]:
        """
        Return (state, action, reward, next_state, done) tuples for PPO training.
        next_state is approximated as the following row (same qubit count).
        """
        from .features import FeatureVector

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, features_json, decision_backend, reward, success
                FROM executions
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

        experiences = []
        for i, row in enumerate(rows):
            try:
                fdict = json.loads(row["features_json"])
                fv = FeatureVector.from_dict(fdict)
                next_fv = (
                    FeatureVector.from_dict(json.loads(rows[i + 1]["features_json"]))
                    if i + 1 < len(rows) else fv
                )
                experiences.append({
                    "state": fv.to_array(),
                    "action": row["decision_backend"],
                    "reward": float(row["reward"]),
                    "next_state": next_fv.to_array(),
                    "done": i == len(rows) - 1,
                })
            except Exception as exc:
                _log.debug("Skipping malformed RL row: %s", exc)
        return experiences

    def count_rows(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(self) -> int:
        """Delete oldest rows, keeping last ``max_rows`` records. Returns deleted count."""
        try:
            total = self.count_rows()
            if total <= self.max_rows:
                return 0
            delete_n = total - self.max_rows
            with self._conn() as conn:
                conn.execute(
                    """
                    DELETE FROM executions WHERE id IN (
                        SELECT id FROM executions ORDER BY timestamp ASC LIMIT ?
                    )
                    """,
                    (delete_n,),
                )
                conn.commit()
            _log.info("Compacted ExecutionLogger: deleted %d rows", delete_n)
            return delete_n
        except Exception as exc:
            _log.debug("compact exception ignored: %s", exc)
            return 0

    def _maybe_compact_async(self) -> None:
        """Non-blocking compaction: run in a daemon thread."""
        try:
            if self.count_rows() > self.max_rows * 1.5:
                t = threading.Thread(target=self.compact, daemon=True)
                t.start()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reward(
        *,
        execution_time_s: float,
        fidelity: float,
        success: bool,
        fallback_used: bool,
    ) -> float:
        """
        Reward function r_t used for RL training:

            r_t = α·fidelity − β·log(1 + exec_time) − γ·fallback_penalty − δ·fail_penalty

        where α=1.0, β=0.1, γ=0.2, δ=1.0
        Clipped to [-2, 1].
        """
        if not success:
            return -1.0

        alpha, beta, gamma = 1.0, 0.1, 0.2
        import math
        r = (alpha * fidelity
             - beta  * math.log1p(execution_time_s)
             - gamma * float(fallback_used))
        return float(np.clip(r, -2.0, 1.0))


class AsyncExecutionLogger(ExecutionLogger):
    """
    ExecutionLogger with an asynchronous non-blocking SQLite write queue.

    Inserts records in the background without introducing disk I/O latency
    to simulation or execution loops.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        max_rows: int = 100_000,
        max_queue_size: int = 10_000,
    ) -> None:
        super().__init__(
            db_path=db_path,
            max_rows=max_rows,
            async_writes=True,
            max_queue_size=max_queue_size,
        )


__all__ = ["ExecutionLogger", "AsyncExecutionLogger"]
