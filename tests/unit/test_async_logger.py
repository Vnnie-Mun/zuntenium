"""
Unit tests for AsyncExecutionLogger.

Verifies:
  - Non-blocking asynchronous SQLite write queue
  - Proper flush() behavior and zero data loss
  - Clean worker shutdown with close()
  - Statistics queries on logged data
"""

import tempfile
import time
from pathlib import Path
import pytest

from zuntenium.ai_engine import AsyncExecutionLogger, ExecutionLogger
from zuntenium.ai_engine.features import FeatureVector
from zuntenium.ai_engine.decision import (
    AIDecision,
    BackendChoice,
    DecisionSource,
    TranspilationStrategy,
)


class TestAsyncExecutionLogger:

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self._tmpdir) / "async_test.db"
        self.logger = AsyncExecutionLogger(
            db_path=self.db_path,
            max_rows=10_000,
            max_queue_size=1000,
        )

    def teardown_method(self):
        self.logger.close()

    def _make_decision(self):
        return AIDecision(
            backend=BackendChoice.STATEVECTOR,
            transpilation_strategy=TranspilationStrategy.RULE_ONLY,
            params={"shots": 1024},
            confidence=0.92,
            reason="async_test",
            source=DecisionSource.AI_SUPERVISED,
        )

    def test_async_write_and_flush(self):
        fv = FeatureVector(num_qubits=4.0)
        dec = self._make_decision()

        # log_execution returns -1 for async non-blocking enqueue
        ret = self.logger.log_execution(
            circuit_name="vqe_ansatz_0",
            num_qubits=4,
            features=fv,
            decision=dec,
            execution_time_s=0.12,
            fidelity=0.995,
            success=True,
        )
        assert ret == -1

        # Flush background worker
        self.logger.flush()

        assert self.logger.count_rows() == 1
        stats = self.logger.query_stats(num_qubits=4)
        assert stats["total_executions"] == 1
        assert stats["success_rate_last_20"] == 1.0

    def test_rapid_concurrent_burst_writes(self):
        fv = FeatureVector(num_qubits=5.0)
        dec = self._make_decision()
        num_records = 50

        # Rapid writes without disk blocking
        t0 = time.perf_counter()
        for i in range(num_records):
            self.logger.log_execution(
                circuit_name=f"burst_circuit_{i}",
                num_qubits=5,
                features=fv,
                decision=dec,
                execution_time_s=0.01 * (i + 1),
                fidelity=0.98,
                success=True,
            )
        elapsed = time.perf_counter() - t0

        # Enqueueing 50 items should be fast (< 0.2s)
        assert elapsed < 0.2

        # Ensure all writes flush cleanly
        self.logger.flush()
        assert self.logger.count_rows() == num_records

        stats = self.logger.query_stats(num_qubits=5)
        assert stats["total_executions"] == num_records

    def test_clean_close(self):
        fv = FeatureVector(num_qubits=2.0)
        dec = self._make_decision()

        self.logger.log_execution(
            circuit_name="shutdown_circuit",
            num_qubits=2,
            features=fv,
            decision=dec,
            execution_time_s=0.05,
            fidelity=1.0,
            success=True,
        )

        # close() terminates worker thread cleanly
        self.logger.close()
        assert self.logger._writer_thread is None or not self.logger._writer_thread.is_alive()
        # Data was flushed prior to shutdown
        assert self.logger.count_rows() == 1
