"""
TrainingDataCollector — Collects IR snapshots for future ML transpiler training.

Records (before_ir, after_ir, reward) tuples where reward is the improvement
in circuit depth and gate count. These are stored for use in v0.4.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class TrainingDataCollector:
    """
    Collects IR snapshots for ML transpiler training data.

    Data is stored in memory and optionally flushed to a JSONL file.

    Args:
        output_path: Optional path to write JSONL training data.
        max_memory: Maximum number of records to keep in memory.
    """

    def __init__(
        self,
        output_path: str | None = None,
        max_memory: int = 10_000,
    ) -> None:
        self._output_path = Path(output_path) if output_path else None
        self._max_memory = max_memory
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        ir_before: QuantumIR,
        ir_after: QuantumIR | None = None,
        reward: float | None = None,
    ) -> None:
        """
        Record a training sample.

        Args:
            ir_before: IR before optimization.
            ir_after: IR after optimization (if available).
            reward: Scalar reward signal (depth reduction, gate count reduction).
        """
        record: dict[str, Any] = {
            "timestamp": time.time(),
            "before": {
                "num_qubits": ir_before.num_qubits,
                "depth": ir_before.depth(),
                "num_gates": ir_before.num_gates(),
                "gate_counts": ir_before.gate_counts(),
            },
        }
        if ir_after is not None:
            record["after"] = {
                "depth": ir_after.depth(),
                "num_gates": ir_after.num_gates(),
                "gate_counts": ir_after.gate_counts(),
            }
        if reward is not None:
            record["reward"] = reward

        self._records.append(record)

        if len(self._records) >= self._max_memory:
            self.flush()

    def flush(self) -> None:
        """Write buffered records to output file (if configured)."""
        if self._output_path and self._records:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._output_path, "a") as f:
                for rec in self._records:
                    f.write(json.dumps(rec) + "\n")
        self._records.clear()

    @property
    def num_records(self) -> int:
        return len(self._records)
