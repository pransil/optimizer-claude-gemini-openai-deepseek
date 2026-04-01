"""
Persists each iteration's full record to a JSONL file so runs are resumable
and auditable.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class IterationRecord:
    iteration: int
    timestamp: float = field(default_factory=time.time)

    # Proposal stage
    initial_proposal: str = ""          # Claude's raw first proposal
    initial_code: str = ""

    # Critic stage
    critiques: Dict[str, str] = field(default_factory=dict)   # model_name -> critique text

    # Refined stage (after critic synthesis)
    refined_proposal: str = ""
    refined_code: str = ""

    # Execution stage
    sandbox_stdout: str = ""
    sandbox_stderr: str = ""
    sandbox_exit_code: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metric_value: Optional[float] = None

    # Human approval
    approved: Optional[bool] = None
    approval_note: str = ""

    error: str = ""                     # Top-level error if iteration failed


class RunHistory:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = run_dir
        self.run_id = run_id
        self.path = run_dir / f"{run_id}.jsonl"
        self.records: List[IterationRecord] = []
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def add(self, record: IterationRecord) -> None:
        self.records.append(record)
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def best_record(self, direction: str) -> Optional[IterationRecord]:
        completed = [r for r in self.records if r.metric_value is not None]
        if not completed:
            return None
        key = max if direction == "maximize" else min
        return key(completed, key=lambda r: r.metric_value)

    def metric_history(self) -> List[Optional[float]]:
        return [r.metric_value for r in self.records]

    def summary(self) -> str:
        lines = [f"Run: {self.run_id}  ({len(self.records)} iterations)"]
        for r in self.records:
            val = f"{r.metric_value:.4f}" if r.metric_value is not None else "N/A"
            err = f"  ERROR: {r.error}" if r.error else ""
            approved = "" if r.approved is None else (" ✓" if r.approved else " ✗ rejected")
            lines.append(f"  Iter {r.iteration:02d}: {val}{approved}{err}")
        return "\n".join(lines)
