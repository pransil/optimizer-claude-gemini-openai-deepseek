"""
Executes AI-generated Python code in an isolated subprocess with a timeout.
The generated code must write its results to a JSON file at the path provided
via the RESULTS_PATH environment variable.

Expected results JSON format:
{
    "metric_name": <float>,     # The primary metric (must match objective.metric_name)
    ...                         # Any additional metrics are stored too
}
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    metrics: Dict[str, Any]
    error: str = ""


SANDBOX_HEADER = '''
import os, json, sys

RESULTS_PATH = os.environ.get("RESULTS_PATH", "/tmp/results.json")

def write_results(metrics: dict):
    """Call this at the end of your script to record metrics."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f)

# ── Generated code below ──────────────────────────────────────────────────────
'''


class Sandbox:
    def __init__(self, timeout_seconds: int = 300, work_dir: Optional[Path] = None):
        self.timeout = timeout_seconds
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="ml_optimizer_"))

    def run(self, code: str, metric_name: str) -> SandboxResult:
        """
        Write code to a temp file, execute it in a subprocess, read back results.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Write the code file
        code_path = self.work_dir / "solution.py"
        results_path = self.work_dir / "results.json"

        # Remove any stale results
        if results_path.exists():
            results_path.unlink()

        full_code = SANDBOX_HEADER + "\n" + code
        code_path.write_text(full_code)

        env = os.environ.copy()
        env["RESULTS_PATH"] = str(results_path)

        try:
            proc = subprocess.run(
                [sys.executable, str(code_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=str(self.work_dir),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr="",
                metrics={},
                error=f"Execution timed out after {self.timeout}s",
            )
        except Exception as e:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr="",
                metrics={},
                error=f"Subprocess error: {e}",
            )

        # Parse results
        metrics: Dict[str, Any] = {}
        error = ""

        if results_path.exists():
            try:
                metrics = json.loads(results_path.read_text())
            except json.JSONDecodeError as e:
                error = f"Could not parse results JSON: {e}"
        else:
            if proc.returncode == 0:
                error = (
                    f"Code ran successfully but did not call write_results(). "
                    f"Ensure your code calls write_results({{'{ metric_name }': <value>}}) "
                    f"at the end."
                )
            else:
                error = "Code failed to run (see stderr)."

        if proc.returncode != 0 and not error:
            error = f"Process exited with code {proc.returncode}"

        return SandboxResult(
            exit_code=proc.returncode,
            stdout=proc.stdout[-8000:],   # cap to avoid huge logs
            stderr=proc.stderr[-4000:],
            metrics=metrics,
            error=error,
        )
