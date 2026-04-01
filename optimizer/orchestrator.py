"""
Orchestrator: the main optimization loop.

Flow per iteration:
  1. Claude proposes a solution (initial or next)
  2. CriticPanel reviews it in parallel (GPT-4, Gemini, DeepSeek)
  3. Claude synthesizes critiques → refined proposal
  4. [Optional] Human approval
  5. Sandbox runs the code, captures metrics
  6. Check stopping conditions
  7. Repeat
"""

import datetime
import time
from pathlib import Path
from typing import Optional

from .claude_agent import ClaudeAgent, Proposal
from .critic_panel import CriticPanel, CritiqueRequest
from .objective import Objective
from .run_history import IterationRecord, RunHistory
from .sandbox import Sandbox


def _print_header(text: str) -> None:
    width = 72
    print("\n" + "─" * width)
    print(f"  {text}")
    print("─" * width)


def _print_proposal(proposal: Proposal) -> None:
    print("\n📋 RATIONALE:")
    print(proposal.rationale)
    print("\n💻 CODE:")
    print("```python")
    print(proposal.code[:3000] + ("..." if len(proposal.code) > 3000 else ""))
    print("```")


def _human_approve(proposal: Proposal, iteration: int) -> tuple[bool, str]:
    print(f"\n{'='*72}")
    print(f"  Human Approval Required — Iteration {iteration}")
    print(f"{'='*72}")
    _print_proposal(proposal)
    print("\nOptions: [y] approve  [n] reject  [q] quit")
    while True:
        choice = input("Your choice: ").strip().lower()
        if choice in ("y", "yes"):
            note = input("Optional note (enter to skip): ").strip()
            return True, note
        elif choice in ("n", "no"):
            note = input("Reason for rejection (enter to skip): ").strip()
            return False, note
        elif choice in ("q", "quit"):
            raise KeyboardInterrupt("User quit during approval.")
        print("Please enter y, n, or q.")


class Orchestrator:
    def __init__(
        self,
        objective: Objective,
        runs_dir: Path = Path("runs"),
        sandbox_timeout: int = 300,
        claude_model: str = "claude-opus-4-5",
        verbose: bool = True,
    ):
        self.objective = objective
        self.verbose = verbose

        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history = RunHistory(runs_dir, run_id)
        self.sandbox = Sandbox(timeout_seconds=sandbox_timeout)
        self.agent = ClaudeAgent(model=claude_model)
        self.critic_panel = CriticPanel()

        self._iteration_history = []   # list of dicts passed to agent.next_proposal
        self._best_value: Optional[float] = None
        self._plateau_count: int = 0

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> RunHistory:
        obj = self.objective
        stop = obj.stopping

        print(f"\n🚀 Starting ML Optimizer")
        print(f"   Objective : {obj.description}")
        print(f"   Metric    : {obj.metric_name} ({obj.direction})")
        print(f"   Max iters : {stop.max_iterations}")
        print(f"   Approval  : {obj.approval_mode}")

        for iteration in range(1, stop.max_iterations + 1):
            _print_header(f"Iteration {iteration} / {stop.max_iterations}")
            record = IterationRecord(iteration=iteration)

            try:
                self._run_iteration(iteration, record)
            except KeyboardInterrupt:
                print("\n⛔ Interrupted by user.")
                self.history.add(record)
                break
            except Exception as e:
                import traceback
                record.error = str(e)
                print(f"\n❌ Iteration {iteration} failed: {e}")
                traceback.print_exc()

            self.history.add(record)

            # ── Stopping checks ───────────────────────────────────────────
            if record.metric_value is not None:
                if obj.met_target(record.metric_value):
                    print(f"\n🎯 Target metric reached ({record.metric_value:.4f}). Stopping.")
                    break

                if self._best_value is None or obj.is_better(record.metric_value, self._best_value):
                    self._best_value = record.metric_value
                    self._plateau_count = 0
                else:
                    self._plateau_count += 1

                if (stop.plateau_patience is not None
                        and self._plateau_count >= stop.plateau_patience):
                    print(f"\n📉 No improvement for {stop.plateau_patience} iterations. Stopping.")
                    break

        print("\n" + "=" * 72)
        print(self.history.summary())
        best = self.history.best_record(obj.direction)
        if best:
            print(f"\n🏆 Best: iteration {best.iteration} → {obj.metric_name} = {best.metric_value:.4f}")
        print(f"📁 Full log: {self.history.path}")

        return self.history

    # ── Single iteration ──────────────────────────────────────────────────────

    def _run_iteration(self, iteration: int, record: IterationRecord) -> None:
        obj = self.objective

        # 1. Generate proposal
        if iteration == 1:
            print("\n⚙️  Claude generating initial proposal...")
            proposal = self.agent.initial_proposal(
                objective_description=obj.description,
                metric_name=obj.metric_name,
                direction=obj.direction,
                dataset_description=obj.dataset_description,
                constraints=obj.constraints,
            )
        else:
            last = self._iteration_history[-1]
            print("\n⚙️  Claude generating next proposal based on results...")
            proposal = self.agent.next_proposal(
                objective_description=obj.description,
                metric_name=obj.metric_name,
                direction=obj.direction,
                dataset_description=obj.dataset_description,
                constraints=obj.constraints,
                history=self._iteration_history,
                last_stdout=last.get("stdout", ""),
                last_stderr=last.get("stderr", ""),
                last_error=last.get("error", ""),
            )

        record.initial_proposal = proposal.rationale
        record.initial_code = proposal.code

        if self.verbose:
            _print_header(f"Initial Proposal (iteration {iteration})")
            _print_proposal(proposal)

        # 2. Critic panel
        print("\n🔍 Consulting critic panel...")
        prev_results = self._format_previous_results()
        critique_req = CritiqueRequest(
            objective_description=obj.description,
            metric_name=obj.metric_name,
            direction=obj.direction,
            constraints=obj.constraints,
            proposal_rationale=proposal.rationale,
            proposal_code=proposal.code,
            iteration=iteration,
            previous_results=prev_results,
        )
        critiques = self.critic_panel.critique(critique_req)
        record.critiques = critiques

        if self.verbose and critiques:
            for name, text in critiques.items():
                print(f"\n  [{name}] {text[:400]}{'...' if len(text) > 400 else ''}")

        # 3. Claude synthesizes critiques → refined proposal
        print("\n⚙️  Claude synthesizing critiques...")
        refined = self.agent.synthesize_critiques(
            proposal=proposal,
            critiques=critiques,
            objective_description=obj.description,
            metric_name=obj.metric_name,
            direction=obj.direction,
            constraints=obj.constraints,
            iteration=iteration,
        )
        record.refined_proposal = refined.rationale
        record.refined_code = refined.code

        if self.verbose:
            _print_header(f"Refined Proposal (iteration {iteration})")
            _print_proposal(refined)

        # 4. Human approval (if required)
        needs_approval = (
            obj.approval_mode == "always"
            or (obj.approval_mode == "first_only" and iteration == 1)
        )
        if needs_approval:
            approved, note = _human_approve(refined, iteration)
            record.approved = approved
            record.approval_note = note
            if not approved:
                print(f"  ⛔ Proposal rejected. Skipping execution.")
                record.error = f"Rejected by user: {note}"
                return
        else:
            record.approved = True

        # 5. Run in sandbox
        print(f"\n🏃 Running code in sandbox (timeout={self.sandbox.timeout}s)...")
        t0 = time.time()
        result = self.sandbox.run(refined.code, obj.metric_name)
        elapsed = time.time() - t0

        record.sandbox_stdout = result.stdout
        record.sandbox_stderr = result.stderr
        record.sandbox_exit_code = result.exit_code
        record.metrics = result.metrics
        record.error = result.error

        print(f"  Exit code : {result.exit_code}  ({elapsed:.1f}s)")
        if result.stdout:
            print(f"  Stdout    :\n{result.stdout[-1500:]}")
        if result.stderr:
            print(f"  Stderr    :\n{result.stderr[-500:]}")
        if result.error:
            print(f"  ⚠️  Error  : {result.error}")

        # Extract primary metric
        if obj.metric_name in result.metrics:
            record.metric_value = float(result.metrics[obj.metric_name])
            print(f"\n  📊 {obj.metric_name} = {record.metric_value:.4f}")
            if result.metrics:
                others = {k: v for k, v in result.metrics.items() if k != obj.metric_name}
                if others:
                    print(f"     Other metrics: {others}")
        elif not result.error:
            record.error = (
                f"Metric '{obj.metric_name}' not found in results. "
                f"Got keys: {list(result.metrics.keys())}"
            )
            print(f"  ⚠️  {record.error}")

        # Update history for next proposal
        self._iteration_history.append({
            "iteration": iteration,
            "metric_name": obj.metric_name,
            "metric_value": record.metric_value,
            "rationale": refined.rationale,
            "outcome": (
                f"{obj.metric_name}={record.metric_value:.4f}" if record.metric_value is not None
                else f"Error: {record.error}"
            ),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
        })

    def _format_previous_results(self) -> str:
        if not self._iteration_history:
            return ""
        lines = []
        for h in self._iteration_history:
            val = f"{h['metric_value']:.4f}" if h.get("metric_value") is not None else "N/A"
            lines.append(f"Iter {h['iteration']}: {h['metric_name']} = {val}")
        return "\n".join(lines)
