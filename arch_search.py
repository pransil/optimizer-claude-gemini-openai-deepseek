"""
Architecture search on MNIST: tests 3 FC and 3 CNN architectures.
After each run, consults Claude + all available AI critics (GPT-4, Gemini, DeepSeek).
Each architecture proposal is informed by results and feedback from prior runs.
Prints a summary table at the end.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anthropic

from optimizer.claude_agent import ClaudeAgent
from optimizer.critic_panel import ALL_CRITICS, BaseCritic, CriticPanel, CritiqueRequest
from optimizer.sandbox import Sandbox


# ── Claude as a critic (separate from the proposer) ───────────────────────────

class ClaudeCritic(BaseCritic):
    name = "claude"
    env_key = "ANTHROPIC_API_KEY"

    def critique(self, req: CritiqueRequest) -> str:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": self._build_prompt(req)}],
        )
        return resp.content[0].text.strip()


# ── Configuration ─────────────────────────────────────────────────────────────

METRIC = "val_accuracy"
DATASET_DESC = (
    "MNIST: 60,000 training images and 10,000 test images of handwritten "
    "digits (0-9), 28x28 grayscale. Available via torchvision.datasets.MNIST."
)
BASE_CONSTRAINTS = (
    "Use MPS if available: device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu'). "
    "Total runtime must be under 4 minutes. "
    "Use only PyTorch and torchvision — no other ML libraries. "
    "val_accuracy must be a float between 0 and 1. "
    "Call write_results({'val_accuracy': <float>}) at the very end of the script."
)

ARCH_SPECS = [
    {
        "name": "FC-Shallow",
        "family": "Fully Connected",
        "desc": (
            "A shallow fully-connected MLP with exactly 2 hidden layers. "
            "Keep it small — try widths like 256 and 128. "
            "Use ReLU activations and light dropout (0.1-0.2). "
            "Use Adam optimizer."
        ),
    },
    {
        "name": "FC-Medium",
        "family": "Fully Connected",
        "desc": (
            "A medium fully-connected MLP with exactly 3 hidden layers. "
            "Add BatchNorm1d after each Linear layer. "
            "Try widths around 512-256-128. "
            "Include a StepLR or CosineAnnealingLR scheduler."
        ),
    },
    {
        "name": "FC-Deep",
        "family": "Fully Connected",
        "desc": (
            "A deeper fully-connected MLP with 4 or more hidden layers. "
            "Use BatchNorm1d, Dropout, and a cosine LR scheduler. "
            "Try widths like 1024-512-256-128. "
            "Apply Kaiming weight initialization."
        ),
    },
    {
        "name": "CNN-Simple",
        "family": "CNN",
        "desc": (
            "A simple CNN with exactly 2 convolutional layers followed by FC layers. "
            "Use ReLU and MaxPool2d after each conv. "
            "Keep channels modest (e.g. 32, 64). "
            "Flatten before the FC head."
        ),
    },
    {
        "name": "CNN-Medium",
        "family": "CNN",
        "desc": (
            "A medium CNN with 3 convolutional layers, BatchNorm2d after each conv, "
            "and a 2-layer FC head. "
            "Use Dropout before the final classifier. "
            "Try channels like 32-64-128 with MaxPool after layer 2."
        ),
    },
    {
        "name": "CNN-Deep",
        "family": "CNN",
        "desc": (
            "A deeper CNN with 4 convolutional layers or residual-style skip connections. "
            "Use BatchNorm2d throughout, AdaptiveAvgPool2d for the pooling, "
            "and an efficient 1-2 layer FC head. "
            "Aim for high accuracy with a lean parameter count."
        ),
    },
]


# ── Result record ─────────────────────────────────────────────────────────────

@dataclass
class ArchResult:
    name: str
    family: str
    val_accuracy: Optional[float]
    train_time_s: float
    error: str = ""
    critiques: Dict[str, str] = field(default_factory=dict)
    code: str = ""  # generated code, used for post-run architecture analysis


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_prior_results(results: List[ArchResult]) -> str:
    if not results:
        return ""
    lines = ["Prior architecture results:"]
    for r in results:
        acc = f"{r.val_accuracy:.4f}" if r.val_accuracy is not None else "N/A"
        lines.append(f"  {r.name} ({r.family}): val_accuracy={acc}, time={r.train_time_s:.1f}s")
    return "\n".join(lines)


def _format_prior_feedback(results: List[ArchResult]) -> str:
    """Collect actionable critique highlights from previous runs."""
    parts = []
    for r in results:
        if not r.critiques:
            continue
        acc = f"{r.val_accuracy:.4f}" if r.val_accuracy is not None else "N/A"
        parts.append(f"--- Feedback for {r.name} (val_accuracy={acc}) ---")
        for critic_name, text in r.critiques.items():
            if not text.startswith("[ERROR]"):
                parts.append(f"[{critic_name}]: {text[:350]}")
    return "\n\n".join(parts) if parts else ""


def _separator(title: str = "") -> None:
    w = 72
    print("\n" + "─" * w)
    if title:
        print(f"  {title}")
        print("─" * w)


def print_arch_details(results: List[ArchResult]) -> None:
    """Ask Claude to extract structured architecture details from each model's code."""
    client = anthropic.Anthropic()

    print("\n" + "=" * 80)
    print("  ARCHITECTURE DETAILS")
    print("=" * 80)

    for r in results:
        print(f"\n{'─' * 72}")
        print(f"  {r.name}  [{r.family}]")
        print(f"{'─' * 72}")

        if not r.code:
            print("  (no code available)")
            continue

        prompt = f"""Given this PyTorch model code, extract and list the architecture details concisely.

Include:
- Model type (MLP / CNN / ResNet-style / etc.)
- Number of layers (conv, linear, etc.) with their sizes/channels
- Activations used
- Regularization (BatchNorm, Dropout rates)
- Optimizer and learning rate
- LR scheduler (if any)
- Estimated parameter count (rough order of magnitude is fine)
- Any notable design choices

Format as a clean bullet list. Be concise — no preamble.

CODE:
```python
{r.code[:6000]}
```"""

        resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        details = resp.content[0].text.strip()
        # Indent each line for clean formatting
        for line in details.splitlines():
            print(f"  {line}")

    print()


def print_table(results: List[ArchResult]) -> None:
    w = 80
    print("\n" + "=" * w)
    print("  ARCHITECTURE SEARCH — FINAL RESULTS")
    print("=" * w)
    print(f"{'Architecture':<16}  {'Family':<18}  {'Val Accuracy':>13}  {'Train Time':>11}  Notes")
    print("-" * w)
    for r in results:
        acc = f"{r.val_accuracy:.4f}" if r.val_accuracy is not None else "   N/A  "
        t = f"{r.train_time_s:.1f}s"
        note = (r.error[:28] + "…") if r.error else "✓"
        print(f"{r.name:<16}  {r.family:<18}  {acc:>13}  {t:>11}  {note}")
    print("=" * w)

    completed = [r for r in results if r.val_accuracy is not None]
    if completed:
        best = max(completed, key=lambda r: r.val_accuracy)
        fastest = min(completed, key=lambda r: r.train_time_s)
        print(f"\n  🏆 Best accuracy : {best.name} → {best.val_accuracy:.4f}")
        print(f"  ⚡ Fastest       : {fastest.name} → {fastest.train_time_s:.1f}s")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    agent = ClaudeAgent(model="claude-opus-4-6")
    critic_panel = CriticPanel(critics=[ClaudeCritic()] + ALL_CRITICS)
    sandbox = Sandbox(timeout_seconds=260)

    results: List[ArchResult] = []

    critic_names = ["claude"] + [c.name for c in ALL_CRITICS]
    print("\n🔬 Architecture Search: 3 FC + 3 CNN on MNIST")
    print(f"   Critics    : {', '.join(critic_names)}")
    print(f"   Device     : MPS (Apple Silicon)")
    print(f"   Sandbox    : {sandbox.timeout}s timeout per run")

    for i, spec in enumerate(ARCH_SPECS, 1):
        _separator(f"Architecture {i}/6 — {spec['name']}  [{spec['family']}]")

        prior_results_str = _format_prior_results(results)
        prior_feedback_str = _format_prior_feedback(results)

        # Objective embeds arch spec + accumulated context from prior runs
        arch_objective = (
            f"Train a PyTorch neural network on MNIST to maximize val_accuracy.\n\n"
            f"ARCHITECTURE REQUIREMENT:\n{spec['desc']}\n"
            + (f"\n{prior_results_str}\n" if prior_results_str else "")
            + (f"\nFEEDBACK FROM PRIOR ARCHITECTURES (use this to improve your design):\n{prior_feedback_str}" if prior_feedback_str else "")
        )

        # Generate proposal
        print(f"\n⚙️  Claude generating {spec['name']} proposal...")
        proposal = agent.initial_proposal(
            objective_description=arch_objective,
            metric_name=METRIC,
            direction="maximize",
            dataset_description=DATASET_DESC,
            constraints=BASE_CONSTRAINTS,
        )
        print(f"\n📋 Rationale (summary):\n   {proposal.rationale[:400].replace(chr(10), chr(10) + '   ')}")

        # Run in sandbox
        print(f"\n🏃 Running in sandbox (timeout={sandbox.timeout}s)...")
        t0 = time.time()
        sandbox_result = sandbox.run(proposal.code, METRIC)
        train_time = time.time() - t0

        val_acc = sandbox_result.metrics.get(METRIC)
        acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        print(f"  Exit code  : {sandbox_result.exit_code}  ({train_time:.1f}s)")
        if sandbox_result.stdout:
            print(f"  Output     :\n{sandbox_result.stdout[-1000:]}")
        if sandbox_result.error:
            print(f"  ⚠️  Error   : {sandbox_result.error}")
        print(f"  📊 {METRIC} = {acc_str}")

        # Consult all critics in parallel
        print(f"\n🔍 Consulting all AI critics ({', '.join(critic_names)})...")
        critique_req = CritiqueRequest(
            objective_description=f"Train a PyTorch {spec['family']} network on MNIST — maximize val_accuracy.",
            metric_name=METRIC,
            direction="maximize",
            constraints=BASE_CONSTRAINTS,
            proposal_rationale=proposal.rationale,
            proposal_code=proposal.code,
            iteration=i,
            previous_results=prior_results_str,
        )
        critiques = critic_panel.critique(critique_req)

        if critiques:
            print("\n  AI Feedback:")
            for critic_name, text in critiques.items():
                if text.startswith("[ERROR]"):
                    print(f"  [{critic_name}]: (unavailable)")
                else:
                    snippet = text[:250].replace("\n", " ")
                    print(f"  [{critic_name}]: {snippet}…")

        results.append(ArchResult(
            name=spec["name"],
            family=spec["family"],
            val_accuracy=val_acc,
            train_time_s=train_time,
            error=sandbox_result.error,
            critiques=critiques,
            code=proposal.code,
        ))

    print_table(results)
    print_arch_details(results)


if __name__ == "__main__":
    main()
