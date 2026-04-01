"""
Demo: shows the full AI interaction loop for one architecture.

Flow:
  1. Claude generates an initial proposal
  2. GPT-4, Gemini, DeepSeek, and Claude-as-critic all review it in parallel
  3. Claude synthesizes the critiques into a refined proposal
  4. Side-by-side diff shows what changed and why

Stops before running any code in the sandbox.
"""

import os
from optimizer.claude_agent import ClaudeAgent
from optimizer.critic_panel import CriticPanel, CritiqueRequest, ALL_CRITICS, BaseCritic
import anthropic

METRIC = "val_accuracy"
DATASET_DESC = (
    "MNIST: 60,000 training images and 10,000 test images of handwritten "
    "digits (0-9), 28x28 grayscale. Available via torchvision.datasets.MNIST."
)
CONSTRAINTS = (
    "Use MPS if available: device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu'). "
    "Total runtime must be under 4 minutes. "
    "Use only PyTorch and torchvision — no other ML libraries. "
    "val_accuracy must be a float between 0 and 1."
)
ARCH_DESC = (
    "A shallow fully-connected MLP with exactly 2 hidden layers. "
    "Keep it small — try widths like 256 and 128. "
    "Use ReLU activations and light dropout. Use Adam optimizer."
)


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


def div(title="", width=72):
    print("\n" + "═" * width)
    if title:
        print(f"  {title}")
        print("═" * width)


def section(title, width=72):
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)


def print_proposal(proposal, label):
    section(label)
    print("\n📋 RATIONALE:\n")
    print(proposal.rationale)
    print("\n💻 CODE (first 60 lines):\n")
    lines = proposal.code.splitlines()
    print("\n".join(lines[:60]))
    if len(lines) > 60:
        print(f"\n  ... ({len(lines) - 60} more lines)")


def diff_rationales(original, refined):
    """Highlight key differences between the two rationales."""
    div("WHAT CHANGED — INITIAL vs REFINED")

    orig_lines = set(original.lower().split("."))
    refined_lines = refined.split(".")

    print("\n📝 REFINED RATIONALE:\n")
    for sent in refined_lines:
        sent = sent.strip()
        if not sent:
            continue
        # Rough heuristic: if this sentence isn't in the original, mark it as new
        is_new = sent.lower() not in original.lower()
        marker = "  ➕ " if is_new else "     "
        print(f"{marker}{sent}.")

    print("\n  Legend:  ➕ = new or changed from original   (unmarked = unchanged)")


def main():
    agent = ClaudeAgent(model="claude-opus-4-6")
    critic_panel = CriticPanel(critics=[ClaudeCritic()] + ALL_CRITICS)

    objective = (
        f"Train a PyTorch neural network on MNIST to maximize val_accuracy.\n\n"
        f"ARCHITECTURE REQUIREMENT:\n{ARCH_DESC}"
    )

    # ── Step 1: Initial proposal ───────────────────────────────────────────────
    div("STEP 1 — CLAUDE'S INITIAL PROPOSAL")
    print("\n⚙️  Claude generating initial FC-Shallow proposal...")

    proposal = agent.initial_proposal(
        objective_description=objective,
        metric_name=METRIC,
        direction="maximize",
        dataset_description=DATASET_DESC,
        constraints=CONSTRAINTS,
    )
    print_proposal(proposal, "Initial Proposal")

    # ── Step 2: All critics review in parallel ─────────────────────────────────
    div("STEP 2 — CRITIC PANEL REVIEWS IN PARALLEL")
    print(f"\n🔍 Sending to: Claude, GPT-4, Gemini, DeepSeek...\n")

    critique_req = CritiqueRequest(
        objective_description="Train a PyTorch FC MLP on MNIST — maximize val_accuracy.",
        metric_name=METRIC,
        direction="maximize",
        constraints=CONSTRAINTS,
        proposal_rationale=proposal.rationale,
        proposal_code=proposal.code,
        iteration=1,
        previous_results="",
    )
    critiques = critic_panel.critique(critique_req)

    print()
    for critic_name, text in critiques.items():
        section(f"[{critic_name.upper()}] CRITIQUE")
        if text.startswith("[ERROR]"):
            print("  (unavailable)")
        else:
            print(text)

    # ── Step 3: Claude synthesizes critiques → refined proposal ────────────────
    div("STEP 3 — CLAUDE SYNTHESIZES FEEDBACK → REFINED PROPOSAL")
    print("\n⚙️  Claude reading all critiques and refining the design...")

    refined = agent.synthesize_critiques(
        proposal=proposal,
        critiques={k: v for k, v in critiques.items() if not v.startswith("[ERROR]")},
        objective_description=objective,
        metric_name=METRIC,
        direction="maximize",
        constraints=CONSTRAINTS,
        iteration=1,
    )
    print_proposal(refined, "Refined Proposal")

    # ── Step 4: Show what changed ──────────────────────────────────────────────
    diff_rationales(proposal.rationale, refined.rationale)

    div("SUMMARY")
    orig_lines = len(proposal.code.splitlines())
    refined_lines = len(refined.code.splitlines())
    print(f"\n  Initial proposal : {orig_lines} lines of code")
    print(f"  Refined proposal : {refined_lines} lines of code  ({refined_lines - orig_lines:+d})")
    print(f"\n  Critics consulted: {', '.join(critiques.keys())}")
    print(f"  Critics succeeded: {', '.join(k for k, v in critiques.items() if not v.startswith('[ERROR]'))}")
    print()


if __name__ == "__main__":
    main()
