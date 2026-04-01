"""
ClaudeAgent: responsible for
  1. Generating an initial proposal (rationale + code)
  2. Synthesizing critic feedback into a refined proposal
  3. Reflecting on run results to generate the next proposal
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import anthropic


@dataclass
class Proposal:
    rationale: str
    code: str


SYSTEM_PROMPT = """You are an expert ML engineer and optimizer. You design, implement, and iteratively improve PyTorch training pipelines.

When asked to produce a proposal you MUST respond in exactly this format — no other text outside these tags:

<rationale>
Explain your approach, key design choices, and why you expect improvement over previous attempts.
</rationale>

<code>
# Pure Python / PyTorch code here.
# IMPORTANT: At the very end, call write_results({"<metric_name>": <float_value>, ...})
# write_results is pre-injected — do not define it yourself.
# Do not use any external data sources unless instructed.
# Keep runtime reasonable (under 5 minutes).
</code>"""


def _extract_tags(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _format_critiques(critiques: Dict[str, str]) -> str:
    if not critiques:
        return "No external critiques were available."
    parts = []
    for name, text in critiques.items():
        parts.append(f"=== {name.upper()} ===\n{text}")
    return "\n\n".join(parts)


def _format_history(history: List[dict]) -> str:
    if not history:
        return "No previous iterations."
    parts = []
    for h in history:
        val = f"{h['metric_value']:.4f}" if h.get("metric_value") is not None else "N/A"
        parts.append(
            f"Iteration {h['iteration']}: {h['metric_name']} = {val}\n"
            f"  Rationale summary: {h['rationale'][:300]}...\n"
            f"  Key outcome: {h.get('outcome', '')}"
        )
    return "\n\n".join(parts)


class ClaudeAgent:
    def __init__(self, model: str = "claude-opus-4-5"):
        self.client = anthropic.Anthropic()
        self.model = model

    def _call(self, messages: List[dict]) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text

    def initial_proposal(
        self,
        objective_description: str,
        metric_name: str,
        direction: str,
        dataset_description: str,
        constraints: str,
    ) -> Proposal:
        prompt = f"""Generate an initial ML solution for the following objective.

OBJECTIVE: {objective_description}
METRIC TO OPTIMIZE: {metric_name} ({direction})
DATASET: {dataset_description or "Not specified — choose an appropriate built-in dataset."}
CONSTRAINTS: {constraints or "None specified."}

Produce a solid baseline — clean, correct, and well-structured. Remember to call write_results() at the end."""

        text = self._call([{"role": "user", "content": prompt}])
        return Proposal(
            rationale=_extract_tags(text, "rationale") or text,
            code=_extract_tags(text, "code"),
        )

    def synthesize_critiques(
        self,
        proposal: Proposal,
        critiques: Dict[str, str],
        objective_description: str,
        metric_name: str,
        direction: str,
        constraints: str,
        iteration: int,
    ) -> Proposal:
        """Refine a proposal given external critiques."""
        if not critiques:
            return proposal  # Nothing to synthesize

        prompt = f"""You previously proposed the following solution (iteration {iteration}):

<rationale>
{proposal.rationale}
</rationale>

<code>
{proposal.code}
</code>

You have received the following critiques from other AI systems:

{_format_critiques(critiques)}

OBJECTIVE: {objective_description}
METRIC: {metric_name} ({direction})
CONSTRAINTS: {constraints or "None specified."}

Carefully consider the critiques. Accept suggestions that are clearly beneficial, reject those that are misguided, and explain your reasoning. Then produce a refined proposal that incorporates the best feedback.

Respond in the same <rationale> / <code> format."""

        text = self._call([{"role": "user", "content": prompt}])
        return Proposal(
            rationale=_extract_tags(text, "rationale") or text,
            code=_extract_tags(text, "code") or proposal.code,
        )

    def next_proposal(
        self,
        objective_description: str,
        metric_name: str,
        direction: str,
        dataset_description: str,
        constraints: str,
        history: List[dict],
        last_stdout: str,
        last_stderr: str,
        last_error: str,
    ) -> Proposal:
        """Generate the next proposal after seeing run results."""
        prompt = f"""You are iteratively optimizing an ML solution.

OBJECTIVE: {objective_description}
METRIC: {metric_name} ({direction})
DATASET: {dataset_description or "Same as before."}
CONSTRAINTS: {constraints or "None specified."}

HISTORY OF ITERATIONS:
{_format_history(history)}

LAST RUN OUTPUT (stdout):
{last_stdout[-3000:] if last_stdout else "(empty)"}

LAST RUN ERRORS:
{last_stderr[-1000:] if last_stderr else "(none)"}

{"EXECUTION ERROR: " + last_error if last_error else ""}

Based on the results so far, propose a meaningfully improved solution. Be specific about what you are changing and why you expect it to help. Remember to call write_results() at the end."""

        text = self._call([{"role": "user", "content": prompt}])
        return Proposal(
            rationale=_extract_tags(text, "rationale") or text,
            code=_extract_tags(text, "code"),
        )
