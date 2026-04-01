"""
CriticPanel: sends a proposal to GPT-4, Gemini, and DeepSeek in parallel
and collects structured critiques.

Each critic that lacks an API key is skipped gracefully with a logged warning.
"""

import concurrent.futures
import os
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional


# ── Critic base class ─────────────────────────────────────────────────────────

@dataclass
class CritiqueRequest:
    objective_description: str
    metric_name: str
    direction: str
    constraints: str
    proposal_rationale: str
    proposal_code: str
    iteration: int
    previous_results: str   # formatted summary of past runs, empty on first iter


class BaseCritic:
    name: str = "base"
    env_key: str = ""

    def available(self) -> bool:
        return bool(os.environ.get(self.env_key))

    def critique(self, req: CritiqueRequest) -> str:
        raise NotImplementedError

    def _build_prompt(self, req: CritiqueRequest) -> str:
        prev = (
            f"\n\nPrevious iteration results:\n{req.previous_results}"
            if req.previous_results else ""
        )
        return f"""You are an expert ML engineer reviewing an optimization proposal.

OBJECTIVE: {req.objective_description}
METRIC: {req.metric_name} ({req.direction})
CONSTRAINTS: {req.constraints or "None specified"}
ITERATION: {req.iteration}{prev}

PROPOSAL RATIONALE:
{req.proposal_rationale}

PROPOSED CODE:
```python
{req.proposal_code}
```

Provide a concise, actionable critique covering:
1. Correctness — will the code run and measure the right thing?
2. ML quality — are the hyperparameters, architecture, and training loop sensible?
3. Improvement potential — what specific changes would most improve the metric?
4. Any bugs or risks.

Be specific. Suggest concrete values where possible. Keep your response under 400 words."""


# ── OpenAI (GPT-4) ────────────────────────────────────────────────────────────

class OpenAICritic(BaseCritic):
    name = "gpt4"
    env_key = "OPENAI_API_KEY"

    def critique(self, req: CritiqueRequest) -> str:
        import openai
        client = openai.OpenAI(api_key=os.environ[self.env_key])
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": self._build_prompt(req)}],
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


# ── Google Gemini ─────────────────────────────────────────────────────────────

class GeminiCritic(BaseCritic):
    name = "gemini"
    env_key = "GEMINI_API_KEY"

    def critique(self, req: CritiqueRequest) -> str:
        from google import genai
        client = genai.Client(api_key=os.environ[self.env_key])
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=self._build_prompt(req),
        )
        return resp.text.strip()


# ── DeepSeek ──────────────────────────────────────────────────────────────────

class DeepSeekCritic(BaseCritic):
    name = "deepseek"
    env_key = "DEEPSEEK_API_KEY"

    def critique(self, req: CritiqueRequest) -> str:
        import openai  # DeepSeek uses an OpenAI-compatible API
        client = openai.OpenAI(
            api_key=os.environ[self.env_key],
            base_url="https://api.deepseek.com",
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": self._build_prompt(req)}],
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


# ── Panel ─────────────────────────────────────────────────────────────────────

ALL_CRITICS: List[BaseCritic] = [
    OpenAICritic(),
    GeminiCritic(),
    DeepSeekCritic(),
]


class CriticPanel:
    """Runs all available critics in parallel and returns their critiques."""

    def __init__(self, critics: Optional[List[BaseCritic]] = None):
        self.critics = critics if critics is not None else ALL_CRITICS

    def critique(self, req: CritiqueRequest) -> Dict[str, str]:
        """
        Returns dict of {critic_name: critique_text}.
        Critics with missing API keys are skipped.
        Critics that raise exceptions return an error string.
        """
        available = [c for c in self.critics if c.available()]
        skipped = [c.name for c in self.critics if not c.available()]

        if skipped:
            print(f"  [CriticPanel] Skipping (no API key): {', '.join(skipped)}")

        if not available:
            print("  [CriticPanel] No critics available — skipping critic stage.")
            return {}

        results: Dict[str, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(available)) as ex:
            future_to_critic = {ex.submit(c.critique, req): c for c in available}
            for future in concurrent.futures.as_completed(future_to_critic):
                critic = future_to_critic[future]
                try:
                    results[critic.name] = future.result()
                    print(f"  [CriticPanel] ✓ {critic.name}")
                except Exception:
                    tb = traceback.format_exc()
                    results[critic.name] = f"[ERROR]\n{tb}"
                    print(f"  [CriticPanel] ✗ {critic.name} failed")

        return results
