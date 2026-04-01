"""
Example: optimize a PyTorch MLP on MNIST to maximize validation accuracy.

Run:
    cd ml_optimizer
    python example_run.py

API keys (set whichever you have — missing ones are skipped):
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...        # optional
    export GEMINI_API_KEY=...        # optional
    export DEEPSEEK_API_KEY=...      # optional
"""

from pathlib import Path
from optimizer import Objective, Orchestrator, StoppingConfig

objective = Objective(
    description=(
        "Train a PyTorch MLP (fully connected neural network) on the MNIST "
        "handwritten digit dataset to maximize validation accuracy. "
        "Use torchvision to load MNIST. Train for a reasonable number of epochs."
    ),
    metric_name="val_accuracy",
    direction="maximize",
    dataset_description=(
        "MNIST: 60,000 training images and 10,000 test images of handwritten digits (0–9), "
        "28x28 grayscale. Available via torchvision.datasets.MNIST."
    ),
    constraints=(
        "Use MPS (Apple Metal) if available: device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu'). "
        "Total runtime must be under 4 minutes. "
        "Use only PyTorch and torchvision — no other ML libraries. "
        "val_accuracy must be a float between 0 and 1."
    ),
    stopping=StoppingConfig(
        max_iterations=4,
        target_metric=0.98,
        plateau_patience=2,
        min_improvement=0.002,
    ),
    approval_mode="never",   # "always" | "first_only" | "never"
)

orchestrator = Orchestrator(
    objective=objective,
    runs_dir=Path("runs"),
    sandbox_timeout=260,
    verbose=True,
)

history = orchestrator.run()
