# Multi-Model ML Optimizer

An iterative ML optimization framework that uses Claude to propose and refine PyTorch training pipelines, with a panel of AI critics (GPT-4, Gemini, DeepSeek) providing feedback between iterations.

## How It Works

Each optimization cycle follows this flow:

```
Claude proposes a solution
        ↓
Critics review in parallel (GPT-4, Gemini, DeepSeek, Claude-as-critic)
        ↓
Claude synthesizes feedback → refined proposal
        ↓
Sandbox executes the code and captures metrics
        ↓
Results feed into the next iteration
```

The system is general-purpose — you define the objective, metric, dataset, and constraints, and the AI loop handles the rest.

## Project Structure

```
optimizer/
├── claude_agent.py     # Generates and refines proposals via Claude API
├── critic_panel.py     # Parallel critique from GPT-4, Gemini, DeepSeek
├── sandbox.py          # Executes generated code in an isolated subprocess
├── objective.py        # Objective + StoppingConfig dataclasses
├── orchestrator.py     # Main optimization loop
└── run_history.py      # Persists iteration records to JSONL

arch_search.py          # Searches 3 FC + 3 CNN architectures on MNIST
demo_interaction.py     # Shows the full proposal → critique → refinement cycle
example_run.py          # Simple single-objective run (MNIST MLP)
```

## Quickstart

### 1. Install dependencies

```bash
pip install anthropic openai google-genai torch torchvision
```

### 2. Set API keys

Create a `.env` file:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...      # optional
export GEMINI_API_KEY=...      # optional
export DEEPSEEK_API_KEY=...    # optional
```

Critics with missing keys are skipped gracefully — only Claude is required.

### 3. Run the examples

```bash
source .env

# Basic optimization loop
python example_run.py

# Architecture search: 3 FC + 3 CNN, all critics, summary table
python arch_search.py

# Demo: see the full AI interaction without running any training code
python demo_interaction.py
```

## Example Output

### Architecture Search (`arch_search.py`)

Tests 6 architectures on MNIST (3 fully connected, 3 CNN), consulting all AI critics after each run. Each architecture's proposal is informed by the accumulated feedback from all prior runs.

```
================================================================================
  ARCHITECTURE SEARCH — FINAL RESULTS
================================================================================
Architecture      Family               Val Accuracy   Train Time  Notes
--------------------------------------------------------------------------------
FC-Shallow        Fully Connected            0.9911       111.0s  ✓
FC-Medium         Fully Connected            0.9932       125.2s  ✓
FC-Deep           Fully Connected            0.9947       187.1s  ✓
CNN-Simple        CNN                        0.9961       194.2s  ✓
CNN-Medium        CNN                        0.9956       201.4s  ✓
CNN-Deep          CNN                        0.9972       201.8s  ✓
================================================================================

  🏆 Best accuracy : CNN-Deep → 99.72%
  ⚡ Fastest       : FC-Shallow → 111.0s
```

### Interaction Demo (`demo_interaction.py`)

Shows the full critique and refinement cycle for one architecture — without running any training code:

- **Step 1**: Claude generates an initial proposal
- **Step 2**: All critics review in parallel and flag issues
- **Step 3**: Claude synthesizes the feedback, accepts/rejects each suggestion with reasoning, and produces a refined design

Example of what changes between initial and refined proposal:
- Dropout 0.15 → 0.2 *(accepted from GPT-4, DeepSeek)*
- Added label smoothing 0.1 *(Claude's own suggestion)*
- Weight decay 1e-5 → 1e-4 *(DeepSeek)*
- Epochs 30 → 50 *(Claude)*
- AdamW rejected — constraint said Adam *(Gemini's suggestion, overruled)*
- `num_workers` / `pin_memory` kept at 0/False — MPS doesn't benefit *(Gemini's suggestion, overruled)*

## Defining Your Own Objective

```python
from optimizer import Objective, Orchestrator, StoppingConfig
from pathlib import Path

objective = Objective(
    description="Train a PyTorch model on CIFAR-10 to maximize validation accuracy",
    metric_name="val_accuracy",
    direction="maximize",
    dataset_description="CIFAR-10: 50k train / 10k test, 32x32 RGB images, 10 classes",
    constraints="Use MPS if available. Runtime under 5 minutes. PyTorch only.",
    stopping=StoppingConfig(
        max_iterations=6,
        target_metric=0.85,
        plateau_patience=2,
    ),
    approval_mode="never",  # "always" | "first_only" | "never"
)

orchestrator = Orchestrator(objective=objective, runs_dir=Path("runs"), verbose=True)
history = orchestrator.run()
```

## Hardware

Runs on Apple Silicon (MPS) out of the box. Generated code uses:

```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

CUDA support can be added by adjusting the device constraint in your objective.

## Notes

- All generated code runs in an isolated subprocess via `sandbox.py`
- Each run is logged to a JSONL file in `runs/` for auditability
- The sandbox injects a `write_results({"metric": value})` helper that generated code must call to report metrics
- Critics that lack API keys or hit quota errors are skipped without failing the run
