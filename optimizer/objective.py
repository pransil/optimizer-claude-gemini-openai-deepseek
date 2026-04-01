"""
Defines the Objective dataclass that describes what the optimizer is trying to achieve.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


@dataclass
class StoppingConfig:
    max_iterations: int = 10
    target_metric: Optional[float] = None        # Stop if metric reaches this value
    plateau_patience: Optional[int] = None       # Stop if no improvement for N iterations
    min_improvement: float = 0.001               # Minimum delta to count as improvement


@dataclass
class Objective:
    """
    Describes the optimization goal.

    Example:
        objective = Objective(
            description="Train a PyTorch MLP on MNIST to maximize validation accuracy",
            metric_name="val_accuracy",
            direction="maximize",
            dataset_description="MNIST handwritten digits, 60k train / 10k test",
            stopping=StoppingConfig(max_iterations=5, target_metric=0.99),
            approval_mode="first_only",
        )
    """
    description: str
    metric_name: str                             # Key the sandbox code must write to results JSON
    direction: Literal["maximize", "minimize"] = "maximize"
    dataset_description: str = ""               # Extra context for the AI agents
    constraints: str = ""                       # e.g. "no external data, CPU only, <60s runtime"
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    approval_mode: Literal["always", "first_only", "never"] = "first_only"

    def is_better(self, new_value: float, old_value: float) -> bool:
        if self.direction == "maximize":
            return new_value > old_value + self.stopping.min_improvement
        return new_value < old_value - self.stopping.min_improvement

    def met_target(self, value: float) -> bool:
        if self.stopping.target_metric is None:
            return False
        if self.direction == "maximize":
            return value >= self.stopping.target_metric
        return value <= self.stopping.target_metric
