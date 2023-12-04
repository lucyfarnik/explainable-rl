from dataclasses import dataclass, field
from numpy import ndarray
from torch import Tensor


@dataclass
class Step:
    time_step: int
    obs: Tensor
    reward: float
    action: int
    render: ndarray

    def __str__(self) -> str:
        return f"t={self.time_step}"


@dataclass
class Episode:
    steps: list[Step] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    outcome: str = field(default="")

    def length(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        parameter_string = ", ".join(
            [f"{key}={value:.2f}" for key, value in self.parameters.items()]
        )
        return f"Episode: {self.outcome} - {self.length()} steps"
