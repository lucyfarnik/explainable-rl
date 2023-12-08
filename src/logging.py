"""Dataclasses for structuring data from test runs of an agent in an environment."""
from dataclasses import dataclass, field
from numpy import ndarray
from torch import Tensor


@dataclass
class Step:
    """A single time step of an episode. A collection of data from a single step of an agent in an environment.

    Attributes:
        time_step (int): The time step of the environment when the step was taken. (0-indexed). Ordinal, not in seconds.
        obs (Tensor): The observation of the environment when the step was taken. Depends on the environment.
        reward (float): The reward given to the agent from the environment when the step was taken.
        action (int): The action taken by the agent when the step was taken. Corresponds to the environment's action space.
        render (ndarray): The rendered image of the environment when the step was taken.
    """

    time_step: int
    obs: Tensor
    reward: float
    action: int
    render: ndarray

    def __str__(self) -> str:
        return f"t={self.time_step}"


@dataclass
class Episode:
    """An episode is a single run of an agent in an environment. A collection of time steps.

    Attributes:
        steps (list[Step]): Sequence of (time) Steps.
        parameters (dict[str, float]): The parameters of the environment when the episode was run.
            e.g. {"mass": 0.1, "length": 0.5}. The keys are the names of the parameters. This needs to be stored as
            the parameters vary between episodes. They are sampled from the distributions defined by the user.
        outcome (str): The outcome of the episode. "Success" or "Failure".
    """

    steps: list[Step] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    outcome: str = field(default="")

    def length(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        return f"Episode: {self.outcome} - {self.length()} steps"
