# Import third-party libraries
import torch
import torch.nn as nn
import mujoco
from typing import TYPE_CHECKING, Any

# Import local libraries

# def policy(
#     model: mujoco.MjModel,  # noqa: ARG001
#     data: mujoco.MjData,
#     cpg: CPGSensoryFeedback,
# ) -> None:
#     """Use feedback term to shift the output of the CPGs."""
#     x, _ = cpg.step()
    # return x * np.pi / 2

class SimpleNN(nn.Module):
        def __init__(self, input_size: int, output_size: int):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, output_size)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return x


def nn_controller(
    model: mujoco.MjModel,  # noqa: ARG001
    data: mujoco.MjData,
) -> Any:
    """A simple feedforward neural network controller."""
    # Define a simple feedforward neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_size: int, output_size: int):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, output_size)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return x

