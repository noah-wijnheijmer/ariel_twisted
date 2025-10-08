"""Neural network brain module for modular robot control.

This module provides a flexible neural network architecture for controlling
modular robots through neuroevolution.
"""

# Import third-party libraries
import mujoco
import torch
import torch.nn as nn
import numpy as np

# Import local libraries

class RobotBrain(nn.Module):
    """
    A flexible feedforward neural network for robot control.
    
    Parameters
    ----------
    input_size : int
        Number of input neurons. (Number of actuators)
    output_size : int
        Number of output neurons. (qpos size)
    hidden_layers : list[int]
        List of hidden layer sizes.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list[int] | None = None,
    ):
        super(RobotBrain, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [64, 32]
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())  # Tanh to bound outputs to [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with small values
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights with small random values."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward_control(self,
                        data: mujoco.MjData,
                        model: mujoco.MjModel) -> np.ndarray:
        """Forward pass for control inputs."""
        state_tensor = torch.tensor(data.qpos, dtype=torch.float32).unsqueeze(0)

        return self.network(state_tensor).detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def get_weights_as_vector(self) -> np.ndarray:
        """
        Extract all network parameters as a flat 1D numpy array.
        
        Returns
        -------
        np.ndarray
            Flattened array of all weights and biases concatenated together.
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights_from_vector(self, weight_vector: np.ndarray) -> None:
        """
        Set network parameters from a flat 1D numpy array.
        
        Parameters
        ----------
        weight_vector : np.ndarray
            Flattened array of all weights and biases to set in the network.
            Must match the total number of parameters in the network.
        """
        weight_vector = np.asarray(weight_vector)
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_shape = param.shape
            param.data = torch.from_numpy(
                weight_vector[idx:idx + param_size].reshape(param_shape)
            ).float()
            idx += param_size