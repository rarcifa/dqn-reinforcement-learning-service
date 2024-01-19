"""
Deep Q-Network (DQN) Module.

This module defines the Deep Q-Network (DQN) class, a neural network architecture
commonly used in reinforcement learning. The DQN class is a PyTorch module
comprising three linear layers, making it suitable for learning value functions
in environments with discrete action spaces.

The network's architecture is designed to approximate the Q-value function in a
given environment. It takes a state representation as input and outputs the
estimated Q-values for each possible action in that state.

Classes:
    DQN: A deep neural network for Q-learning based reinforcement learning tasks.
"""

from typing import Any
import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network using PyTorch.

    This class represents a deep neural network using a standard architecture
    for Q-learning based reinforcement learning tasks. It consists of three
    linear layers with ReLU activations, suitable for approximating the
    Q-value function in a given environment.

    Attributes:
        layer1 (nn.Linear): The first linear layer of the network.
        layer2 (nn.Linear): The second linear layer of the network.
        layer3 (nn.Linear): The third linear layer of the network that outputs Q-values for each action.
    """

    def __init__(self, n_observations: int, n_actions: Any):
        """
        Initializes the DQN model with three linear layers.

        Args:
            n_observations (int): The number of features/observations in the input state representation.
            n_actions (int): The number of possible actions, defining the size of the output layer.
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor representing the state of the environment.

        Returns:
            torch.Tensor: The tensor containing Q-values for each action.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
