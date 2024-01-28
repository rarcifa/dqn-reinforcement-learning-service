"""
Defender Module.

Module implementing a reinforcement learning agent using a Deep Q-Network (DQN).

This module defines the Agent class which encapsulates the functionality required for a DQN-based
reinforcement learning agent. It includes methods for selecting actions and interacting with the
environment using an epsilon-greedy strategy. The agent is designed to operate in environments like
CartPole-v1 from OpenAI's gym, using a DQN for learning optimal actions.

Classes:
    Agent: Encapsulates a reinforcement learning agent using a DQN.
"""

import math
import random
import torch

from dqn_reinforcement_learning.environment.config import *
from dqn_reinforcement_learning.utils.constants import *


class Defender:
    """
    A reinforcement learning agent using a Deep Q-Network (DQN).

    Attributes:
        policy_net (DQN): The policy network for the agent.
        eps_start (float): Starting value of epsilon in epsilon-greedy policy.
        eps_end (float): Minimum value of epsilon in epsilon-greedy policy.
        eps_decay (float): Rate of exponential decay of epsilon.
        steps_done (int): Counter for the number of steps taken.
    """

    def __init__(
        self,
        policy_net: DQN,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
    ):
        """
        Initializes the Agent with a policy network and parameters for the epsilon-greedy policy.

        Args:
            policy_net (DQN): The policy network for the agent.
            eps_start (float): Starting value of epsilon in epsilon-greedy policy.
            eps_end (float): Minimum value of epsilon in epsilon-greedy policy.
            eps_decay (float): Rate of exponential decay of epsilon.
        """
        self.policy_net = policy_net
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Selects an action based on the current state using an epsilon-greedy strategy.

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: The selected action.
        """

        # Generate a random sample for epsilon-greedy decision
        sample = random.random()

        # Calculate the threshold for epsilon-greedy strategy
        eps_threshold = self.eps_end + (
            self.eps_start - self.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.eps_decay)

        # Increment the step counter
        self.steps_done += 1

        # Determines the action to take based on the epsilon-greedy strategy:
        # - If the random sample is greater than the epsilon threshold (more exploitation):
        #   (1) Turn off gradients for inference to improve performance and reduce memory usage.
        #   (2) Use the policy network to select the action with the highest estimated Q-value.
        # - Otherwise (more exploration):
        #   Select a random action from the action space.
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[env.action_space.sample()]], device=device, dtype=torch.long
            )
