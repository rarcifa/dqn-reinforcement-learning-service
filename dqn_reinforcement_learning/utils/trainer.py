"""
Trainer Module.

This module provides a Trainer class for managing the training process of a reinforcement learning agent
using a Deep Q-Network (DQN).

The Trainer class is responsible for optimizing the DQN model, updating the target network, and handling
the sampling of experiences from the replay memory. This class plays a pivotal role in updating the agent's
knowledge and improving its performance over time through learning.

Classes:
    Trainer: Manages the training process of a DQN-based reinforcement learning agent.
"""

import torch.nn as nn

from dqn_reinforcement_learning.environment.config import *
from dqn_reinforcement_learning.utils.replay_memory import Transition


class Trainer:
    """
    Manages the training process of a reinforcement learning agent using a Deep Q-Network (DQN).

    The Trainer class handles the optimization of the policy network, updates the target network,
    and manages the replay memory. It uses experiences from the memory to update the agent's policy.

    Attributes:
        policy_net (DQN): The policy network being optimized.
        target_net (DQN): The target network used for calculating loss.
        optimizer (torch.optim.Optimizer): The optimizer for training the policy network.
        memory (ReplayMemory): The memory buffer storing experiences.
        tau (float): The rate at which the target network is updated.
        gamma (float): The discount factor for future rewards.
        batch_size (int): The size of batches sampled from the memory.
    """

    def __init__(
        self,
        policy_net: DQN,
        target_net: DQN,
        optimizer: torch.optim.Optimizer,
        memory: ReplayMemory,
        tau: float,
        gamma: float,
        batch_size: int,
    ):
        """
        Initializes the Trainer with networks, optimizer, memory, and training parameters.

        Args:
            policy_net (DQN): The policy network for the agent.
            target_net (DQN): The target network for the agent, used in calculating loss.
            optimizer (torch.optim.Optimizer): The optimizer for training the policy network.
            memory (ReplayMemory): Memory buffer for storing experiences.
            tau (float): Rate at which the target network is updated.
            gamma (float): Discount factor for future rewards.
            batch_size (int): Size of batches sampled from the memory.
        """
        self.policy_net: DQN = policy_net
        self.target_net: DQN = target_net
        self.optimizer: optim.Optimizer = optimizer
        self.memory: ReplayMemory = memory
        self.tau: float = tau
        self.gamma: float = gamma
        self.batch_size: int = batch_size

    def optimize_model(self) -> None:
        """
        Optimizes the policy network using experiences sampled from the replay memory.

        This method samples a batch of transitions from the replay memory and uses them to update
        the policy network. The update involves calculating the loss based on the current state and
        action values and the expected Q values for the next states. The method uses the Huber loss
        for stability.

        After calculating the loss, the method performs backpropagation and updates the weights of
        the policy network using the optimizer. This optimization step is crucial for the learning
        process of the agent.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from the memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute masks and concatenate batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # Prepare batch data
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q values for the current states and actions
        state_action_values = POLICY_NET(state_batch).gather(1, action_batch)

        # Compute expected Q values for next states
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                TARGET_NET(non_final_next_states).max(1).values
            )

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * GAMMA
        ) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(POLICY_NET.parameters(), 100)
        optimizer.step()

    def update_target_net(self) -> None:
        """
        Updates the weights of the target network.

        This method performs a soft update of the target network's weights using the weights from
        the policy network. The update is controlled by the tau parameter, which determines the
        rate at which the target network is updated.

        The soft update ensures gradual changes to the target network, contributing to the stability
        of the learning process.
        """

        # Retrieve the current state dictionaries of the target and policy networks
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # Loop through the parameters in the policy network
        for key in policy_net_state_dict:
            # Perform the soft update: a weighted sum of the target and policy network parameters
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)

        # Load the updated state dictionary back into the target network
        self.target_net.load_state_dict(target_net_state_dict)
