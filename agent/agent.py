"""
Module implementing a reinforcement learning agent using a Deep Q-Network (DQN).

This module defines the Agent class which encapsulates the functionality required for a DQN-based
reinforcement learning agent. It includes methods for selecting actions, optimizing the model,
updating the target network, and plotting training results. The agent is designed to operate in
environments like CartPole-v1 from OpenAI's gym, using a DQN for learning optimal actions.

Classes:
    Agent: Encapsulates a reinforcement learning agent using a DQN.
"""
from environment.environment import *
from model.dqn import DQN
from utils.replay_memory import ReplayMemory
from utils.constants import *

import matplotlib
import matplotlib.pyplot as plt
import math
import random
import torch
from torch import optim
from typing import Any, Deque, List, NamedTuple, Optional
from IPython import display


class Agent:
    """
    A reinforcement learning agent using a Deep Q-Network (DQN).

    Attributes:
        policy_net (DQN): The policy network for the agent.
        target_net (DQN): The target network for the agent, used in calculating loss.
        optimizer (optim.AdamW): The optimizer for training the policy network.
        memory (ReplayMemory): Memory buffer for storing experiences.
        steps_done (int): Counter for the number of steps taken.
        device (torch.device): The device (CPU/GPU) on which the computation is performed.
    """

    def __init__(
        self, n_observations: int, n_actions: Any, device: torch.device
    ):
        """
        Initializes the Agent with policy and target networks, an optimizer, and a replay memory.

        Args:
            n_observations (int): The number of observation inputs.
            n_actions (int): The number of actions in the action space.
            device (torch.device): The device (CPU/GPU) for running computations.
        """
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=LR, amsgrad=True
        )
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.device = device

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        """
        Selects an action based on the current state using an epsilon-greedy strategy.

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: The selected action.
        """
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[env.action_space.sample()]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        """
        Optimizes the policy network based on a batch of experiences from the replay memory.
        """
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (
            next_state_values * GAMMA
        ) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """
        Updates the target network.
        """
        for key in self.policy_net.state_dict():
            self.target_net.state_dict()[key] = self.policy_net.state_dict()[
                key
            ] * TAU + self.target_net.state_dict()[key] * (1 - TAU)
        self.target_net.load_state_dict(self.target_net.state_dict())

    def plot_durations(self, show_result: bool = False):
        """
        Plots the durations of episodes.

        Args:
            episode_durations (List[int]): A list containing the duration of each episode.
            show_result (bool): If True, displays the result plot. Otherwise, shows training progress.
        """
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title("Training..." if not show_result else "Result")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

        global is_ipython
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
        else:
            plt.show()
