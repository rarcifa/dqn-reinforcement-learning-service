from model.dqn import DQN
from utils.replay_memory import ReplayMemory
from utils.constants import *

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import optim
from typing import List

# Set up environment
env = gym.make("CartPole-v1")
n_actions = getattr(env.action_space, "n", None)
state, info = env.reset()
n_observations = len(state)

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
plt.ion()

# GPU usage configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize network and optimizer
POLICY_NET = DQN(n_observations, n_actions).to(device)
TARGET_NET = DQN(n_observations, n_actions).to(device)
TARGET_NET.load_state_dict(POLICY_NET.state_dict())
optimizer = optim.AdamW(POLICY_NET.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

STEPS_DONE = 0
episode_durations: List[int] = []
