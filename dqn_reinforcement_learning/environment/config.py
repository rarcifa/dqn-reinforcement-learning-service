import gymnasium as gym
import torch
from torch import optim

from dqn_reinforcement_learning.model.dqn import DQN
from dqn_reinforcement_learning.utils.replay_memory import ReplayMemory
from dqn_reinforcement_learning.utils.constants import *

# Set up environment
env = gym.make("LunarLander-v2", render_mode="human")
n_actions = getattr(env.action_space, "n", None)
state, info = env.reset()
n_observations = len(state)
steps_done = 0

# GPU usage configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize network and optimizer
POLICY_NET = DQN(n_observations, n_actions).to(device)
TARGET_NET = DQN(n_observations, n_actions).to(device)
TARGET_NET.load_state_dict(POLICY_NET.state_dict())
optimizer = optim.AdamW(POLICY_NET.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
