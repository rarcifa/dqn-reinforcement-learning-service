"""
Replay Memory Module.

Module containing utility structures and classes used in reinforcement learning.

It includes a namedtuple for representing transitions in the environment and a class 
for storing these transitions in a replay memory for later retrieval during training.

Classes:
    ReplayMemory: A simple storage class for transitions observed during training.
"""

from collections import namedtuple, deque
from typing import Any, Deque, List
import random

from dqn_reinforcement_learning.environment.config import *

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


class ReplayMemory:
    """
    Simple storage for transitions observed during training.

    This class provides a memory buffer that stores transitions up to a fixed maximum size.
    Transitions are stored in a deque, which automatically removes the oldest transitions
    once the capacity is exceeded.

    Attributes:
        memory (Deque[Transition]): A deque object to store transitions with a fixed maximum size.
    """

    def __init__(self, capacity: int):
        """
        Initializes the ReplayMemory with a given capacity.

        Args:
            capacity (int): The maximum number of items the memory can hold.
        """
        self.memory: Deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        """
        Saves a transition to the memory.

        Args:
            *args: The transition data to be stored.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Any]:
        """
        Randomly samples a batch of transitions from the memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            List[Any]: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the current size of the memory.

        Returns:
            int: The number of items in the memory.
        """
        return len(self.memory)
