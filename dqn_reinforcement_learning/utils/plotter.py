"""
Plotter Module.

Module providing a Plotter class for visualizing the training progress of a reinforcement learning agent.

The Plotter class is responsible for plotting the duration of each episode over the course of the training,
allowing users to visually assess the agent's learning and improvement. It supports both the ongoing training
visualization and the final result presentation.

Classes:
    Plotter: A class for plotting and visualizing the training progress of a reinforcement learning agent.
"""

import matplotlib
import matplotlib.pyplot as plt
import torch

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


class Plotter:
    """
    Class for plotting the training progress of a reinforcement learning agent.

    This class provides functionality to plot the duration of each episode over the course of training,
    allowing for visualization of the agent's learning progress.

    Attributes:
        episode_durations (List[int]): A list to store the duration of each episode during training.
    """

    def __init__(self):
        """
        Initializes the Plotter with an empty list to store episode durations.
        """
        self.episode_durations = []

    def plot_durations(self, show_result: bool = False) -> None:
        """
        Plots the durations of episodes.

        Args:
            show_result (bool): If True, displays the final plot. If False, updates the ongoing training plot.
        """

        # Set up a new figure for plotting
        plt.figure(1)

        # Convert episode durations to a PyTorch tensor for easy manipulation
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)

        # Set the plot title based on whether it's showing the result or training progress
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")

        # Label the axes
        plt.xlabel("Episode")
        plt.ylabel("Duration")

        # Plot the durations of episodes
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            # Unfold the tensor to calculate the rolling average over the last 100 episodes
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)

            # Prepend zeros to the means to align them correctly with the x-axis
            means = torch.cat((torch.zeros(99), means))

            # Plot the rolling average
            plt.plot(means.numpy())

        # Pause to update the plots
        plt.pause(0.001)

        # If in an IPython environment, handle display updates
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


plt.ioff()
plt.show()
