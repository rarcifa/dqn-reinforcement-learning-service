from itertools import count

from dqn_reinforcement_learning.agents.defender import Defender
from dqn_reinforcement_learning.utils.plotter import Plotter
from dqn_reinforcement_learning.utils.trainer import Trainer
from dqn_reinforcement_learning.model.dqn import DQN
from dqn_reinforcement_learning.utils.replay_memory import ReplayMemory
from dqn_reinforcement_learning.utils.constants import *
from dqn_reinforcement_learning.environment.config import *


def main() -> None:
    """
    Main function for running the reinforcement learning training loop.

    Initializes the necessary components (agent, trainer, plotter) and executes
    the training loop for a predefined number of episodes. Each episode involves
    interacting with the environment, updating the agent's model, and tracking progress.
    """

    # Initialise actions, plots and model trainer
    action_selector = Defender(POLICY_NET, EPS_START, EPS_END, EPS_DECAY)
    plotter = Plotter()
    trainer = Trainer(
        POLICY_NET, TARGET_NET, optimizer, memory, TAU, GAMMA, BATCH_SIZE
    )

    # Main training loop
    num_episodes = 600

    # Training loop for each episode
    for i_episode in range(num_episodes):
        # Reset environment at the start of each episode and get initial state
        state, info = env.reset()
        state = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Iterate over each step in the episode
        for t in count():
            # Select an action based on the current state
            action = action_selector.select_action(state)

            # Perform the action in the environment and observe the result
            observation, reward, terminated, truncated, _ = env.step(
                action.item()
            )

            # Convert reward to tensor
            reward = torch.tensor([reward], device=device)

            # Check if the episode has ended
            done = terminated or truncated

            # Optimize the model based on the current state and action
            trainer.optimize_model()

            # Update the target network
            trainer.update_target_net()

            # Process next state
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = TARGET_NET.state_dict()
            policy_net_state_dict = POLICY_NET.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            TARGET_NET.load_state_dict(target_net_state_dict)

            # If the episode is done, log the duration and break from the loop
            if done:
                plotter.episode_durations.append(t + 1)
                plotter.plot_durations()
                break

    # Training complete
    print("Complete")

    # Display final results
    plotter.plot_durations(show_result=True)
