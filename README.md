# Deep Q-Network (DQN) Agent for Reinforcement Learning

This repository contains a Deep Q-Network (DQN) agent implemented in Python for reinforcement learning tasks using the [OpenAI Gym](https://gym.openai.com/) environment. The agent is designed to learn and make optimal decisions in environments with discrete action spaces.

## Prerequisites

Before running the agent, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- OpenAI Gym

You can install these dependencies using pip:

```bash
pip install torch numpy matplotlib gym
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/rarcifa/dqn-reinforcement-learning-service
cd your-repo-name
```

2. Run the main script:

```bash
py main.py
```

The main script main.py initializes the DQN agent, trains it in the specified Gym environment (e.g., CartPole), and visualizes the training progress using Matplotlib.

## Agent Configuration

You can configure the agent by modifying the parameters in the `main.py` script.

Key configurations include:

- The choice of Gym environment (`env.make("CartPole-v1"`) in the provided example).
- The number of episodes for training (`num_episodes`).
- Hyperparameters such as learning rate (`LR`), epsilon-greedy exploration parameters (`EPS_START`, `EPS_END`, `EPS_DECAY`), and more.
- Replay memory capacity (`ReplayMemory(10000)`).

Feel free to customize these parameters to suit your specific reinforcement learning task.

## Visualization

The agent provides visualization of training progress using Matplotlib. You can observe the training durations of episodes as they progress. Once training is complete, the final results are displayed.

## Credits

This implementation is based on the principles of Deep Q-Networks (DQN) for reinforcement learning and follows the structure commonly used in DQN-based agents.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the LICENSE file for details.
