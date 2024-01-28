# Deep Q-Network (DQN) Agent for Reinforcement Learning

This repository contains a Deep Q-Network (DQN) agent implemented in Python for reinforcement learning tasks using the [OpenAI Gym](https://github.com/openai/gym) environment. The agent is designed to learn and make optimal decisions in environments with discrete action spaces.

## Prerequisites

Before running the agent, make sure you have the following dependencies installed:

- Python 3.x
- Poetry
- PyTorch
- NumPy
- Matplotlib
- OpenAI Gym

You can install Poetry and other dependencies using the following commands:

```bash
pip install poetry
poetry install
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/rarcifa/dqn-reinforcement-learning-service
   cd dqn-reinforcement-learning-service
   ```

2. Run the main script:

   - Using Python directly:

     ```bash
     py main.py
     ```

   - Using Poetry to run the module:

     ```bash
     poetry run python -m dqn_reinforcement_learning
     ```

3. Using Docker:

   - Build the Docker image:

     ```bash
     docker build -t dqn-reinforcement-learning -f .docker/Dockerfile .
     ```

   - Run the Docker container:

     ```bash
     docker run -p 4000:80 dqn-reinforcement-learning
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
