from agent.agent import Agent
from model.dqn import DQN
from utils.replay_memory import ReplayMemory
from agent.agent import Agent
from utils.constants import *
from environment.environment import *

from itertools import count

# Initialize agent
agent = Agent(n_observations, n_actions, device)

# Main training loop
num_episodes = 600 if torch.cuda.is_available() else 50
for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        next_state = (
            None
            if terminated
            else torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
        )
        memory.push(state, action, next_state, reward)
        state = next_state

        agent.optimize_model()

        for key in POLICY_NET.state_dict():
            TARGET_NET.state_dict()[key] = POLICY_NET.state_dict()[
                key
            ] * TAU + TARGET_NET.state_dict()[key] * (1 - TAU)
        TARGET_NET.load_state_dict(TARGET_NET.state_dict())

        if done:
            episode_durations.append(t + 1)
            agent.plot_durations()
            break


print("Complete")
agent.plot_durations(show_result=True)
plt.ioff()
plt.show()
