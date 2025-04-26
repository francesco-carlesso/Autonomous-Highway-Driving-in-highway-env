import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Set seeds and device for reproducibility
base_seed = 0
np.random.seed(base_seed)
random.seed(base_seed)
torch.manual_seed(base_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the evaluation environment
env_name = "highway-v0"
env = gymnasium.make(
    env_name,
    config={'action': {'type': 'DiscreteMetaAction'},
            "lanes_count": 3,
            "ego_spacing": 1.5},
    render_mode='human'
)

# Define the Dueling DQN Network Architecture
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Advantage branch
        self.advantage = nn.Linear(128, action_dim)
        # Value branch
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        advantage = self.advantage(x)
        value = self.value(x)
        # Combine streams: Q(s,a)=V(s)+[A(s,a)-mean(A(s,·))]
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

# Get state and action dimensions from the environment
state, _ = env.reset(seed=base_seed)
state = state.reshape(-1)
state_dim = state.shape[0]
action_dim = env.action_space.n

# Initialize the model and load trained weights
model = DuelingDQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("dueling_dqn_model.pth", map_location=device))
model.eval()

def select_action(state):
    """Select the best action for a given state using the loaded dueling DQN model."""
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    return q_values.argmax().item()

# Evaluation Loop
state, _ = env.reset(seed=base_seed)
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

results = []

while episode <= 10:
    episode_steps += 1

    # Select action using the trained model
    action = select_action(state)

    # Step in the environment
    state, reward, done, truncated, _ = env.step(action)
    state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done or truncated:
        results.append([episode, episode_steps, episode_return, done])
        print(f"Episode Num: {episode} | Steps: {episode_steps} | Return: {episode_return:.3f} | Crash: {done}")

        # Reset the environment for the next episode
        episode_seed = base_seed + episode
        state, _ = env.reset(seed=episode_seed)
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

np.savetxt("dueling_dqn_results.csv", np.array(results), delimiter=",", header="Episode,Steps,Return,Crash", comments="")
