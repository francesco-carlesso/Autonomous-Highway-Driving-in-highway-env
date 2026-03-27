import gymnasium
import highway_env
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# Set the seeds and device
base_seed = 0
np.random.seed(base_seed)
random.seed(base_seed)
torch.manual_seed(base_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = "highway-v0"  # Evaluation environment
env = gymnasium.make(
    env_name,
    config={'action': {'type': 'DiscreteMetaAction'},
            "lanes_count": 3,
            "ego_spacing": 1.5,
            'high_speed_reward': 1},
    render_mode='human'
)

# Define the DQN Network Architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Model and Load Trained Weightsn
state, _ = env.reset(seed=base_seed)
state = state.reshape(-1)
state_dim = state.shape[0]
action_dim = env.action_space.n

# Create the model instance and load the trained weights
model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("double_dqn_model.pth", map_location=device))
model.eval()

def select_action(state):
    """Selects the best action for a given state using the trained model."""
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
    action = select_action(state)

    state, reward, done, truncated, _ = env.step(action)
    state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done or truncated:
        results.append([episode, episode_steps, episode_return, done])
        print(f"Episode Num: {episode} | Steps: {episode_steps} | Return: {episode_return:.3f} | Crash: {done}")

        episode_seed = base_seed + episode
        state, _ = env.reset(seed=episode_seed)
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

np.savetxt("double_dqn_results.csv", results, delimiter=",", header="Episode,Steps,Return,Crash", comments='')
