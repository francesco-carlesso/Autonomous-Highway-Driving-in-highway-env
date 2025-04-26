import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Set seeds and device
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = int(2e4)  # Total training steps
env_name = "highway-fast-v0"  # Fast environment for training
env = gymnasium.make(
    env_name,
    config={'action': {'type': 'DiscreteMetaAction'},
            'duration': 40,
            "vehicles_count": 50}
)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Dueling DQN Network
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

# Dueling DQN Agent
class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Initialize the policy and target networks using the dueling architecture
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99

        # epsilon–greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Update target network every fixed number of steps
        self.update_target_every = 1000
        self.step_count = 0

    def select_action(self, state):
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def push(self, state, action, reward, next_state, done):
        #Store a transition in replay memory
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        #Perform one training step using a batch from the replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Compute current Q values using policy network
        q_values = self.policy_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.max(dim=1)[0]
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """Save the policy network's parameters."""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """Load parameters from file."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Initialize Agent and Environment
state, _ = env.reset()
state = state.reshape(-1)
state_dim = state.shape[0]
action_dim = env.action_space.n
agent = DuelingDQNAgent(state_dim, action_dim, device)

done, truncated = False, False
episode = 1
episode_steps = 0
episode_return = 0

# List to store cumulative reward per episode for plotting later
episode_rewards = []

# Training Loop
for t in range(MAX_STEPS):
    episode_steps += 1
    action = agent.select_action(state)
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    # Store transition and train
    agent.push(state, action, reward, next_state, done or truncated)
    agent.train_step()

    state = next_state
    episode_return += reward

    if done or truncated:
        print(f"Total T: {t} | Episode: {episode} | Steps: {episode_steps} | Return: {episode_return:.3f}")
        episode_rewards.append([episode, episode_steps, episode_return])
        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

    if t % 500 == 0 and t > 0:
        agent.save("dueling_dqn_model.pth")

env.close()

np.savetxt("dueling_dqn_episode_rewards.csv", np.array(episode_rewards), delimiter=",", header="Episode,Steps,Return", comments="")
