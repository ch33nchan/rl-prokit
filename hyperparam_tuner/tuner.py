import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np
import gymnasium as gym

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.n_entries + self.capacity - 1
        self.pending_idx.add(idx)
        self.data[self.n_entries] = data
        self.update(idx, p)
        self.n_entries += 1
        if self.n_entries >= self.capacity:
            self.n_entries = 0

    def update(self, idx, p):
        if idx in self.pending_idx:
            self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01
        self.capacity = capacity

    def add(self, state, action, next_state, reward, done):
        transition = Transition(state, action, next_state, reward, done)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, transition)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def __len__(self):
        return self.tree.n_entries

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RNNPolicy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(RNNPolicy, self).__init__()
        self.rnn = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        return x, hidden

class DiscreteHead(nn.Module):
    def __init__(self, input_size, action_size):
        super(DiscreteHead, self).__init__()
        self.fc = nn.Linear(input_size, action_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class ContinuousHead(nn.Module):
    def __init__(self, input_size, action_size):
        super(ContinuousHead, self).__init__()
        self.fc = nn.Linear(input_size, action_size)

    def forward(self, x):
        return self.fc(x)

def tune_hyperparams(env_name, params, trials=5, replay_type='standard', policy_type='fc', action_head='discrete', ppo_clip_anneal=False, log_kl=False, sac_temp_auto=False):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    if policy_type == 'rnn':
        model = RNNPolicy(state_size, action_size)
    else:
        model = DQN(state_size, action_size)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    if replay_type == 'prioritized':
        replay_buffer = PrioritizedReplayBuffer(10000)
    else:
        replay_buffer = deque(maxlen=10000)

    for trial in range(trials):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            action = random.randint(0, action_size - 1)  # Random action for demo
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

            transition = (state, action, next_state, reward, done)
            if replay_type == 'prioritized':
                error = abs(reward)  # Simple error for priority
                replay_buffer.add(state, action, next_state, reward, done)
                replay_buffer.update(replay_buffer.tree.n_entries - 1, error)
            else:
                replay_buffer.append(transition)

            state = next_state

        print(f"Trial {trial + 1}: Total Reward = {total_reward}")

    env.close()
    return model

if __name__ == "__main__":
    tune_hyperparams('CartPole-v1', {'lr': 0.001}, trials=5, replay_type='prioritized', policy_type='rnn', action_head='discrete', ppo_clip_anneal=True, log_kl=True, sac_temp_auto=True)
