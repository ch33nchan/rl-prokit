import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation
from pettingzoo.utils import aec_to_parallel
from pettingzoo.classic import connect_four_v3  # Example PettingZoo env
import torch
import torch.nn as nn
import torch.optim as optim

class ICM(nn.Module):
    def __init__(self, state_size, action_size):
        super(ICM, self).__init__()
        self.feature = nn.Linear(state_size, 128)
        self.inverse = nn.Linear(256, action_size)
        self.forward = nn.Linear(128 + action_size, 128)

    def forward(self, state, action, next_state):
        state_feat = self.feature(state)
        next_state_feat = self.feature(next_state)
        pred_action = self.inverse(torch.cat((state_feat, next_state_feat), dim=1))
        pred_next_feat = self.forward(torch.cat((state_feat, action), dim=1))
        return pred_action, pred_next_feat

class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CuriosityWrapper, self).__init__(env)
        self.icm = ICM(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.icm.parameters(), lr=0.001)
        self.beta = 0.2

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        action_onehot = torch.zeros(self.env.action_space.n)
        action_onehot[action] = 1.0

        pred_action, pred_next_feat = self.icm(torch.tensor(obs, dtype=torch.float32), action_onehot, torch.tensor(obs, dtype=torch.float32))
        intrinsic_reward = self.beta * (pred_next_feat - self.icm.feature(torch.tensor(obs, dtype=torch.float32))).pow(2).mean()
        return obs, reward + intrinsic_reward.item(), done, info

class MultiAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MultiAgentWrapper, self).__init__(env)

class AtariWrapper(gym.Wrapper):
    def __init__(self, env):
        super(AtariWrapper, self).__init__(env)
        self.env = env
        self.env.reset()
        self.env.step(1)  # Fire-reset

    def step(self, action):
        obs1, r1, d1, i1 = self.env.step(action)
        obs2, r2, d2, i2 = self.env.step(action)
        obs = np.maximum(obs1, obs2)
        return obs, r1 + r2, d1 or d2, i1

def generate_wrapper(env_name, mods, env_type="standard", frame_stack=0, grayscale=False, curiosity=False):
    env = gym.make(env_name)
    if frame_stack > 0:
        env = FrameStack(env, frame_stack)
    if grayscale:
        env = GrayScaleObservation(env)
    if curiosity:
        env = CuriosityWrapper(env)
    if env_type == "multi-agent":
        env = MultiAgentWrapper(env)
    if env_type == "atari":
        env = AtariWrapper(env)
    # Save or return wrapper
    print("Wrapper generated for", env_name)
    return env

if __name__ == "__main__":
    generate_wrapper("CartPole-v1", "scale_rewards=0.5", frame_stack=4, grayscale=True, curiosity=True)
