import torch
import gymnasium as gym

def debug_policy(model, env_name):
    env = gym.make(env_name)
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = false

    while not done:
        action = model(state).argmax().item()
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        print(f"State: {state}, Action: {action}, Reward: {reward}")
        state = next_state

    env.close()

if __name__ == "__main__":
    model = torch.load('tuned_model.pth')
    debug_policy(model, 'CartPole-v1')
