import torch
import gymnasium as gym
from rich.console import Console
from rich.table import Table
from hyperparam_tuner.tuner import SimpleDQN  # Reuse model class

def debug_policy(model_path, steps):
    env = gym.make('CartPole-v1')
    model = SimpleDQN(4, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    console = Console()
    table = Table(title="Policy Debug Steps")
    table.add_column("Step", style="cyan")
    table.add_column("State", style="magenta")
    table.add_column("Action", style="green")
    table.add_column("Q-Values", style="yellow")
    table.add_column("Confidence", style="red")
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    for step in range(steps):
        q_values = model(obs)
        action = torch.argmax(q_values).item()
        confidence = torch.softmax(q_values, dim=0)[action].item()
        obs, _, done, _, _ = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32)
        table.add_row(
            str(step),
            str(obs.numpy().round(2)),
            str(action),
            str(q_values.detach().numpy().round(2)),
            f"{confidence:.2f}"
        )
        if done:
            console.print("[bold red]Episode ended early![/bold red]")
            break
    
    console.print(table)