# RL ProtoKit

A lightweight, unified toolkit for RL prototyping. Generate custom Gym wrappers, tune hyperparameters, and debug policies—all in one CLI.

## Installation
pip install -e .

text

## Usage
- Generate wrapper: `protokit generate --env CartPole-v1 --mods "scale_rewards=0.5"`
- Tune params: `protokit tune --params "lr: [0.001, 0.01]" --trials 5`
- Debug policy: `protokit debug --model tuned_model.pth --steps 10`
- Full pipeline: `protokit full --env CartPole-v1 --mods "scale_rewards=0.5" --params "lr: [0.001, 0.01]" --trials 5 --debug_steps 10`

See docs for more details.