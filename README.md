```markdown
# RL ProtoKit: Lightweight CLI Tool for RL Prototyping

RL ProtoKit is a modular, open-source command-line tool designed for rapid reinforcement learning (RL) prototyping. It enables developers to generate custom environment wrappers, tune hyperparameters, debug policies, and run full RL pipelines with ease. Built to stand out in job applications and hackathons, it's extensible and supports advanced features like prioritized replay, RNN policies, multi-agent environments, and more.

## Key Features

- **Custom Wrapper Generation**: Create Gym-compatible wrappers with mods like reward scaling, frame stacking, grayscale transforms, and Atari-specific preprocessing (fire-reset, max-pool).
- **Hyperparameter Tuning**: Run grid search with support for prioritized replay buffers, RNN policy networks, PPO clip annealing, KL-divergence logging, and SAC temperature auto-tuning.
- **Policy Debugging**: Step through trained models, visualize Q-values, and log confidence for better insights.
- **Full Pipeline Execution**: Chain generation, tuning, and debugging into a single workflow.
- **Advanced Modules**: Includes intrinsic curiosity for exploration, multi-agent support via PettingZoo, and discrete/continuous action heads.
- **Transforms and Wrappers**: Frame stacking, grayscale, and Atari wrappers for image-based environments.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ch33nchan/rl-protokit.git
   cd rl-protokit
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Requirements include `torch`, `gymnasium`, `pettingzoo`, `numpy`.)

3. Install RL ProtoKit as an editable package:
   ```
   pip install -e .
   ```

## Usage

RL ProtoKit is used via the CLI with the `protokit` command. Below are the main subcommands and their options.

### Generate Command

Generate custom Gym wrappers with various mods and transforms.

```
protokit generate --env  --mods  [options]
```

- `--env`: Environment name (e.g., "CartPole-v1").
- `--mods`: Comma-separated mods (e.g., "scale_rewards=0.5").
- `--env-type`: "standard" (default), "multi-agent", or "atari".
- `--frame-stack`: Number of frames to stack (e.g., 4).
- `--grayscale`: Enable grayscale transform (boolean).
- `--curiosity`: Add intrinsic curiosity module (boolean).

**Example:**
```
protokit generate --env CartPole-v1 --mods "scale_rewards=0.5" --frame-stack 4 --grayscale --curiosity
```

This outputs a `custom_wrapper.py` file with the specified wrapper.

### Tune Command

Tune hyperparameters using grid search with advanced options.

```
protokit tune --env  --params  [options]
```

- `--env`: Environment name.
- `--params`: Params like "lr: [0.001, 0.01]".
- `--trials`: Number of trials (default: 5).
- `--replay-type`: "standard" (default) or "prioritized".
- `--policy-type`: "fc" (default) or "rnn".
- `--action-head`: "discrete" (default) or "continuous".
- `--ppo-clip-anneal`: Enable PPO clip annealing (boolean).
- `--log-kl`: Log KL-divergence for PPO (boolean).
- `--sac-temp-auto`: Auto-tune SAC temperature (boolean).

**Example:**
```
protokit tune --env CartPole-v1 --params "lr: [0.001, 0.01]" --trials 10 --replay-type prioritized --policy-type rnn --ppo-clip-anneal
```

Outputs a tuned model file (e.g., `tuned_model.pth`).

### Debug Command

Debug a trained policy with step-by-step visualization.

```
protokit debug --model  --env 
```

- `--model`: Path to trained model (e.g., "tuned_model.pth").
- `--env`: Environment name.

**Example:**
```
protokit debug --model tuned_model.pth --env CartPole-v1
```

Displays Q-values, confidence, and policy decisions.

### Full Pipeline Command

Run the entire workflow: generate, tune, debug.

```
protokit full --env  --mods  --params  --trials 
```

- Same options as generate and tune.

**Example:**
```
protokit full --env CartPole-v1 --mods "scale_rewards=0.5" --params "lr: [0.001, 0.01]" --trials 10
```

Generates wrapper, tunes model, and debugs in one go.

## Components and Functionalities

### Wrapper Generator

- **Core Functionality:** Creates custom Gym wrappers with mods like reward scaling.
- **Advanced Features:**
  - **Prioritized Replay Buffer:** Samples experiences based on TD error for efficient learning.
  - **RNN Policy Networks:** Supports LSTM-based policies for temporal dependencies.
  - **Intrinsic Curiosity Module:** Adds exploration bonuses via forward/inverse models.
  - **Multi-Agent Wrapper:** Integrates PettingZoo for multi-agent environments.
  - **Action Heads:** Discrete (softmax) or continuous (Gaussian) outputs.
  - **Transforms:** Frame stacking, grayscale conversion.
  - **Atari Wrapper:** Fire-reset, max-pooling for Atari games.

### Hyperparameter Tuner

- **Core Functionality:** Grid search with parallel trials.
- **Advanced Features:**
  - **PPO-Clip Schedule Annealing:** Gradually reduces clip parameter for stable updates.
  - **KL-Divergence Logging:** Monitors policy changes during PPO training.
  - **SAC Temperature Auto-Tuning:** Dynamically adjusts entropy temperature.

### Policy Debugger

- **Core Functionality:** Steps through policies, logs states/actions/rewards.
- **Advanced Features:** Visualizes Q-values and confidence for RNN and multi-agent policies.

### Full Pipeline

- **Core Functionality:** Chains generation, tuning, and debugging.
- **Advanced Features:** Supports all wrapper and tuner options in a single run.

## Examples

### Generating a Wrapper with Curiosity

```
protokit generate --env CartPole-v1 --mods "scale_rewards=0.5" --curiosity
```

### Tuning with RNN and Prioritized Replay

```
protokit tune --env CartPole-v1 --params "lr: [0.001, 0.01]" --policy-type rnn --replay-type prioritized
```

### Full Pipeline with Atari Wrapper

```
protokit full --env Breakout-v0 --mods "scale_rewards=0.5" --params "lr: [0.001]" --trials 5 --env-type atari
```

## Contributing

Fork the repo, make changes, and submit a pull request. We welcome contributions to new wrappers or algorithm features.

## License

MIT License. See LICENSE file for details.
```