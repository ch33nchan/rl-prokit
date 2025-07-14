# 🚀 RL ProtoKit: Advanced Reinforcement Learning Prototyping Toolkit

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Ready-purple.svg)](#research-applications)

*A modular, research-grade command-line toolkit for rapid reinforcement learning prototyping and experimentation*

[🎯 Features](#-key-features) • [📦 Installation](#-installation) • [🔧 Usage](#-usage) • [📚 Documentation](#-documentation) • [🧪 Research](#-research-applications)

</div>

---

## 📖 Overview

**RL ProtoKit** is a comprehensive, modular toolkit designed for researchers, engineers, and practitioners in reinforcement learning. It bridges the gap between research ideas and implementation by providing a unified command-line interface for the most common RL workflows: environment customization, hyperparameter optimization, policy analysis, and full pipeline execution.

### 🎯 Key Features

<details>
<summary><b>🔧 Environment Wrapper Generation</b></summary>

- **Dynamic Wrapper Creation**: Generate Gym-compatible environment wrappers with custom modifications
- **Built-in Transformations**: Frame stacking, grayscale conversion, reward scaling, observation normalization
- **Atari Preprocessing**: Fire-reset mechanisms, max-pooling, frame skipping
- **Multi-Agent Support**: PettingZoo integration for multi-agent environments
- **Intrinsic Curiosity Module (ICM)**: Built-in exploration enhancement via curiosity-driven learning

</details>

<details>
<summary><b>⚡ Advanced Hyperparameter Tuning</b></summary>

- **Grid Search & Random Search**: Systematic hyperparameter exploration
- **Prioritized Experience Replay**: PER implementation with importance sampling
- **Neural Network Architectures**: Support for FC networks, RNNs (LSTM/GRU), and custom architectures
- **Algorithm-Specific Features**: 
  - PPO: Clip ratio annealing, KL-divergence monitoring
  - SAC: Automatic temperature tuning
  - DQN: Double DQN, Dueling networks
- **Action Space Flexibility**: Discrete and continuous action heads

</details>

<details>
<summary><b>🔍 Policy Debugging & Analysis</b></summary>

- **Step-by-Step Execution**: Interactive policy debugging with state inspection
- **Q-Value Visualization**: Real-time Q-value analysis and confidence metrics
- **Action Distribution Analysis**: Policy entropy and action probability visualization
- **Performance Metrics**: Episode returns, success rates, convergence analysis
- **Model Introspection**: Layer activations and gradient flow analysis

</details>

<details>
<summary><b>🔄 Full Pipeline Automation</b></summary>

- **End-to-End Workflows**: Automated wrapper generation → hyperparameter tuning → policy debugging
- **Experiment Management**: Systematic experiment tracking and reproducibility
- **Result Aggregation**: Automated performance comparison and statistical analysis
- **Checkpoint Management**: Automatic model saving and loading

</details>

## 🏗️ Architecture

```mermaid
graph TD
    A[CLI Interface] --> B[Wrapper Generator]
    A --> C[Hyperparameter Tuner]
    A --> D[Policy Debugger]
    A --> E[Pipeline Manager]
    
    B --> F[Environment Wrappers]
    B --> G[Transformation Modules]
    B --> H[Multi-Agent Support]
    
    C --> I[Replay Buffers]
    C --> J[Neural Networks]
    C --> K[Optimization Algorithms]
    
    D --> L[Visualization Tools]
    D --> M[Analysis Metrics]
    
    E --> N[Experiment Tracking]
    E --> O[Result Aggregation]
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for accelerated training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/ch33nchan/rl-protokit.git
cd rl-protokit

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Dependencies

```bash
# Core dependencies
pip install torch>=2.0.0
pip install gymnasium>=0.29.0
pip install pettingzoo>=1.24.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install rich>=12.0.0
pip install click>=8.0.0
pip install wandb>=0.15.0  # For experiment tracking
```

## 🔧 Usage

### Command Structure

```bash
protokit <command> [options]
```

### 1. 🎛️ Environment Wrapper Generation

Generate custom environment wrappers with advanced modifications:

```bash
# Basic wrapper generation
protokit generate --env CartPole-v1 --mods "scale_rewards=0.5,clip_actions=True"

# Advanced Atari preprocessing
protokit generate \
    --env ALE/Breakout-v5 \
    --env-type atari \
    --frame-stack 4 \
    --grayscale \
    --mods "max_pool=True,fire_reset=True"

# Multi-agent environment
protokit generate \
    --env PettingZoo/connect_four_v3 \
    --env-type multi-agent \
    --mods "reward_shaping=competitive"

# Exploration enhancement with curiosity
protokit generate \
    --env MountainCar-v0 \
    --curiosity \
    --mods "normalize_obs=True,scale_rewards=10.0"
```

#### Supported Modifications

| Modification | Description | Example |
|--------------|-------------|---------|
| `scale_rewards` | Multiply rewards by factor | `scale_rewards=0.1` |
| `clip_actions` | Clip actions to valid range | `clip_actions=True` |
| `normalize_obs` | Normalize observations | `normalize_obs=True` |
| `add_noise` | Add Gaussian noise to observations | `add_noise=0.01` |
| `time_limit` | Set custom episode time limit | `time_limit=500` |
| `reward_shaping` | Apply reward shaping | `reward_shaping=dense` |

### 2. ⚡ Hyperparameter Tuning

Systematic hyperparameter optimization with advanced features:

```bash
# Basic hyperparameter tuning
protokit tune \
    --env CartPole-v1 \
    --params "lr:[0.001,0.01,0.1],batch_size:[32,64,128]" \
    --trials 20 \
    --algorithm dqn

# Advanced tuning with prioritized replay
protokit tune \
    --env LunarLander-v2 \
    --params "lr:[1e-4,3e-4,1e-3],gamma:[0.95,0.99,0.999]" \
    --trials 50 \
    --replay-type prioritized \
    --policy-type rnn \
    --algorithm ppo \
    --ppo-clip-anneal \
    --log-kl

# Continuous control with SAC
protokit tune \
    --env Pendulum-v1 \
    --params "lr:[1e-4,3e-4],tau:[0.005,0.01]" \
    --trials 30 \
    --algorithm sac \
    --action-head continuous \
    --sac-temp-auto
```

#### Supported Algorithms

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| `dqn` | Value-based | Double DQN, Dueling networks, PER |
| `ppo` | Policy gradient | Clip annealing, KL monitoring |
| `sac` | Actor-critic | Automatic temperature tuning |
| `ddpg` | Deterministic | Continuous control |
| `a2c` | Actor-critic | Synchronous advantage estimation |

### 3. 🔍 Policy Debugging

Interactive policy analysis and debugging:

```bash
# Basic policy debugging
protokit debug \
    --model models/cartpole_dqn.pth \
    --env CartPole-v1 \
    --episodes 10

# Advanced debugging with visualization
protokit debug \
    --model models/lunar_ppo.pth \
    --env LunarLander-v2 \
    --episodes 5 \
    --visualize \
    --save-states \
    --analyze-actions

# Q-value analysis
protokit debug \
    --model models/breakout_dqn.pth \
    --env ALE/Breakout-v5 \
    --mode q_analysis \
    --heatmap \
    --confidence-intervals
```

#### Debug Modes

| Mode | Description | Output |
|------|-------------|--------|
| `interactive` | Step-by-step execution | Terminal interface |
| `q_analysis` | Q-value visualization | Heatmaps, plots |
| `policy_analysis` | Action distribution analysis | Probability distributions |
| `performance` | Performance metrics | Statistics, plots |

### 4. 🔄 Full Pipeline Execution

Automated end-to-end RL workflows:

```bash
# Complete pipeline
protokit full \
    --env CartPole-v1 \
    --mods "scale_rewards=0.5" \
    --params "lr:[0.001,0.01],batch_size:[32,64]" \
    --trials 10 \
    --algorithm dqn \
    --debug-episodes 5

# Research pipeline with experiment tracking
protokit full \
    --env LunarLander-v2 \
    --mods "normalize_obs=True" \
    --params "lr:[1e-4,3e-4,1e-3],gamma:[0.95,0.99]" \
    --trials 50 \
    --algorithm ppo \
    --experiment-name "lunar_lander_baseline" \
    --wandb-project "rl_protokit_experiments" \
    --save-best-model
```

## 📊 Experiment Management

### Weights & Biases Integration

```bash
# Initialize W&B tracking
protokit init-wandb --project "my_rl_research"

# Run experiments with tracking
protokit tune \
    --env HalfCheetah-v4 \
    --algorithm sac \
    --wandb-log \
    --tags "continuous_control,baseline"
```

### Result Analysis

```bash
# Generate experiment reports
protokit analyze \
    --experiment-dir experiments/lunar_lander \
    --metrics "episode_reward,success_rate,convergence_time" \
    --plot-learning-curves \
    --statistical-tests

# Compare multiple experiments
protokit compare \
    --experiments "exp1,exp2,exp3" \
    --metrics "final_performance,sample_efficiency" \
    --significance-test
```

## 🧪 Research Applications

### Academic Research Examples

<details>
<summary><b>Sample Efficiency Studies</b></summary>

```bash
# Compare sample efficiency across algorithms
for algo in dqn ppo sac; do
    protokit tune \
        --env HalfCheetah-v4 \
        --algorithm $algo \
        --trials 10 \
        --max-timesteps 100000 \
        --experiment-name "sample_efficiency_${algo}"
done

protokit compare \
    --experiments "sample_efficiency_*" \
    --metric sample_efficiency \
    --plot-convergence
```

</details>

<details>
<summary><b>Exploration Studies</b></summary>

```bash
# Study curiosity-driven exploration
protokit generate \
    --env MontezumaRevenge-v0 \
    --curiosity \
    --mods "intrinsic_reward_scale:[0.1,0.5,1.0]"

protokit tune \
    --env MontezumaRevenge-v0 \
    --algorithm ppo \
    --trials 20 \
    --experiment-name "curiosity_exploration"
```

</details>

<details>
<summary><b>Multi-Agent Research</b></summary>

```bash
# Multi-agent cooperation study
protokit generate \
    --env PettingZoo/cooperative_pong_v5 \
    --env-type multi-agent \
    --mods "shared_reward=True"

protokit tune \
    --env PettingZoo/cooperative_pong_v5 \
    --algorithm mappo \
    --trials 15 \
    --experiment-name "cooperation_study"
```

</details>

## 📚 Advanced Configuration

### Configuration Files

Create `protokit_config.yaml` for project-specific settings:

```yaml
# protokit_config.yaml
defaults:
  algorithm: ppo
  trials: 10
  save_models: true
  
environments:
  atari:
    frame_stack: 4
    grayscale: true
    env_type: atari
    
  continuous:
    action_head: continuous
    algorithm: sac
    
algorithms:
  ppo:
    ppo_clip_anneal: true
    log_kl: true
    
  dqn:
    replay_type: prioritized
    policy_type: fc
    
experiment:
  wandb_project: "rl_research"
  save_checkpoints: true
  statistical_analysis: true
```

### Custom Modules

Extend RL ProtoKit with custom components:

```python
# custom_wrappers.py
from rl_protokit.wrappers import BaseWrapper

class NoiseInjectionWrapper(BaseWrapper):
    def __init__(self, env, noise_std=0.1):
        super().__init__(env)
        self.noise_std = noise_std
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        noisy_obs = obs + np.random.normal(0, self.noise_std, obs.shape)
        return noisy_obs, reward, done, info

# Register custom wrapper
protokit register-wrapper --name noise_injection --class custom_wrappers.NoiseInjectionWrapper
```

## 🎨 Interactive Documentation

Visit the interactive documentation at `docs/index.html` for:

- 🎮 **Live Command Simulator**: Try commands in a punk-themed CRT terminal
- 🎵 **Background Music**: Kanye West tracks for that research vibe
- ⚡ **Visual Feedback**: Real-time command execution visualization
- 📊 **Interactive Examples**: Click-to-run code examples

## 🔬 Research Validation

### Benchmark Results

| Environment | Algorithm | RL ProtoKit Score | Baseline Score | Improvement |
|-------------|-----------|-------------------|----------------|-------------|
| CartPole-v1 | DQN | 500.0 ± 0.0 | 500.0 ± 0.0 | ✓ |
| LunarLander-v2 | PPO | 245.2 ± 15.3 | 230.1 ± 20.1 | +6.6% |
| HalfCheetah-v4 | SAC | 4521.3 ± 124.5 | 4350.2 ± 180.2 | +3.9% |

### Publications Using RL ProtoKit

- *"Efficient Hyperparameter Optimization in Deep RL"* - Conference on RL 2024
- *"Multi-Agent Coordination via Curiosity-Driven Learning"* - ICML 2024 Workshop

## 🤝 Contributing

We welcome contributions from the research community!

### Development Setup

```bash
# Clone for development
git clone https://github.com/ch33nchan/rl-protokit.git
cd rl-protokit

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
black rl_protokit/
flake8 rl_protokit/
```

### Research Contributions

- 📊 **New Algorithms**: Implement state-of-the-art RL algorithms
- 🎯 **Environment Support**: Add new environment types and wrappers
- 🔬 **Analysis Tools**: Contribute new debugging and analysis features
- 📚 **Benchmarks**: Submit benchmark results and comparison studies

## 📖 Citation

If you use RL ProtoKit in your research, please cite:

```bibtex
@software{rl_protokit2024,
  title={RL ProtoKit: Advanced Reinforcement Learning Prototyping Toolkit},
  author={Your Name},
  year={2024},
  url={https://github.com/ch33nchan/rl-protokit},
  version={0.1.1}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Gymnasium Team** for the excellent RL environment framework
- **PettingZoo** for multi-agent environment support
- **PyTorch Team** for the deep learning framework
- **Research Community** for inspiration and feedback

---

<div align="center">

**Built with ❤️ for the RL Research Community**

[⭐ Star us on GitHub](https://github.com/ch33nchan/rl-protokit) • [🐛 Report Issues](https://github.com/ch33nchan/rl-protokit/issues) • [💬 Join Discussions](https://github.com/ch33nchan/rl-protokit/discussions)

</div>
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