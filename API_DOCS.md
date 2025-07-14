# RL ProtoKit API Documentation

## Core Modules

### 1. Wrapper Generator (`wrapper_generator/`)

#### `generate_wrapper(env_name, mods, **kwargs)`

Generate custom environment wrappers with specified modifications.

**Parameters:**
- `env_name` (str): Gymnasium environment name (e.g., "CartPole-v1")
- `mods` (str): Comma-separated modifications (e.g., "scale_rewards=0.5,clip_actions=True")
- `env_type` (str): Environment type - "standard", "multi-agent", "atari"
- `frame_stack` (int): Number of frames to stack (default: 0)
- `grayscale` (bool): Apply grayscale transformation (default: False)
- `curiosity` (bool): Add intrinsic curiosity module (default: False)

**Returns:**
- `gym.Env`: Wrapped environment ready for training

**Example:**
```python
from rl_protokit.wrapper_generator import generate_wrapper

env = generate_wrapper(
    env_name="CartPole-v1",
    mods="scale_rewards=0.5,normalize_obs=True",
    frame_stack=4,
    curiosity=True
)
```

#### Available Wrapper Classes

##### `CuriosityWrapper`
Implements Intrinsic Curiosity Module (ICM) for exploration enhancement.

**Features:**
- Forward and inverse dynamics models
- Intrinsic reward computation
- Configurable curiosity strength (β parameter)

##### `AtariWrapper`
Specialized wrapper for Atari environments.

**Features:**
- Fire-reset mechanism
- Max-pooling over consecutive frames
- Frame skipping
- Action space simplification

##### `MultiAgentWrapper`
Integration with PettingZoo for multi-agent environments.

**Features:**
- Agent coordination utilities
- Reward sharing mechanisms
- Observation space management

### 2. Hyperparameter Tuner (`hyperparam_tuner/`)

#### `tune_hyperparams(env_name, params, **kwargs)`

Perform systematic hyperparameter optimization.

**Parameters:**
- `env_name` (str): Environment name
- `params` (dict): Parameter grid for optimization
- `trials` (int): Number of optimization trials
- `algorithm` (str): RL algorithm - "dqn", "ppo", "sac", "ddpg", "a2c"
- `replay_type` (str): Replay buffer type - "standard", "prioritized"
- `policy_type` (str): Network architecture - "fc", "rnn"
- `action_head` (str): Action space handling - "discrete", "continuous"

**Returns:**
- `dict`: Best hyperparameters and performance metrics

**Example:**
```python
from rl_protokit.hyperparam_tuner import tune_hyperparams

results = tune_hyperparams(
    env_name="LunarLander-v2",
    params={
        "lr": [1e-4, 3e-4, 1e-3],
        "gamma": [0.95, 0.99, 0.999],
        "batch_size": [32, 64, 128]
    },
    algorithm="ppo",
    trials=20,
    ppo_clip_anneal=True
)
```

#### Supported Algorithms

##### Deep Q-Network (DQN)
- **Features**: Double DQN, Dueling networks, Prioritized Experience Replay
- **Parameters**: learning_rate, gamma, epsilon_decay, target_update_freq
- **Best for**: Discrete action spaces, sample efficiency

##### Proximal Policy Optimization (PPO)
- **Features**: Clip ratio annealing, KL-divergence monitoring, GAE
- **Parameters**: learning_rate, clip_ratio, entropy_coef, value_coef
- **Best for**: Stable policy learning, continuous and discrete actions

##### Soft Actor-Critic (SAC)
- **Features**: Automatic temperature tuning, off-policy learning
- **Parameters**: learning_rate, tau, alpha, target_entropy
- **Best for**: Continuous control, exploration

##### Deep Deterministic Policy Gradient (DDPG)
- **Features**: Deterministic policy, experience replay
- **Parameters**: learning_rate, tau, noise_std
- **Best for**: Continuous control, deterministic policies

#### Advanced Features

##### Prioritized Experience Replay (PER)
- **Implementation**: Sum-tree data structure for efficient sampling
- **Benefits**: Faster convergence, better sample efficiency
- **Parameters**: alpha (prioritization strength), beta (importance sampling)

##### Recurrent Neural Networks (RNN)
- **Architectures**: LSTM, GRU support
- **Benefits**: Partial observability handling, temporal dependencies
- **Parameters**: hidden_size, num_layers, sequence_length

### 3. Policy Debugger (`policy_debugger/`)

#### `debug_policy(model, env_name, **kwargs)`

Interactive policy debugging and analysis.

**Parameters:**
- `model` (str/torch.nn.Module): Path to model file or loaded model
- `env_name` (str): Environment name
- `episodes` (int): Number of episodes to debug
- `mode` (str): Debug mode - "interactive", "q_analysis", "policy_analysis"
- `visualize` (bool): Enable visualization
- `save_states` (bool): Save episode states for analysis

**Returns:**
- `dict`: Debug results and analysis metrics

**Example:**
```python
from rl_protokit.policy_debugger import debug_policy

results = debug_policy(
    model="models/cartpole_dqn.pth",
    env_name="CartPole-v1",
    episodes=10,
    mode="q_analysis",
    visualize=True
)
```

#### Debug Modes

##### Interactive Mode
- Step-by-step execution with user control
- State inspection and action selection
- Real-time performance metrics

##### Q-Value Analysis
- Q-value visualization and heatmaps
- Confidence interval computation
- Action preference analysis

##### Policy Analysis
- Action distribution visualization
- Policy entropy computation
- Behavioral pattern analysis

### 4. Pipeline Manager (`pipeline.py`)

#### `run_full_pipeline(env_name, **kwargs)`

Execute complete RL workflow from wrapper generation to policy debugging.

**Parameters:**
- `env_name` (str): Environment name
- `mods` (str): Environment modifications
- `params` (dict): Hyperparameter grid
- `trials` (int): Tuning trials
- `algorithm` (str): RL algorithm
- `debug_episodes` (int): Episodes for debugging

**Example:**
```python
from rl_protokit.pipeline import run_full_pipeline

results = run_full_pipeline(
    env_name="CartPole-v1",
    mods="scale_rewards=0.5",
    params={"lr": [0.001, 0.01]},
    algorithm="dqn",
    trials=10,
    debug_episodes=5
)
```

## Command Line Interface

### Core Commands

#### `protokit generate`
Generate environment wrappers.

```bash
protokit generate --env ENV_NAME [options]
```

**Options:**
- `--env`: Environment name (required)
- `--mods`: Modifications string
- `--env-type`: Environment type
- `--frame-stack`: Frame stacking count
- `--grayscale`: Enable grayscale
- `--curiosity`: Enable curiosity module

#### `protokit tune`
Hyperparameter optimization.

```bash
protokit tune --env ENV_NAME --params PARAMS [options]
```

**Options:**
- `--env`: Environment name (required)
- `--params`: Parameter grid (required)
- `--algorithm`: RL algorithm
- `--trials`: Number of trials
- `--replay-type`: Replay buffer type
- `--policy-type`: Network architecture

#### `protokit debug`
Policy debugging and analysis.

```bash
protokit debug --model MODEL_PATH --env ENV_NAME [options]
```

**Options:**
- `--model`: Model file path (required)
- `--env`: Environment name (required)
- `--episodes`: Number of episodes
- `--mode`: Debug mode
- `--visualize`: Enable visualization

#### `protokit full`
Full pipeline execution.

```bash
protokit full --env ENV_NAME [options]
```

**Options:**
- `--env`: Environment name (required)
- `--mods`: Environment modifications
- `--params`: Hyperparameter grid
- `--algorithm`: RL algorithm
- `--trials`: Tuning trials

### Configuration

#### YAML Configuration
Create `protokit_config.yaml` for default settings:

```yaml
defaults:
  algorithm: ppo
  trials: 10
  save_models: true

environments:
  atari:
    frame_stack: 4
    grayscale: true
    env_type: atari

algorithms:
  ppo:
    ppo_clip_anneal: true
    log_kl: true
```

#### Environment Variables
- `PROTOKIT_CONFIG`: Path to configuration file
- `PROTOKIT_MODEL_DIR`: Default model save directory
- `PROTOKIT_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Research Integration

### Weights & Biases
Automatic experiment tracking and visualization.

```python
# Enable W&B logging
import wandb
wandb.init(project="rl_research", name="experiment_1")

# Run with tracking
results = tune_hyperparams(
    env_name="HalfCheetah-v4",
    algorithm="sac",
    wandb_log=True
)
```

### Statistical Analysis
Built-in statistical testing and significance analysis.

```python
from rl_protokit.analysis import compare_experiments

# Compare multiple experiments
comparison = compare_experiments(
    experiments=["exp1", "exp2", "exp3"],
    metrics=["final_reward", "sample_efficiency"],
    significance_test="wilcoxon"
)
```

### Custom Extensions
Extend functionality with custom components.

```python
from rl_protokit.extensions import register_algorithm

@register_algorithm("custom_dqn")
class CustomDQN:
    def __init__(self, **kwargs):
        # Custom implementation
        pass
    
    def train(self, env, **kwargs):
        # Training logic
        pass
```

## Performance Optimization

### Parallel Training
Utilize multiple CPU cores for hyperparameter tuning.

```python
results = tune_hyperparams(
    env_name="CartPole-v1",
    params={"lr": [0.001, 0.01, 0.1]},
    trials=30,
    n_jobs=4  # Parallel workers
)
```

### GPU Acceleration
Automatic GPU detection and utilization.

```python
# GPU automatically detected if available
results = tune_hyperparams(
    env_name="Breakout-v5",
    algorithm="dqn",
    device="auto"  # or "cuda:0", "cpu"
)
```

### Memory Management
Efficient memory usage for large-scale experiments.

```python
results = tune_hyperparams(
    env_name="LunarLander-v2",
    algorithm="ppo",
    memory_efficient=True,  # Reduced memory footprint
    checkpoint_freq=100     # Regular checkpointing
)
```
