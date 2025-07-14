#!/usr/bin/env python3
"""
RL ProtoKit Usage Examples
==========================

This script demonstrates various usage patterns for RL ProtoKit.
Run different examples by calling the respective functions.
"""

import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Helper function to run shell commands."""
    print(f"\n🚀 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    # In a real scenario, you'd run: subprocess.run(cmd.split())
    # For demonstration, we just print what would be executed
    print(f"[DEMO] Would execute: {cmd}")

def example_1_basic_wrapper():
    """Example 1: Generate a basic environment wrapper."""
    run_command(
        "protokit generate --env CartPole-v1 --mods 'scale_rewards=0.5'",
        "Basic Environment Wrapper Generation"
    )

def example_2_advanced_atari():
    """Example 2: Advanced Atari preprocessing."""
    run_command(
        "protokit generate --env ALE/Breakout-v5 --env-type atari --frame-stack 4 --grayscale --mods 'fire_reset=True'",
        "Advanced Atari Environment Setup"
    )

def example_3_hyperparameter_tuning():
    """Example 3: Hyperparameter tuning with PPO."""
    run_command(
        "protokit tune --env LunarLander-v2 --algorithm ppo --params 'lr:[1e-4,3e-4,1e-3],gamma:[0.95,0.99]' --trials 20 --ppo-clip-anneal",
        "Hyperparameter Tuning with PPO"
    )

def example_4_continuous_control():
    """Example 4: Continuous control with SAC."""
    run_command(
        "protokit tune --env Pendulum-v1 --algorithm sac --action-head continuous --sac-temp-auto --trials 15",
        "Continuous Control with SAC"
    )

def example_5_policy_debugging():
    """Example 5: Policy debugging and analysis."""
    run_command(
        "protokit debug --model models/cartpole_dqn.pth --env CartPole-v1 --episodes 10 --visualize",
        "Policy Debugging and Analysis"
    )

def example_6_full_pipeline():
    """Example 6: Full end-to-end pipeline."""
    run_command(
        "protokit full --env CartPole-v1 --mods 'scale_rewards=0.5' --algorithm dqn --trials 10 --debug-episodes 5",
        "Full End-to-End Pipeline"
    )

def example_7_multi_agent():
    """Example 7: Multi-agent environment."""
    run_command(
        "protokit generate --env PettingZoo/connect_four_v3 --env-type multi-agent --mods 'reward_shaping=competitive'",
        "Multi-Agent Environment Setup"
    )

def example_8_research_pipeline():
    """Example 8: Research pipeline with experiment tracking."""
    run_command(
        "protokit full --env HalfCheetah-v4 --algorithm sac --wandb-log --experiment-name 'baseline_study' --trials 50",
        "Research Pipeline with Experiment Tracking"
    )

def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        example_1_basic_wrapper,
        example_2_advanced_atari,
        example_3_hyperparameter_tuning,
        example_4_continuous_control,
        example_5_policy_debugging,
        example_6_full_pipeline,
        example_7_multi_agent,
        example_8_research_pipeline,
    ]
    
    print("🎯 RL ProtoKit Usage Examples")
    print("=" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\n📋 Example {i}:")
        example()
    
    print("\n✅ All examples completed!")
    print("\n💡 Tips:")
    print("   - Modify parameters based on your research needs")
    print("   - Use --help with any command for detailed options")
    print("   - Check the interactive docs at docs/index.html")
    print("   - Configure defaults in protokit_config.yaml")

if __name__ == "__main__":
    run_all_examples()
