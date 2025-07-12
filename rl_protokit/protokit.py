import argparse
from wrapper_generator import generate_wrapper
from tuner import tune_hyperparams
from debugger import debug_policy
from pipeline import run_full_pipeline

def main():
    parser = argparse.ArgumentParser(description="RL ProtoKit CLI Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate custom Gym wrapper")
    generate_parser.add_argument("--env", required=True, help="Environment name")
    generate_parser.add_argument("--mods", default="", help="Mods like 'scale_rewards=0.5'")
    generate_parser.add_argument("--env-type", default="standard", choices=["standard", "multi-agent", "atari"], help="Environment type")
    generate_parser.add_argument("--frame-stack", type=int, default=0, help="Number of frames to stack")
    generate_parser.add_argument("--grayscale", action="store_true", help="Apply grayscale transform")
    generate_parser.add_argument("--curiosity", action="store_true", help="Add intrinsic curiosity module")

    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters")
    tune_parser.add_argument("--env", required=True, help="Environment name")
    tune_parser.add_argument("--params", default="lr:0.001", help="Params like 'lr:0.001'")
    tune_parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    tune_parser.add_argument("--replay-type", default="standard", choices=["standard", "prioritized"], help="Replay buffer type")
    tune_parser.add_argument("--policy-type", default="fc", choices=["fc", "rnn"], help="Policy network type")
    tune_parser.add_argument("--action-head", default="discrete", choices=["discrete", "continuous"], help="Action head type")
    tune_parser.add_argument("--ppo-clip-anneal", action="store_true", help="Enable PPO clip annealing")
    tune_parser.add_argument("--log-kl", action="store_true", help="Log KL-divergence for PPO")
    tune_parser.add_argument("--sac-temp-auto", action="store_true", help="Auto-tune SAC temperature")

    # Debug command
    debug_parser = subparsers.add_parser("debug", help="Debug RL policy")
    debug_parser.add_argument("--model", required=True, help="Path to trained model")
    debug_parser.add_argument("--env", required=True, help="Environment name")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full RL pipeline")
    full_parser.add_argument("--env", required=True, help="Environment name")
    full_parser.add_argument("--mods", default="", help="Mods for wrapper")
    full_parser.add_argument("--params", default="lr:0.001", help="Params for tuning")
    full_parser.add_argument("--trials", type=int, default=5, help="Number of trials")

    args = parser.parse_args()

    if args.command == "generate":
        generate_wrapper(args.env, args.mods, args.env_type, args.frame_stack, args.grayscale, args.curiosity)
    elif args.command == "tune":
        tune_hyperparams(args.env, args.params, args.trials, args.replay_type, args.policy_type, args.action_head, args.ppo_clip_anneal, args.log_kl, args.sac_temp_auto)
    elif args.command == "debug":
        debug_policy(args.model, args.env)
    elif args.command == "full":
        run_full_pipeline(args.env, args.mods, args.params, args.trials)

if __name__ == "__main__":
    main()
