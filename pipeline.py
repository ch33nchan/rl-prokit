from wrapper_generator import generate_wrapper
from tuner import tune_hyperparams
from debugger import debug_policy

def run_full_pipeline(env_name, mods, params, trials):
    env = generate_wrapper(env_name, mods)
    model = tune_hyperparams(env_name, params, trials)
    debug_policy(model, env_name)

if __name__ == "__main__":
    run_full_pipeline('CartPole-v1', "scale_rewards=0.5", {'lr': 0.001}, 5)
