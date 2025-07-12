import click
from wrapper_generator.wrapper_generator import generate_wrapper
from hyperparam_tuner.tuner import run_tuning
from policy_debugger.debugger import debug_policy
import ast  # For safe param parsing

@click.group()
def main():
    pass

def parse_params(param_str):
    """Parse params like 'lr: [0.001, 0.01]' into dict."""
    key, values_str = param_str.split(':')
    values = ast.literal_eval(values_str.strip())
    return {key.strip(): values}

@main.command()
@click.option('--env', required=True, help='Base Gym environment name')
@click.option('--mods', multiple=True, help='Modifications, e.g., "scale_rewards=0.5" (repeat for multiple configs)')
@click.option('--output', multiple=True, default=['custom_wrapper.py'], help='Output filename(s); defaults to custom_wrapper.py (repeat for multiple)')
def generate(env, mods, output):
    if len(mods) == 0:
        click.echo('No modifications provided; generating basic wrapper.')
        mods = ['']
    
    if len(output) < len(mods):
        # Auto-name additional files if not enough outputs specified
        base_name = output[0].replace('.py', '')
        output = [f"{base_name}_{i}.py" if i > 0 else output[0] for i in range(len(mods))]
    
    for idx, mod_set in enumerate(mods):
        mod_list = mod_set.split(',') if mod_set else []  # Support comma-separated mods in one flag
        wrapper_code = generate_wrapper(env, mod_list)
        out_file = output[idx]
        with open(out_file, 'w') as f:
            f.write(wrapper_code)
        click.echo(f'Wrapper generated in {out_file}')

@main.command()
@click.option('--params', required=True, help='Params as "key: [val1, val2]"')
@click.option('--trials', default=5, help='Number of trials per combination')
def tune(params, trials):
    param_dict = parse_params(params)
    best_model = run_tuning(param_dict, trials)
    click.echo(f'Best model saved as {best_model}')

@main.command()
@click.option('--model', required=True, help='Path to model file')
@click.option('--steps', default=10, help='Number of debug steps')
def debug(model, steps):
    debug_policy(model, steps)

@main.command()
@click.option('--env', required=True)
@click.option('--mods', multiple=True)
@click.option('--params', required=True)
@click.option('--trials', default=5)
@click.option('--debug_steps', default=10)
def full(env, mods, params, trials, debug_steps):
    generate(env, mods)  # Calls the generate command logic
    param_dict = parse_params(params)
    run_tuning(param_dict, trials)
    debug_policy('tuned_model.pth', debug_steps)
    click.echo('Full pipeline completed!')

if __name__ == '__main__':
    main() 