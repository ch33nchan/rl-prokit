import gymnasium as gym
import numpy as np
import re  # Added for robust range parsing

def parse_range(value):
    """Parse a range string like '-10-10' into two floats, handling negatives."""
    match = re.match(r'^(-?\d*\.?\d+)-(-?\d*\.?\d+)$', value)
    if not match:
        raise ValueError(f"Invalid range format: {value}. Use 'low-high' (e.g., '-10-10').")
    low, high = match.groups()
    return float(low), float(high)

def generate_wrapper(base_env_name, mods):
    mod_code = ""
    imports = "import gymnasium as gym\nimport numpy as np\n"
    class_extra = ""  # For rich features needing class-level vars

    for mod in mods:
        key, value = mod.split('=') if '=' in mod else (mod, None)
        key = key.strip()
        if key == 'scale_rewards':
            scale = float(value)
            mod_code += f"        reward *= {scale}  # Scale rewards\n"
        elif key == 'add_noise':
            noise = float(value)
            mod_code += f"        obs += np.random.normal(0, {noise}, obs.shape)  # Add observation noise\n"
        elif key == 'clip_actions':
            low, high = parse_range(value)
            mod_code += f"        action = np.clip(action, {low}, {high})  # Clip actions\n"
        elif key == 'clip_rewards':
            low, high = parse_range(value)
            mod_code += f"        reward = np.clip(reward, {low}, {high})  # Clip rewards\n"
        elif key == 'normalize_states':
            if value.lower() == 'true':
                imports += "from gymnasium.wrappers import NormalizeObservation\n"
                class_extra += "        self = NormalizeObservation(self)  # Normalize states (running mean/std)\n"
        elif key == 'bonus_for_termination':
            bonus = float(value)
            mod_code += f"        if terminated: reward += {bonus}  # Bonus reward on termination\n"
        # Add more rich features here as needed

    wrapper_template = f"""
{imports}

class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
{class_extra}
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
{mod_code}
        return obs, reward, terminated, truncated, info

# Usage: env = CustomWrapper(gym.make('{base_env_name}'))
"""
    return wrapper_template
