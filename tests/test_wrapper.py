import pytest
from wrapper_generator.wrapper_generator import generate_wrapper

def test_generate_wrapper():
    code = generate_wrapper('CartPole-v1', 'scale_rewards=0.5')
    assert 'CustomWrapper' in code
    assert 'reward *= 0.5' in code