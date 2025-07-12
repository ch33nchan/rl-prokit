import pytest
from hyperparam_tuner.tuner import run_tuning

def test_run_tuning():
    model_path = run_tuning({'lr': [0.001]}, 1)
    assert model_path == 'tuned_model.pth'