import pytest
from policy_debugger.debugger import debug_policy

def test_debug_policy(capsys):
    debug_policy('tuned_model.pth', 1)
    # Basic check; expand as needed
    assert True



