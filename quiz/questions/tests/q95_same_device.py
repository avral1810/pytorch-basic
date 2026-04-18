from __future__ import annotations

import torch
from torch import nn


def get_visible_tests():
    return [
        ("returns True for matching devices", test_matching_devices),
        ("returns a bool", test_returns_bool),
    ]


def get_hidden_tests():
    return [
        ("returns False for mismatched devices", test_mismatched_devices),
        ("works with multi-layer modules", test_multilayer_module),
    ]


def test_matching_devices(solution_module):
    model = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    actual = solution_module.same_device(model, x)
    assert actual is True, f"Expected True for matching cpu devices, got {actual}"


def test_returns_bool(solution_module):
    model = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    actual = solution_module.same_device(model, x)
    assert isinstance(actual, bool), f"Expected bool, got {type(actual)}"


def test_mismatched_devices(solution_module):
    class FakeTensor:
        device = torch.device("meta")

    model = nn.Linear(4, 2)
    actual = solution_module.same_device(model, FakeTensor())
    assert actual is False, f"Expected False for mismatched devices, got {actual}"


def test_multilayer_module(solution_module):
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(5, 4)
    actual = solution_module.same_device(model, x)
    assert actual is True
