from __future__ import annotations

import torch
from torch import nn


def get_visible_tests():
    return [
        ("returns the module", test_returns_module),
        ("moves parameters to the requested device", test_moves_parameters),
    ]


def get_hidden_tests():
    return [
        ("works with cpu device objects", test_cpu_device_object),
        ("keeps module behavior intact", test_forward_still_works),
    ]


def build_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_returns_module(solution_module):
    model = build_model()
    actual = solution_module.move_module_to_device(model, torch.device("cpu"))
    assert isinstance(actual, nn.Module), f"Expected an nn.Module, got {type(actual)}"


def test_moves_parameters(solution_module):
    model = build_model()
    actual = solution_module.move_module_to_device(model, "cpu")
    assert next(actual.parameters()).device.type == "cpu"


def test_cpu_device_object(solution_module):
    model = build_model()
    actual = solution_module.move_module_to_device(model, torch.device("cpu"))
    assert next(actual.parameters()).device.type == "cpu"


def test_forward_still_works(solution_module):
    model = build_model()
    actual = solution_module.move_module_to_device(model, "cpu")
    output = actual(torch.randn(3, 4))
    assert tuple(output.shape) == (3, 2)
