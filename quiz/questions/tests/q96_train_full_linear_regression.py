from __future__ import annotations

import torch


def get_visible_tests():
    return [
        ("returns weight, bias, and loss", test_returns_tuple),
        ("learns the line well", test_learns_line),
    ]


def get_hidden_tests():
    return [
        ("is deterministic for the same seed", test_deterministic),
        ("returns detached outputs", test_detached_outputs),
    ]


def test_returns_tuple(solution_module):
    actual = solution_module.train_full_linear_regression(42)
    assert isinstance(actual, tuple), f"Expected tuple output, got {type(actual)}"
    assert len(actual) == 3, f"Expected 3 outputs, got {len(actual)}"


def test_learns_line(solution_module):
    weight, bias, loss = solution_module.train_full_linear_regression(42)
    assert abs(weight.item() - 3.0) < 0.15, f"Expected weight near 3.0, got {weight.item():.4f}"
    assert abs(bias.item() - 1.0) < 0.15, f"Expected bias near 1.0, got {bias.item():.4f}"
    assert loss.item() < 0.02, f"Expected final loss < 0.02, got {loss.item():.6f}"


def test_deterministic(solution_module):
    first = solution_module.train_full_linear_regression(42)
    second = solution_module.train_full_linear_regression(42)
    for left, right in zip(first, second):
        assert torch.allclose(left, right), f"Expected deterministic outputs, got {first} and {second}"


def test_detached_outputs(solution_module):
    weight, bias, loss = solution_module.train_full_linear_regression(42)
    assert not weight.requires_grad, "Expected detached weight"
    assert not bias.requires_grad, "Expected detached bias"
    assert not loss.requires_grad, "Expected detached loss"
