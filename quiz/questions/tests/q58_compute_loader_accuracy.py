from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_visible_tests():
    return [
        ("returns a float accuracy", test_returns_float),
        ("computes the correct accuracy", test_correct_accuracy),
    ]


def get_hidden_tests():
    return [
        ("switches the model to eval mode", test_sets_eval_mode),
        ("handles multiple batches", test_multiple_batches),
    ]


class FixedModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.size(0), 3)
        logits[:, 1] = 2.0
        return logits


def build_case():
    images = torch.randn(6, 1, 16, 16)
    labels = torch.tensor([1, 1, 0, 1, 2, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    return FixedModel(), loader


def test_returns_float(solution_module):
    model, loader = build_case()
    actual = solution_module.compute_loader_accuracy(model, loader)
    assert isinstance(actual, float), f"Expected a float, got {type(actual)}"


def test_correct_accuracy(solution_module):
    model, loader = build_case()
    actual = solution_module.compute_loader_accuracy(model, loader)
    assert abs(actual - (4 / 6)) < 1e-8, f"Expected accuracy 4/6, got {actual}"


def test_sets_eval_mode(solution_module):
    model, loader = build_case()
    model.train()
    solution_module.compute_loader_accuracy(model, loader)
    assert model.training is False, "Expected the function to switch the model to eval mode"


def test_multiple_batches(solution_module):
    model, loader = build_case()
    solution_module.compute_loader_accuracy(model, loader)
