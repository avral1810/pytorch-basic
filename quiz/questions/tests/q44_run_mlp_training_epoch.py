from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_visible_tests():
    return [
        ("returns the average loss as a float", test_returns_float),
        ("updates the model parameters", test_updates_parameters),
    ]


def get_hidden_tests():
    return [
        ("loss is positive", test_positive_loss),
        ("works on a full loader", test_loader_runs),
    ]


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_case():
    torch.manual_seed(0)
    x = torch.randn(64, 2)
    y = ((x[:, 0] + x[:, 1]) > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, loader, optimizer


def test_returns_float(solution_module):
    model, loader, optimizer = build_case()
    actual = solution_module.run_mlp_training_epoch(model, loader, optimizer)
    assert isinstance(actual, float), f"Expected float loss, got {type(actual)}"


def test_updates_parameters(solution_module):
    model, loader, optimizer = build_case()
    before = model.net[0].weight.detach().clone()
    solution_module.run_mlp_training_epoch(model, loader, optimizer)
    after = model.net[0].weight.detach().clone()
    assert not torch.equal(before, after), "Expected parameters to change during the training epoch"


def test_positive_loss(solution_module):
    model, loader, optimizer = build_case()
    actual = solution_module.run_mlp_training_epoch(model, loader, optimizer)
    assert actual > 0.0, f"Expected a positive average loss, got {actual}"


def test_loader_runs(solution_module):
    model, loader, optimizer = build_case()
    solution_module.run_mlp_training_epoch(model, loader, optimizer)
