from __future__ import annotations


ANSWER_CODES = {
    "q30_create_tensor_from_list": """import torch


def make_tensor(values) -> torch.Tensor:
    return torch.tensor(values)
""",
    "q31_tensor_dtype": """import torch


def make_float_tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)
""",
    "q32_make_zeros_tensor": """import torch


def make_zeros(rows: int, cols: int) -> torch.Tensor:
    return torch.zeros(rows, cols)
""",
    "q33_slice_first_row": """import torch


def first_row(x: torch.Tensor) -> torch.Tensor:
    return x[0]
""",
    "q34_tensor_num_dims": """import torch


def num_dims(x: torch.Tensor) -> int:
    return x.ndim
""",
    "q35_sum_columns": """import torch


def sum_columns(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=0)
""",
    "q00_add_tensors": """import torch


def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b
""",
    "q01_trainable_scalar": """import torch


def build_trainable_scalar(value: float) -> torch.Tensor:
    return torch.tensor(value, requires_grad=True)
""",
    "q08_matrix_multiply": """import torch


def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b
""",
    "q09_tensor_mean": """import torch


def tensor_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean()
""",
    "q10_gradient_step": """import torch


def gradient_step(weight: torch.Tensor, learning_rate: float) -> torch.Tensor:
    loss = (3 * weight - 9) ** 2
    loss.backward()
    with torch.no_grad():
        weight -= learning_rate * weight.grad
    return weight
""",
    "q02_reshape_to_column": """import torch


def reshape_to_column(values) -> torch.Tensor:
    return torch.tensor(values).reshape(-1, 1)
""",
    "q11_add_batch_dim": """import torch


def add_batch_dim(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0)
""",
    "q12_remove_singleton_dim": """import torch


def remove_singleton_dim(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(1)
""",
    "q13_flatten_tensor": """import torch


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)
""",
    "q14_add_bias": """import torch


def add_bias(matrix: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return matrix + bias
""",
    "q03_linear_model": """import torch


def linear_model(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight + bias
""",
    "q04_mse_loss": """import torch


def mean_squared_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return ((preds - targets) ** 2).mean()
""",
    "q15_make_linear_params": """import torch


def make_linear_params():
    weight = torch.randn(1, 1, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)
    return weight, bias
""",
    "q16_compute_residuals": """import torch


def compute_residuals(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return preds - targets
""",
    "q17_apply_sgd_update": """import torch


def apply_sgd_update(weight: torch.Tensor, bias: torch.Tensor, learning_rate: float):
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad
    return weight, bias
""",
    "q05_tiny_linear_forward": """import torch
from torch import nn


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
""",
    "q18_build_linear_layer": """from torch import nn


def build_linear_layer():
    return nn.Linear(1, 1)
""",
    "q19_make_regression_loader": """from torch.utils.data import DataLoader, TensorDataset


def make_regression_loader(x, y):
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
""",
    "q20_compute_mse_loss": """from torch import nn


def compute_mse_loss(preds, targets):
    loss_fn = nn.MSELoss()
    return loss_fn(preds, targets)
""",
    "q21_run_training_step": """from torch import nn


def run_training_step(model, batch_x, batch_y, optimizer):
    loss_fn = nn.MSELoss()
    optimizer.zero_grad()
    preds = model(batch_x)
    loss = loss_fn(preds, batch_y)
    loss.backward()
    optimizer.step()
    return loss
""",
    "q06_logistic_probability": """import torch


def logistic_probability(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.sigmoid(x @ weight + bias)
""",
    "q22_sigmoid_function": """import torch


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-z))
""",
    "q23_threshold_predictions": """import torch


def threshold_predictions(probs: torch.Tensor) -> torch.Tensor:
    return (probs >= 0.5).float()
""",
    "q24_make_binary_labels": """import torch


def make_binary_labels(scores: torch.Tensor) -> torch.Tensor:
    return (scores > 0).float().unsqueeze(1)
""",
    "q25_binary_cross_entropy": """import torch


def binary_cross_entropy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.clamp(preds, 1e-7, 1 - 1e-7)
    loss = -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
    return loss.mean()
""",
    "q07_logistic_module_forward": """import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
""",
    "q26_build_logistic_layer": """from torch import nn


def build_logistic_layer():
    return nn.Linear(2, 1)
""",
    "q27_logits_to_probs": """import torch


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)
""",
    "q28_make_bce_loss": """from torch import nn


def make_bce_loss():
    return nn.BCEWithLogitsLoss()
""",
    "q29_predict_binary_classes": """import torch


def predict_binary_classes(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs >= 0.5).float()
""",
}
