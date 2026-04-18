from __future__ import annotations


EXTRA_ANSWER_CODES = {
    "q36_init_manual_mlp_params": """import torch


def init_manual_mlp_params(seed: int = 42):
    torch.manual_seed(seed)
    w1 = (torch.randn(2, 16) * 0.5).requires_grad_()
    b1 = torch.zeros(16, requires_grad=True)
    w2 = (torch.randn(16, 2) * 0.5).requires_grad_()
    b2 = torch.zeros(2, requires_grad=True)
    return w1, b1, w2, b2
""",
    "q37_manual_mlp_forward": """import torch


def manual_mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    hidden = torch.relu(x @ w1 + b1)
    return hidden @ w2 + b2
""",
    "q38_manual_predict_classes": """import torch


def manual_predict_classes(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)
""",
    "q39_manual_mlp_loss": """import torch
import torch.nn.functional as F


def manual_mlp_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)
""",
    "q40_train_manual_mlp": """import torch
import torch.nn.functional as F


def train_manual_mlp(seed: int = 42) -> float:
    torch.manual_seed(seed)
    x = torch.rand(600, 2) * 2 - 1
    y = ((x[:, 0] * x[:, 1]) > 0).long()
    x = x + 0.15 * torch.randn_like(x)

    w1 = (torch.randn(2, 16) * 0.5).requires_grad_()
    b1 = torch.zeros(16, requires_grad=True)
    w2 = (torch.randn(16, 2) * 0.5).requires_grad_()
    b2 = torch.zeros(2, requires_grad=True)
    params = [w1, b1, w2, b2]

    for _ in range(120):
        hidden = torch.relu(x @ w1 + b1)
        logits = hidden @ w2 + b2
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            for param in params:
                param -= 0.05 * param.grad
        for param in params:
            param.grad.zero_()

    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()
""",
    "q41_build_mlp_sequential": """from torch import nn


def build_mlp_sequential():
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
""",
    "q42_make_classification_loader": """from torch.utils.data import DataLoader, TensorDataset


def make_classification_loader(x, y):
    return DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
""",
    "q43_compute_multiclass_accuracy": """import torch


def compute_multiclass_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
""",
    "q44_run_mlp_training_epoch": """from torch import nn


def run_mlp_training_epoch(model, loader, optimizer) -> float:
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    steps = 0
    for batch_x, batch_y in loader:
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / steps
""",
    "q45_train_module_mlp": """import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_module_mlp(seed: int = 42) -> float:
    torch.manual_seed(seed)
    x = torch.rand(800, 2) * 2 - 1
    y = ((x[:, 0] * x[:, 1]) > 0).long()
    x = x + 0.15 * torch.randn_like(x)
    split = 640
    loader = DataLoader(TensorDataset(x[:split], y[:split]), batch_size=64, shuffle=True)
    val_x = x[split:]
    val_y = y[split:]

    model = MLPClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(40):
        model.train()
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(val_x)
        preds = logits.argmax(dim=1)
        return (preds == val_y).float().mean().item()
""",
    "q46_init_manual_cnn_params": """import torch


def init_manual_cnn_params(seed: int = 123):
    torch.manual_seed(seed)
    conv1_weight = (torch.randn(8, 1, 3, 3) * 0.1).requires_grad_()
    conv1_bias = torch.zeros(8, requires_grad=True)
    conv2_weight = (torch.randn(16, 8, 3, 3) * 0.1).requires_grad_()
    conv2_bias = torch.zeros(16, requires_grad=True)
    fc1_weight = (torch.randn(16 * 4 * 4, 32) * 0.1).requires_grad_()
    fc1_bias = torch.zeros(32, requires_grad=True)
    fc2_weight = (torch.randn(32, 3) * 0.1).requires_grad_()
    fc2_bias = torch.zeros(3, requires_grad=True)
    return (
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )
""",
    "q47_manual_conv_relu_pool": """import torch
import torch.nn.functional as F


def manual_conv_relu_pool(
    images: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor
) -> torch.Tensor:
    x = F.conv2d(images, conv_weight, conv_bias, padding=1)
    x = torch.relu(x)
    return F.max_pool2d(x, kernel_size=2)
""",
    "q48_flatten_image_features": """import torch


def flatten_image_features(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1)
""",
    "q49_manual_cnn_classifier": """import torch


def manual_cnn_classifier(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> torch.Tensor:
    hidden = torch.relu(x @ fc1_weight + fc1_bias)
    return hidden @ fc2_weight + fc2_bias
""",
    "q50_predict_image_classes": """import torch


def predict_image_classes(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)
""",
    "q51_build_cnn_feature_extractor": """from torch import nn


def build_cnn_feature_extractor():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
""",
    "q52_conv_output_size": """def conv_output_size(
    input_size: int, kernel_size: int, padding: int = 0, stride: int = 1
) -> int:
    return ((input_size + 2 * padding - kernel_size) // stride) + 1
""",
    "q53_build_cnn_classifier_head": """from torch import nn


def build_cnn_classifier_head():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )
""",
    "q54_flatten_cnn_batch": """import torch


def flatten_cnn_batch(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1)
""",
    "q55_predict_cnn_classes": """import torch


def predict_cnn_classes(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)
""",
    "q56_add_channel_dim": """import torch


def add_channel_dim(image: torch.Tensor) -> torch.Tensor:
    return image.unsqueeze(0)
""",
    "q57_build_pattern_loader": """from torch.utils.data import DataLoader, TensorDataset


def build_pattern_loader(images, labels):
    return DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=True)
""",
    "q58_compute_loader_accuracy": """import torch


def compute_loader_accuracy(model, loader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
""",
    "q59_build_vision_cnn": """from torch import nn


def build_vision_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )
""",
    "q60_generate_diagonal_pattern": """import torch


def generate_diagonal_pattern(image_size: int) -> torch.Tensor:
    image = torch.zeros(image_size, image_size)
    idx = torch.arange(image_size)
    image[idx, idx] = 1.0
    return image.unsqueeze(0)
""",
    "q61_embedding_lookup": """import torch


def embedding_lookup(
    embedding_table: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    return embedding_table[tokens]
""",
    "q62_manual_rnn_step": """import torch


def manual_rnn_step(
    x_t: torch.Tensor,
    hidden: torch.Tensor,
    wxh: torch.Tensor,
    whh: torch.Tensor,
    bh: torch.Tensor,
) -> torch.Tensor:
    return torch.tanh(x_t @ wxh + hidden @ whh + bh)
""",
    "q63_run_manual_rnn": """import torch


def run_manual_rnn(
    embedded: torch.Tensor,
    wxh: torch.Tensor,
    whh: torch.Tensor,
    bh: torch.Tensor,
) -> torch.Tensor:
    hidden = torch.zeros(embedded.size(0), whh.size(0))
    for t in range(embedded.size(1)):
        x_t = embedded[:, t, :]
        hidden = torch.tanh(x_t @ wxh + hidden @ whh + bh)
    return hidden
""",
    "q64_manual_sequence_logits": """import torch


def manual_sequence_logits(
    hidden: torch.Tensor, why: torch.Tensor, by: torch.Tensor
) -> torch.Tensor:
    return hidden @ why + by
""",
    "q65_predict_manual_rnn_classes": """import torch


def predict_manual_rnn_classes(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)
""",
    "q66_build_vanilla_rnn": """from torch import nn


def build_vanilla_rnn():
    return nn.RNN(input_size=32, hidden_size=32, batch_first=True, nonlinearity="tanh")
""",
    "q67_take_last_hidden": """import torch


def take_last_hidden(hidden_state: torch.Tensor) -> torch.Tensor:
    return hidden_state[-1]
""",
    "q68_make_rnn_labels": """import torch


def make_rnn_labels(tokens: torch.Tensor) -> torch.Tensor:
    first_half = tokens[:, : tokens.size(1) // 2].float().mean(dim=1)
    second_half = tokens[:, tokens.size(1) // 2 :].float().mean(dim=1)
    return (second_half > first_half).long()
""",
    "q69_build_rnn_classifier_head": """from torch import nn


def build_rnn_classifier_head():
    return nn.Linear(32, 2)
""",
    "q70_predict_rnn_classes": """import torch


def predict_rnn_classes(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)
""",
    "q71_build_embedding_layer": """from torch import nn


def build_embedding_layer():
    return nn.Embedding(15, 32)
""",
    "q72_build_lstm_layer": """from torch import nn


def build_lstm_layer():
    return nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
""",
    "q73_take_lstm_final_hidden": """import torch


def take_lstm_final_hidden(hidden_state: torch.Tensor) -> torch.Tensor:
    return hidden_state[-1]
""",
    "q74_build_lstm_classifier_head": """from torch import nn


def build_lstm_classifier_head():
    return nn.Linear(32, 2)
""",
    "q75_predict_lstm_classes": """import torch


def predict_lstm_classes(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)
""",
    "q76_positional_encoding": """import math
import torch


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
""",
    "q77_scaled_attention_scores": """import math
import torch


def scaled_attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return q @ k.transpose(1, 2) / math.sqrt(q.size(-1))
""",
    "q78_attention_weighted_values": """import torch


def attention_weighted_values(
    weights: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return weights @ v
""",
    "q79_mean_pool_sequence": """import torch


def mean_pool_sequence(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)
""",
    "q80_manual_transformer_logits": """import torch


def manual_transformer_logits(
    pooled: torch.Tensor, w_cls: torch.Tensor, b_cls: torch.Tensor
) -> torch.Tensor:
    return pooled @ w_cls + b_cls
""",
    "q81_pool_encoder_output": """import torch


def pool_encoder_output(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)
""",
    "q82_build_transformer_encoder_layer": """from torch import nn


def build_transformer_encoder_layer():
    return nn.TransformerEncoderLayer(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.1,
        batch_first=True,
    )
""",
    "q83_build_transformer_classifier_head": """from torch import nn


def build_transformer_classifier_head():
    return nn.Linear(32, 2)
""",
    "q84_make_transformer_labels": """import torch


def make_transformer_labels(
    tokens: torch.Tensor, vocab_size: int = 20
) -> torch.Tensor:
    threshold = tokens.size(1) * vocab_size / 2
    return (tokens.sum(dim=1) > threshold).long()
""",
    "q85_predict_transformer_classes": """import torch


def predict_transformer_classes(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)
""",
    "q86_build_generator": """from torch import nn


def build_generator():
    return nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
""",
    "q87_build_discriminator": """from torch import nn


def build_discriminator():
    return nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
""",
    "q88_sample_gan_noise": """import torch


def sample_gan_noise(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, 3)
""",
    "q89_make_real_and_fake_targets": """import torch


def make_real_and_fake_targets(logits: torch.Tensor):
    return torch.ones_like(logits), torch.zeros_like(logits)
""",
    "q90_detach_fake_batch": """import torch


def detach_fake_batch(fake_samples: torch.Tensor) -> torch.Tensor:
    return fake_samples.detach()
""",
    "q91_get_device": """import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
""",
    "q92_move_module_to_device": """def move_module_to_device(model, device):
    return model.to(device)
""",
    "q93_make_device_inputs": """import torch


def make_device_inputs(batch_size: int, device):
    x = torch.randn(batch_size, 4, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device)
    return x, y
""",
    "q94_move_tensor_pair_to_device": """def move_tensor_pair_to_device(x, y, device):
    return x.to(device), y.to(device)
""",
    "q95_same_device": """def same_device(model, x) -> bool:
    return next(model.parameters()).device == x.device
""",
    "q96_train_full_linear_regression": """import torch
from torch import nn


def train_full_linear_regression(seed: int = 42):
    torch.manual_seed(seed)
    x = torch.linspace(-1.0, 1.0, steps=80).unsqueeze(1)
    y = 3.0 * x + 1.0

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(120):
        preds = model(x)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (
        model.weight.detach().clone(),
        model.bias.detach().clone(),
        loss.detach().clone(),
    )
""",
    "q97_train_full_logistic_regression": """import torch
from torch import nn


def train_full_logistic_regression(seed: int = 42) -> float:
    torch.manual_seed(seed)
    x = torch.rand(600, 2) * 2 - 1
    y = ((x[:, 0] + x[:, 1]) > 0).float().unsqueeze(1)
    x = x + 0.10 * torch.randn_like(x)

    model = nn.Linear(2, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for _ in range(120):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(x))
        preds = (probs >= 0.5).float()
        return (preds == y).float().mean().item()
""",
}
