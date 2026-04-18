from __future__ import annotations

from quiz.questions.models import QuestionAnswer, QuestionTemplate, QuestionTests, QuestionText, QuizQuestion


def question(
    question_id: str,
    chapter_id: str,
    title: str,
    lesson: str,
    prompt: str,
    starter_code: str,
    visible_examples: list[str],
    test_module: str,
    *,
    symbol_name: str | None = None,
    hidden_binary_key: str | None = None,
) -> QuizQuestion:
    return QuizQuestion(
        id=question_id,
        chapter_id=chapter_id,
        text=QuestionText(title=title, prompt=prompt),
        template=QuestionTemplate(starter_code=starter_code),
        answer=QuestionAnswer(lesson=lesson, symbol_name=symbol_name),
        tests=QuestionTests(module_name=test_module, hidden_binary_key=hidden_binary_key),
        visible_examples=tuple(visible_examples),
    )


EXTRA_QUESTIONS = [
    question(
        "q36_init_manual_mlp_params",
        "06",
        "Initialize Manual MLP Parameters",
        "06_manual_mlp_from_scratch",
        "Write `init_manual_mlp_params(seed=42)` that sets `torch.manual_seed(seed)` and returns `(w1, b1, w2, b2)` for a `2 -> 16 -> 2` manual MLP. The weights should use `torch.randn(...)*0.5` and every parameter must require gradients.",
        """import torch


def init_manual_mlp_params(seed: int = 42):
    # Return w1, b1, w2, b2 for a manual 2 -> 16 -> 2 MLP.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["`w1` should have shape `(2, 16)` and `w2` should have shape `(16, 2)`.", "All returned tensors should require gradients."],
        "quiz.questions.tests.q36_init_manual_mlp_params",
        symbol_name="init_manual_mlp_params",
        hidden_binary_key="q36_init_manual_mlp_params",
    ),
    question(
        "q37_manual_mlp_forward",
        "06",
        "Manual MLP Forward Pass",
        "06_manual_mlp_from_scratch",
        "Write `manual_mlp_forward(x, w1, b1, w2, b2)` that computes `relu(x @ w1 + b1)` and then returns logits from `hidden @ w2 + b2`.",
        """import torch


def manual_mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    # Return the 2-class logits for the batch.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `x` is shape `(batch, 2)`, the output should be shape `(batch, 2)`.", "Use ReLU on the hidden layer before the final logits."],
        "quiz.questions.tests.q37_manual_mlp_forward",
        symbol_name="manual_mlp_forward",
        hidden_binary_key="q37_manual_mlp_forward",
    ),
    question(
        "q38_manual_predict_classes",
        "06",
        "Predict Classes From Logits",
        "06_manual_mlp_from_scratch",
        "Write `manual_predict_classes(logits)` that returns the predicted class index for each example.",
        """import torch


def manual_predict_classes(logits: torch.Tensor) -> torch.Tensor:
    # Return one class id per row of logits.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Logits with shape `(batch, 2)` should return predictions with shape `(batch,)`.", "Use the largest logit in each row."],
        "quiz.questions.tests.q38_manual_predict_classes",
        symbol_name="manual_predict_classes",
        hidden_binary_key="q38_manual_predict_classes",
    ),
    question(
        "q39_manual_mlp_loss",
        "06",
        "Manual MLP Cross Entropy",
        "06_manual_mlp_from_scratch",
        "Write `manual_mlp_loss(logits, labels)` using `torch.nn.functional.cross_entropy`.",
        """import torch
import torch.nn.functional as F


def manual_mlp_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Return the scalar cross-entropy loss.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The output should be a scalar tensor.", "Pass raw logits and integer class labels into cross entropy."],
        "quiz.questions.tests.q39_manual_mlp_loss",
        symbol_name="manual_mlp_loss",
        hidden_binary_key="q39_manual_mlp_loss",
    ),
    question(
        "q40_train_manual_mlp",
        "06",
        "Train A Manual MLP",
        "06_manual_mlp_from_scratch",
        "Write `train_manual_mlp(seed=42)` that sets the seed, makes XOR-like data, builds the manual `2 -> 16 -> 2` MLP from raw tensors, trains it for multiple epochs with manual gradient updates, and returns the final accuracy as a Python float.",
        """import torch
import torch.nn.functional as F


def train_manual_mlp(seed: int = 42) -> float:
    # Train the manual MLP end to end and return the final accuracy.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `torch.manual_seed(seed)` so the result is deterministic.", "Return the final accuracy as a float, not logits or parameter tensors."],
        "quiz.questions.tests.q40_train_manual_mlp",
        symbol_name="train_manual_mlp",
    ),
    question(
        "q41_build_mlp_sequential",
        "07",
        "Build An nn.Sequential MLP",
        "07_basic_nn_mlp_with_nn_module",
        "Write `build_mlp_sequential()` that returns `nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))`.",
        """from torch import nn


def build_mlp_sequential():
    # Return the small 2 -> 16 -> 16 -> 2 classifier.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The returned object should be an `nn.Sequential` module.", "The first linear layer should read 2 features and the last should return 2 logits."],
        "quiz.questions.tests.q41_build_mlp_sequential",
        symbol_name="build_mlp_sequential",
        hidden_binary_key="q41_build_mlp_sequential",
    ),
    question(
        "q42_make_classification_loader",
        "07",
        "Make A Classification DataLoader",
        "07_basic_nn_mlp_with_nn_module",
        "Write `make_classification_loader(x, y)` that returns `DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)`.",
        """from torch.utils.data import DataLoader, TensorDataset


def make_classification_loader(x, y):
    # Return a shuffled DataLoader with batch_size=64.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The loader should keep each input paired with its label.", "Use `batch_size=64` and `shuffle=True`."],
        "quiz.questions.tests.q42_make_classification_loader",
        symbol_name="make_classification_loader",
        hidden_binary_key="q42_make_classification_loader",
    ),
    question(
        "q43_compute_multiclass_accuracy",
        "07",
        "Compute Multiclass Accuracy",
        "07_basic_nn_mlp_with_nn_module",
        "Write `compute_multiclass_accuracy(logits, targets)` that returns accuracy as a Python float.",
        """import torch


def compute_multiclass_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # Convert logits to class ids and return mean accuracy.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `argmax(dim=1)` to get predicted classes.", "Return a Python float such as `0.875`, not a tensor."],
        "quiz.questions.tests.q43_compute_multiclass_accuracy",
        symbol_name="compute_multiclass_accuracy",
        hidden_binary_key="q43_compute_multiclass_accuracy",
    ),
    question(
        "q44_run_mlp_training_epoch",
        "07",
        "Run One MLP Training Epoch",
        "07_basic_nn_mlp_with_nn_module",
        "Write `run_mlp_training_epoch(model, loader, optimizer)` that loops over the loader, runs the normal CrossEntropyLoss training step, and returns the average loss as a Python float.",
        """from torch import nn


def run_mlp_training_epoch(model, loader, optimizer) -> float:
    # Train for one epoch and return the mean batch loss.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Call `model.train()` before the loop.", "Use `nn.CrossEntropyLoss()` and return the average batch loss as a float."],
        "quiz.questions.tests.q44_run_mlp_training_epoch",
        symbol_name="run_mlp_training_epoch",
    ),
    question(
        "q45_train_module_mlp",
        "07",
        "Train A Module MLP",
        "07_basic_nn_mlp_with_nn_module",
        "Write `train_module_mlp(seed=42)` that sets the seed, makes XOR-like data, builds the `nn.Module` MLP, trains it for multiple epochs, and returns the final validation accuracy as a Python float.",
        """import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_module_mlp(seed: int = 42) -> float:
    # Train the module-based MLP end to end and return the validation accuracy.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `torch.manual_seed(seed)` so repeated runs match.", "Return the final validation accuracy as a float."],
        "quiz.questions.tests.q45_train_module_mlp",
        symbol_name="train_module_mlp",
    ),
    question(
        "q46_init_manual_cnn_params",
        "15",
        "Initialize Manual CNN Parameters",
        "15_manual_cnn_from_scratch",
        "Write `init_manual_cnn_params(seed=123)` that sets the seed and returns the eight trainable tensors used in the manual CNN chapter: two conv weights, two conv biases, two linear weights, and two linear biases.",
        """import torch


def init_manual_cnn_params(seed: int = 123):
    # Return the eight trainable tensors for the manual CNN.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Include conv weights for `(8, 1, 3, 3)` and `(16, 8, 3, 3)`.", "All returned tensors should require gradients."],
        "quiz.questions.tests.q46_init_manual_cnn_params",
        symbol_name="init_manual_cnn_params",
        hidden_binary_key="q46_init_manual_cnn_params",
    ),
    question(
        "q47_manual_conv_relu_pool",
        "15",
        "Manual Conv ReLU Pool Block",
        "15_manual_cnn_from_scratch",
        "Write `manual_conv_relu_pool(images, conv_weight, conv_bias)` that applies `F.conv2d(..., padding=1)`, ReLU, and `F.max_pool2d(..., kernel_size=2)`.",
        """import torch
import torch.nn.functional as F


def manual_conv_relu_pool(
    images: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor
) -> torch.Tensor:
    # Apply conv -> ReLU -> max pool and return the result.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If images are `(batch, 1, 16, 16)` and the conv has 8 filters, the output should be `(batch, 8, 8, 8)`.", "Use padding so the convolution keeps the 16x16 spatial size before pooling."],
        "quiz.questions.tests.q47_manual_conv_relu_pool",
        symbol_name="manual_conv_relu_pool",
        hidden_binary_key="q47_manual_conv_relu_pool",
    ),
    question(
        "q48_flatten_image_features",
        "15",
        "Flatten CNN Features",
        "15_manual_cnn_from_scratch",
        "Write `flatten_image_features(x)` that flattens everything except the batch dimension.",
        """import torch


def flatten_image_features(x: torch.Tensor) -> torch.Tensor:
    # Return shape (batch, features).
    raise NotImplementedError("Replace this line with your code.")
""",
        ["A tensor with shape `(4, 16, 4, 4)` should become `(4, 256)`.", "Keep the batch dimension intact."],
        "quiz.questions.tests.q48_flatten_image_features",
        symbol_name="flatten_image_features",
        hidden_binary_key="q48_flatten_image_features",
    ),
    question(
        "q49_manual_cnn_classifier",
        "15",
        "Manual CNN Classifier Head",
        "15_manual_cnn_from_scratch",
        "Write `manual_cnn_classifier(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias)` that applies a hidden ReLU layer and returns class logits.",
        """import torch


def manual_cnn_classifier(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> torch.Tensor:
    # Return the final class logits.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `torch.relu(x @ fc1_weight + fc1_bias)` for the hidden layer.", "If `x` is `(batch, 256)`, the output logits should be `(batch, 3)`."],
        "quiz.questions.tests.q49_manual_cnn_classifier",
        symbol_name="manual_cnn_classifier",
        hidden_binary_key="q49_manual_cnn_classifier",
    ),
    question(
        "q50_predict_image_classes",
        "15",
        "Predict Image Classes",
        "15_manual_cnn_from_scratch",
        "Write `predict_image_classes(logits)` that returns the winning class index for each image.",
        """import torch


def predict_image_classes(logits: torch.Tensor) -> torch.Tensor:
    # Return one predicted class id per image.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `argmax(dim=1)` on the logits.", "The output should have shape `(batch,)`."],
        "quiz.questions.tests.q50_predict_image_classes",
        symbol_name="predict_image_classes",
        hidden_binary_key="q50_predict_image_classes",
    ),
    question(
        "q51_build_cnn_feature_extractor",
        "08",
        "Build A CNN Feature Extractor",
        "08_cnn_basics",
        "Write `build_cnn_feature_extractor()` that returns the convolutional `nn.Sequential` block from the lesson.",
        """from torch import nn


def build_cnn_feature_extractor():
    # Return the feature extractor with conv, ReLU, and pooling layers.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The first convolution should be `Conv2d(1, 8, kernel_size=3, padding=1)`.", "The block should include two max-pooling layers."],
        "quiz.questions.tests.q51_build_cnn_feature_extractor",
        symbol_name="build_cnn_feature_extractor",
        hidden_binary_key="q51_build_cnn_feature_extractor",
    ),
    question(
        "q52_conv_output_size",
        "08",
        "Compute Convolution Output Size",
        "08_cnn_basics",
        "Write `conv_output_size(input_size, kernel_size, padding=0, stride=1)` using the standard convolution size formula.",
        """def conv_output_size(
    input_size: int, kernel_size: int, padding: int = 0, stride: int = 1
) -> int:
    # Return the integer output size for one spatial dimension.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["With input 28, kernel 3, padding 1, stride 1, the output should be 28.", "Use floor-style integer math."],
        "quiz.questions.tests.q52_conv_output_size",
        symbol_name="conv_output_size",
        hidden_binary_key="q52_conv_output_size",
    ),
    question(
        "q53_build_cnn_classifier_head",
        "08",
        "Build A CNN Classifier Head",
        "08_cnn_basics",
        "Write `build_cnn_classifier_head()` that returns `nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 32), nn.ReLU(), nn.Linear(32, 3))`.",
        """from torch import nn


def build_cnn_classifier_head():
    # Return the classifier head used after the feature extractor.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The first layer should flatten the feature maps.", "The final linear layer should output 3 class logits."],
        "quiz.questions.tests.q53_build_cnn_classifier_head",
        symbol_name="build_cnn_classifier_head",
        hidden_binary_key="q53_build_cnn_classifier_head",
    ),
    question(
        "q54_flatten_cnn_batch",
        "08",
        "Flatten A CNN Batch",
        "08_cnn_basics",
        "Write `flatten_cnn_batch(x)` that flattens a tensor from `(batch, channels, height, width)` into `(batch, features)`.",
        """import torch


def flatten_cnn_batch(x: torch.Tensor) -> torch.Tensor:
    # Flatten everything except the batch dimension.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["A `(4, 16, 7, 7)` tensor should become `(4, 784)`.", "Keep the first dimension as the batch size."],
        "quiz.questions.tests.q54_flatten_cnn_batch",
        symbol_name="flatten_cnn_batch",
        hidden_binary_key="q54_flatten_cnn_batch",
    ),
    question(
        "q55_predict_cnn_classes",
        "08",
        "Predict CNN Classes",
        "08_cnn_basics",
        "Write `predict_cnn_classes(logits)` that returns the predicted class id for each row of logits.",
        """import torch


def predict_cnn_classes(logits: torch.Tensor) -> torch.Tensor:
    # Return one class prediction per example.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `argmax(dim=1)`.", "The output should be shape `(batch,)`."],
        "quiz.questions.tests.q55_predict_cnn_classes",
        symbol_name="predict_cnn_classes",
        hidden_binary_key="q55_predict_cnn_classes",
    ),
    question(
        "q56_add_channel_dim",
        "09",
        "Add The Channel Dimension",
        "09_vision_classifier",
        "Write `add_channel_dim(image)` that turns a 2D image into shape `(1, H, W)`.",
        """import torch


def add_channel_dim(image: torch.Tensor) -> torch.Tensor:
    # Add the single-channel dimension expected by CNNs.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["A `(16, 16)` image should become `(1, 16, 16)`.", "Use `unsqueeze`, not a Python loop."],
        "quiz.questions.tests.q56_add_channel_dim",
        symbol_name="add_channel_dim",
        hidden_binary_key="q56_add_channel_dim",
    ),
    question(
        "q57_build_pattern_loader",
        "09",
        "Build A Pattern DataLoader",
        "09_vision_classifier",
        "Write `build_pattern_loader(images, labels)` that returns `DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=True)`.",
        """from torch.utils.data import DataLoader, TensorDataset


def build_pattern_loader(images, labels):
    # Return a shuffled DataLoader for the synthetic vision dataset.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Keep the images paired with their labels.", "Use `batch_size=64` and `shuffle=True`."],
        "quiz.questions.tests.q57_build_pattern_loader",
        symbol_name="build_pattern_loader",
        hidden_binary_key="q57_build_pattern_loader",
    ),
    question(
        "q58_compute_loader_accuracy",
        "09",
        "Compute Accuracy Over A Loader",
        "09_vision_classifier",
        "Write `compute_loader_accuracy(model, loader)` that switches the model to eval mode, uses `torch.no_grad()`, and returns accuracy as a Python float.",
        """import torch


def compute_loader_accuracy(model, loader) -> float:
    # Return the fraction of correct predictions across the full loader.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `logits.argmax(dim=1)` to get class predictions.", "Return a Python float between 0.0 and 1.0."],
        "quiz.questions.tests.q58_compute_loader_accuracy",
        symbol_name="compute_loader_accuracy",
    ),
    question(
        "q59_build_vision_cnn",
        "09",
        "Build The Vision CNN",
        "09_vision_classifier",
        "Write `build_vision_cnn()` that returns the lesson's `nn.Sequential` network with two conv blocks followed by `Flatten`, `Linear(16 * 4 * 4, 32)`, `ReLU`, and `Linear(32, 3)`.",
        """from torch import nn


def build_vision_cnn():
    # Return the CNN used for the synthetic image patterns.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The input channel count is 1.", "The final layer should return 3 logits."],
        "quiz.questions.tests.q59_build_vision_cnn",
        symbol_name="build_vision_cnn",
        hidden_binary_key="q59_build_vision_cnn",
    ),
    question(
        "q60_generate_diagonal_pattern",
        "09",
        "Generate A Diagonal Pattern",
        "09_vision_classifier",
        "Write `generate_diagonal_pattern(image_size)` that returns a `(1, image_size, image_size)` tensor with ones on the main diagonal and zeros elsewhere.",
        """import torch


def generate_diagonal_pattern(image_size: int) -> torch.Tensor:
    # Return a single-channel diagonal image pattern.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The diagonal entries should be 1.0.", "The output should include the channel dimension."],
        "quiz.questions.tests.q60_generate_diagonal_pattern",
        symbol_name="generate_diagonal_pattern",
        hidden_binary_key="q60_generate_diagonal_pattern",
    ),
    question(
        "q61_embedding_lookup",
        "16",
        "Manual Embedding Lookup",
        "16_manual_rnn_from_scratch",
        "Write `embedding_lookup(embedding_table, tokens)` that performs the embedding-table indexing step.",
        """import torch


def embedding_lookup(
    embedding_table: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    # Return the embedded token vectors.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `tokens` is `(batch, seq_len)`, the result should be `(batch, seq_len, embed_dim)`.", "Use tensor indexing with the token ids."],
        "quiz.questions.tests.q61_embedding_lookup",
        symbol_name="embedding_lookup",
        hidden_binary_key="q61_embedding_lookup",
    ),
    question(
        "q62_manual_rnn_step",
        "16",
        "Manual RNN Step",
        "16_manual_rnn_from_scratch",
        "Write `manual_rnn_step(x_t, hidden, wxh, whh, bh)` that computes the next hidden state with a tanh update.",
        """import torch


def manual_rnn_step(
    x_t: torch.Tensor,
    hidden: torch.Tensor,
    wxh: torch.Tensor,
    whh: torch.Tensor,
    bh: torch.Tensor,
) -> torch.Tensor:
    # Return the next hidden state.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `torch.tanh(...)` on the affine update.", "The output shape should match the current hidden state shape."],
        "quiz.questions.tests.q62_manual_rnn_step",
        symbol_name="manual_rnn_step",
        hidden_binary_key="q62_manual_rnn_step",
    ),
    question(
        "q63_run_manual_rnn",
        "16",
        "Run The Manual RNN Loop",
        "16_manual_rnn_from_scratch",
        "Write `run_manual_rnn(embedded, wxh, whh, bh)` that starts hidden state at zeros and loops through time to return the final hidden state.",
        """import torch


def run_manual_rnn(
    embedded: torch.Tensor,
    wxh: torch.Tensor,
    whh: torch.Tensor,
    bh: torch.Tensor,
) -> torch.Tensor:
    # Return the final hidden state after looping through all time steps.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["`embedded` has shape `(batch, seq_len, embed_dim)`.", "Return shape `(batch, hidden_size)`."],
        "quiz.questions.tests.q63_run_manual_rnn",
        symbol_name="run_manual_rnn",
        hidden_binary_key="q63_run_manual_rnn",
    ),
    question(
        "q64_manual_sequence_logits",
        "16",
        "Manual Sequence Logits",
        "16_manual_rnn_from_scratch",
        "Write `manual_sequence_logits(hidden, why, by)` that maps the final hidden state to class logits.",
        """import torch


def manual_sequence_logits(
    hidden: torch.Tensor, why: torch.Tensor, by: torch.Tensor
) -> torch.Tensor:
    # Return the class logits for the sequence batch.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `hidden` is `(batch, hidden_size)` and `why` is `(hidden_size, 2)`, the output should be `(batch, 2)`.", "Use matrix multiplication plus bias."],
        "quiz.questions.tests.q64_manual_sequence_logits",
        symbol_name="manual_sequence_logits",
        hidden_binary_key="q64_manual_sequence_logits",
    ),
    question(
        "q65_predict_manual_rnn_classes",
        "16",
        "Predict Manual RNN Classes",
        "16_manual_rnn_from_scratch",
        "Write `predict_manual_rnn_classes(logits)` that returns the winning class id for each sequence.",
        """import torch


def predict_manual_rnn_classes(logits: torch.Tensor) -> torch.Tensor:
    # Return one predicted class id per sequence.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `argmax(dim=1)`.", "The result should be shape `(batch,)`."],
        "quiz.questions.tests.q65_predict_manual_rnn_classes",
        symbol_name="predict_manual_rnn_classes",
        hidden_binary_key="q65_predict_manual_rnn_classes",
    ),
    question(
        "q66_build_vanilla_rnn",
        "14",
        "Build A Vanilla RNN Layer",
        "14_basic_rnn_sequence_classifier",
        "Write `build_vanilla_rnn()` that returns `nn.RNN(input_size=32, hidden_size=32, batch_first=True, nonlinearity='tanh')`.",
        """from torch import nn


def build_vanilla_rnn():
    # Return the RNN layer used in the lesson.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `batch_first=True`.", "The hidden size should be 32."],
        "quiz.questions.tests.q66_build_vanilla_rnn",
        symbol_name="build_vanilla_rnn",
        hidden_binary_key="q66_build_vanilla_rnn",
    ),
    question(
        "q67_take_last_hidden",
        "14",
        "Take The Final Hidden State",
        "14_basic_rnn_sequence_classifier",
        "Write `take_last_hidden(hidden_state)` that returns the final layer's hidden state for classification.",
        """import torch


def take_last_hidden(hidden_state: torch.Tensor) -> torch.Tensor:
    # Return the final hidden state for the last recurrent layer.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `hidden_state` is `(num_layers, batch, hidden_size)`, the result should be `(batch, hidden_size)`.", "Use the last recurrent layer."],
        "quiz.questions.tests.q67_take_last_hidden",
        symbol_name="take_last_hidden",
        hidden_binary_key="q67_take_last_hidden",
    ),
    question(
        "q68_make_rnn_labels",
        "14",
        "Make RNN Sequence Labels",
        "14_basic_rnn_sequence_classifier",
        "Write `make_rnn_labels(tokens)` that labels a sequence as 1 when the later half has a larger average token value than the earlier half.",
        """import torch


def make_rnn_labels(tokens: torch.Tensor) -> torch.Tensor:
    # Return integer labels of shape (batch,).
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Compare the mean of the first half with the mean of the second half.", "Return integer labels with dtype `torch.long`."],
        "quiz.questions.tests.q68_make_rnn_labels",
        symbol_name="make_rnn_labels",
        hidden_binary_key="q68_make_rnn_labels",
    ),
    question(
        "q69_build_rnn_classifier_head",
        "14",
        "Build The RNN Classifier Head",
        "14_basic_rnn_sequence_classifier",
        "Write `build_rnn_classifier_head()` that returns `nn.Linear(32, 2)`.",
        """from torch import nn


def build_rnn_classifier_head():
    # Return the final linear classifier for the sequence summary.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The head should read 32 hidden features.", "The head should output 2 logits."],
        "quiz.questions.tests.q69_build_rnn_classifier_head",
        symbol_name="build_rnn_classifier_head",
        hidden_binary_key="q69_build_rnn_classifier_head",
    ),
    question(
        "q70_predict_rnn_classes",
        "14",
        "Predict RNN Classes",
        "14_basic_rnn_sequence_classifier",
        "Write `predict_rnn_classes(logits)` that returns the predicted class index for each sequence.",
        """import torch


def predict_rnn_classes(logits: torch.Tensor) -> torch.Tensor:
    # Return one predicted class id per sequence.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `argmax(dim=1)` on the logits.", "The result should be shape `(batch,)`."],
        "quiz.questions.tests.q70_predict_rnn_classes",
        symbol_name="predict_rnn_classes",
        hidden_binary_key="q70_predict_rnn_classes",
    ),
    question(
        "q71_build_embedding_layer",
        "10",
        "Build An Embedding Layer",
        "10_lstm_sequence_classifier",
        "Write `build_embedding_layer()` that returns `nn.Embedding(15, 32)`.",
        """from torch import nn


def build_embedding_layer():
    # Return the embedding layer used in the LSTM chapter.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The vocabulary size should be 15.", "Each token should map to a 32-dimensional vector."],
        "quiz.questions.tests.q71_build_embedding_layer",
        symbol_name="build_embedding_layer",
        hidden_binary_key="q71_build_embedding_layer",
    ),
    question(
        "q72_build_lstm_layer",
        "10",
        "Build An LSTM Layer",
        "10_lstm_sequence_classifier",
        "Write `build_lstm_layer()` that returns `nn.LSTM(input_size=32, hidden_size=32, batch_first=True)`.",
        """from torch import nn


def build_lstm_layer():
    # Return the LSTM layer used in the lesson.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `batch_first=True`.", "The input and hidden size should both be 32."],
        "quiz.questions.tests.q72_build_lstm_layer",
        symbol_name="build_lstm_layer",
        hidden_binary_key="q72_build_lstm_layer",
    ),
    question(
        "q73_take_lstm_final_hidden",
        "10",
        "Take The LSTM Final Hidden State",
        "10_lstm_sequence_classifier",
        "Write `take_lstm_final_hidden(hidden_state)` that returns the last layer's hidden state for the batch.",
        """import torch


def take_lstm_final_hidden(hidden_state: torch.Tensor) -> torch.Tensor:
    # Return shape (batch, hidden_size).
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `hidden_state` is `(num_layers, batch, hidden_size)`, the result should be `(batch, hidden_size)`.", "Take the last recurrent layer."],
        "quiz.questions.tests.q73_take_lstm_final_hidden",
        symbol_name="take_lstm_final_hidden",
        hidden_binary_key="q73_take_lstm_final_hidden",
    ),
    question(
        "q74_build_lstm_classifier_head",
        "10",
        "Build The LSTM Classifier Head",
        "10_lstm_sequence_classifier",
        "Write `build_lstm_classifier_head()` that returns `nn.Linear(32, 2)`.",
        """from torch import nn


def build_lstm_classifier_head():
    # Return the final linear layer for the LSTM summary.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The head should read 32 features and output 2 logits.", "Return an `nn.Linear` module."],
        "quiz.questions.tests.q74_build_lstm_classifier_head",
        symbol_name="build_lstm_classifier_head",
        hidden_binary_key="q74_build_lstm_classifier_head",
    ),
    question(
        "q75_predict_lstm_classes",
        "10",
        "Predict LSTM Classes",
        "10_lstm_sequence_classifier",
        "Write `predict_lstm_classes(logits)` that returns the predicted class index for each sequence.",
        """import torch


def predict_lstm_classes(logits: torch.Tensor) -> torch.Tensor:
    # Return one class id per row of logits.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `argmax(dim=1)`.", "The result should be shape `(batch,)`."],
        "quiz.questions.tests.q75_predict_lstm_classes",
        symbol_name="predict_lstm_classes",
        hidden_binary_key="q75_predict_lstm_classes",
    ),
    question(
        "q76_positional_encoding",
        "17",
        "Manual Positional Encoding",
        "17_manual_transformer_from_scratch",
        "Write `positional_encoding(seq_len, d_model)` using the sine/cosine construction from the lesson.",
        """import math
import torch


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # Return a tensor of shape (seq_len, d_model).
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The output should have shape `(seq_len, d_model)`.", "Even columns use sine and odd columns use cosine."],
        "quiz.questions.tests.q76_positional_encoding",
        symbol_name="positional_encoding",
        hidden_binary_key="q76_positional_encoding",
    ),
    question(
        "q77_scaled_attention_scores",
        "17",
        "Compute Scaled Attention Scores",
        "17_manual_transformer_from_scratch",
        "Write `scaled_attention_scores(q, k)` that returns `q @ k.transpose(1, 2) / sqrt(d_model)`.",
        """import math
import torch


def scaled_attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # Return the scaled attention score matrix.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `q` and `k` are `(batch, seq_len, d_model)`, the result should be `(batch, seq_len, seq_len)`.", "Scale by the square root of the last dimension."],
        "quiz.questions.tests.q77_scaled_attention_scores",
        symbol_name="scaled_attention_scores",
        hidden_binary_key="q77_scaled_attention_scores",
    ),
    question(
        "q78_attention_weighted_values",
        "17",
        "Apply Attention Weights",
        "17_manual_transformer_from_scratch",
        "Write `attention_weighted_values(weights, v)` that returns the weighted sum over values.",
        """import torch


def attention_weighted_values(
    weights: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # Return the attended value vectors.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `weights` is `(batch, seq_len, seq_len)` and `v` is `(batch, seq_len, d_model)`, the result should be `(batch, seq_len, d_model)`.", "Use matrix multiplication."],
        "quiz.questions.tests.q78_attention_weighted_values",
        symbol_name="attention_weighted_values",
        hidden_binary_key="q78_attention_weighted_values",
    ),
    question(
        "q79_mean_pool_sequence",
        "17",
        "Mean Pool A Sequence",
        "17_manual_transformer_from_scratch",
        "Write `mean_pool_sequence(x)` that averages over the sequence dimension.",
        """import torch


def mean_pool_sequence(x: torch.Tensor) -> torch.Tensor:
    # Return one pooled vector per sequence.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `x` is `(batch, seq_len, d_model)`, the output should be `(batch, d_model)`.", "Average across `dim=1`."],
        "quiz.questions.tests.q79_mean_pool_sequence",
        symbol_name="mean_pool_sequence",
        hidden_binary_key="q79_mean_pool_sequence",
    ),
    question(
        "q80_manual_transformer_logits",
        "17",
        "Manual Transformer Logits",
        "17_manual_transformer_from_scratch",
        "Write `manual_transformer_logits(pooled, w_cls, b_cls)` that returns the final class logits from the pooled sequence vector.",
        """import torch


def manual_transformer_logits(
    pooled: torch.Tensor, w_cls: torch.Tensor, b_cls: torch.Tensor
) -> torch.Tensor:
    # Return the final class logits.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `pooled` is `(batch, d_model)` and `w_cls` is `(d_model, 2)`, the output should be `(batch, 2)`.", "Use matrix multiplication plus bias."],
        "quiz.questions.tests.q80_manual_transformer_logits",
        symbol_name="manual_transformer_logits",
        hidden_binary_key="q80_manual_transformer_logits",
    ),
    question(
        "q81_pool_encoder_output",
        "11",
        "Pool Transformer Encoder Output",
        "11_transformer_basics",
        "Write `pool_encoder_output(x)` that performs mean pooling across the sequence length.",
        """import torch


def pool_encoder_output(x: torch.Tensor) -> torch.Tensor:
    # Return one pooled vector per sequence.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["If `x` is `(batch, seq_len, 32)`, the result should be `(batch, 32)`.", "Average across `dim=1`."],
        "quiz.questions.tests.q81_pool_encoder_output",
        symbol_name="pool_encoder_output",
        hidden_binary_key="q81_pool_encoder_output",
    ),
    question(
        "q82_build_transformer_encoder_layer",
        "11",
        "Build A Transformer Encoder Layer",
        "11_transformer_basics",
        "Write `build_transformer_encoder_layer()` that returns `nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)`.",
        """from torch import nn


def build_transformer_encoder_layer():
    # Return the encoder layer used in the lesson.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `d_model=32` and `nhead=4`.", "Set `batch_first=True`."],
        "quiz.questions.tests.q82_build_transformer_encoder_layer",
        symbol_name="build_transformer_encoder_layer",
        hidden_binary_key="q82_build_transformer_encoder_layer",
    ),
    question(
        "q83_build_transformer_classifier_head",
        "11",
        "Build The Transformer Classifier Head",
        "11_transformer_basics",
        "Write `build_transformer_classifier_head()` that returns `nn.Linear(32, 2)`.",
        """from torch import nn


def build_transformer_classifier_head():
    # Return the final classification layer for pooled sequence vectors.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The head should read 32 features and output 2 logits.", "Return an `nn.Linear` module."],
        "quiz.questions.tests.q83_build_transformer_classifier_head",
        symbol_name="build_transformer_classifier_head",
        hidden_binary_key="q83_build_transformer_classifier_head",
    ),
    question(
        "q84_make_transformer_labels",
        "11",
        "Make Transformer Sequence Labels",
        "11_transformer_basics",
        "Write `make_transformer_labels(tokens, vocab_size=20)` that labels a sequence as 1 when its token sum is larger than `(seq_len * vocab_size / 2)`.",
        """import torch


def make_transformer_labels(
    tokens: torch.Tensor, vocab_size: int = 20
) -> torch.Tensor:
    # Return integer labels of shape (batch,).
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Return dtype `torch.long`.", "Use the same sum-based rule as the lesson data generator."],
        "quiz.questions.tests.q84_make_transformer_labels",
        symbol_name="make_transformer_labels",
        hidden_binary_key="q84_make_transformer_labels",
    ),
    question(
        "q85_predict_transformer_classes",
        "11",
        "Predict Transformer Classes",
        "11_transformer_basics",
        "Write `predict_transformer_classes(logits)` that returns the winning class index for each sequence.",
        """import torch


def predict_transformer_classes(logits: torch.Tensor) -> torch.Tensor:
    # Return one predicted class id per row of logits.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `argmax(dim=1)`.", "The output should be shape `(batch,)`."],
        "quiz.questions.tests.q85_predict_transformer_classes",
        symbol_name="predict_transformer_classes",
        hidden_binary_key="q85_predict_transformer_classes",
    ),
    question(
        "q86_build_generator",
        "12",
        "Build The Generator",
        "12_toy_gan",
        "Write `build_generator()` that returns the generator `nn.Sequential` from the lesson.",
        """from torch import nn


def build_generator():
    # Return the generator network that maps 3D noise to 1D fake samples.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The network should start with `nn.Linear(3, 16)`.", "The final layer should return one scalar output per sample."],
        "quiz.questions.tests.q86_build_generator",
        symbol_name="build_generator",
        hidden_binary_key="q86_build_generator",
    ),
    question(
        "q87_build_discriminator",
        "12",
        "Build The Discriminator",
        "12_toy_gan",
        "Write `build_discriminator()` that returns the discriminator `nn.Sequential` from the lesson.",
        """from torch import nn


def build_discriminator():
    # Return the discriminator network that maps 1D samples to 1 logit.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The network should start with `nn.Linear(1, 16)`.", "The final layer should return one logit per sample."],
        "quiz.questions.tests.q87_build_discriminator",
        symbol_name="build_discriminator",
        hidden_binary_key="q87_build_discriminator",
    ),
    question(
        "q88_sample_gan_noise",
        "12",
        "Sample GAN Noise",
        "12_toy_gan",
        "Write `sample_gan_noise(batch_size)` that returns `torch.randn(batch_size, 3)`.",
        """import torch


def sample_gan_noise(batch_size: int) -> torch.Tensor:
    # Return one 3D noise vector per sample.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["For batch size 5, the output should have shape `(5, 3)`.", "Sample from a standard normal distribution."],
        "quiz.questions.tests.q88_sample_gan_noise",
        symbol_name="sample_gan_noise",
        hidden_binary_key="q88_sample_gan_noise",
    ),
    question(
        "q89_make_real_and_fake_targets",
        "12",
        "Make Real And Fake Targets",
        "12_toy_gan",
        "Write `make_real_and_fake_targets(logits)` that returns `(torch.ones_like(logits), torch.zeros_like(logits))`.",
        """import torch


def make_real_and_fake_targets(logits: torch.Tensor):
    # Return targets for real samples and fake samples.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The first returned tensor should be all ones.", "The second returned tensor should be all zeros and both should match the logits shape."],
        "quiz.questions.tests.q89_make_real_and_fake_targets",
        symbol_name="make_real_and_fake_targets",
        hidden_binary_key="q89_make_real_and_fake_targets",
    ),
    question(
        "q90_detach_fake_batch",
        "12",
        "Detach Fake Samples",
        "12_toy_gan",
        "Write `detach_fake_batch(fake_samples)` that returns a detached copy for the discriminator step.",
        """import torch


def detach_fake_batch(fake_samples: torch.Tensor) -> torch.Tensor:
    # Return fake samples without generator gradient tracking.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["The returned tensor should have the same values as the input.", "The returned tensor should not require gradients."],
        "quiz.questions.tests.q90_detach_fake_batch",
        symbol_name="detach_fake_batch",
        hidden_binary_key="q90_detach_fake_batch",
    ),
    question(
        "q91_get_device",
        "13",
        "Choose The Best Device",
        "13_device_cpu_to_mps",
        "Write `get_device()` that prefers CUDA when available, then MPS, otherwise CPU.",
        """import torch


def get_device() -> torch.device:
    # Prefer CUDA, then MPS, then CPU.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Return a `torch.device` object.", "Check CUDA first, then MPS, then CPU."],
        "quiz.questions.tests.q91_get_device",
        symbol_name="get_device",
        hidden_binary_key="q91_get_device",
    ),
    question(
        "q92_move_module_to_device",
        "13",
        "Move A Module To Device",
        "13_device_cpu_to_mps",
        "Write `move_module_to_device(model, device)` that moves the model to the given device and returns it.",
        """def move_module_to_device(model, device):
    # Move the module to the requested device and return it.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `model.to(device)`.", "Return the moved module."],
        "quiz.questions.tests.q92_move_module_to_device",
        symbol_name="move_module_to_device",
    ),
    question(
        "q93_make_device_inputs",
        "13",
        "Create Inputs On A Device",
        "13_device_cpu_to_mps",
        "Write `make_device_inputs(batch_size, device)` that returns `x` with shape `(batch_size, 4)` and integer labels `y` with shape `(batch_size,)`, both created on the given device.",
        """import torch


def make_device_inputs(batch_size: int, device):
    # Return x and y created directly on the given device.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["`x` should be random floats with shape `(batch_size, 4)`.", "`y` should be integer class labels with shape `(batch_size,)` on the same device."],
        "quiz.questions.tests.q93_make_device_inputs",
        symbol_name="make_device_inputs",
        hidden_binary_key="q93_make_device_inputs",
    ),
    question(
        "q94_move_tensor_pair_to_device",
        "13",
        "Move Two Tensors To Device",
        "13_device_cpu_to_mps",
        "Write `move_tensor_pair_to_device(x, y, device)` that returns `(x.to(device), y.to(device))`.",
        """def move_tensor_pair_to_device(x, y, device):
    # Move both tensors to the requested device and return them.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Both returned tensors should live on the requested device.", "Keep the same shapes and values."],
        "quiz.questions.tests.q94_move_tensor_pair_to_device",
        symbol_name="move_tensor_pair_to_device",
        hidden_binary_key="q94_move_tensor_pair_to_device",
    ),
    question(
        "q95_same_device",
        "13",
        "Check Whether Model And Input Match Devices",
        "13_device_cpu_to_mps",
        "Write `same_device(model, x)` that returns `True` when the model parameters and input tensor are on the same device.",
        """def same_device(model, x) -> bool:
    # Return True when the model and tensor live on the same device.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use one of the model parameters to inspect the model device.", "Return a Python bool."],
        "quiz.questions.tests.q95_same_device",
        symbol_name="same_device",
    ),
    question(
        "q96_train_full_linear_regression",
        "03",
        "Train A Full Linear Regression Model",
        "03_linear_regression_with_nn_module",
        "Write `train_full_linear_regression(seed=42)` that sets the seed, builds `nn.Linear(1, 1)`, trains it on synthetic data `y = 3x + 1` with `nn.MSELoss()` and SGD, and returns `(weight, bias, loss)` where each item is detached from the graph.",
        """import torch
from torch import nn


def train_full_linear_regression(seed: int = 42):
    # Train the full linear model and return (weight, bias, final_loss).
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `torch.manual_seed(seed)` so repeated runs match.", "Return the learned weight, bias, and final loss after training."],
        "quiz.questions.tests.q96_train_full_linear_regression",
        symbol_name="train_full_linear_regression",
    ),
    question(
        "q97_train_full_logistic_regression",
        "05",
        "Train A Full Logistic Regression Model",
        "05_logistic_regression_with_nn_module",
        "Write `train_full_logistic_regression(seed=42)` that sets the seed, builds `nn.Linear(2, 1)`, trains it with `nn.BCEWithLogitsLoss()` on a simple binary classification dataset, and returns the final accuracy as a Python float.",
        """import torch
from torch import nn


def train_full_logistic_regression(seed: int = 42) -> float:
    # Train the full logistic regression model and return the final accuracy.
    raise NotImplementedError("Replace this line with your code.")
""",
        ["Use `torch.manual_seed(seed)` so repeated runs match.", "Return the final accuracy as a float after training."],
        "quiz.questions.tests.q97_train_full_logistic_regression",
        symbol_name="train_full_logistic_regression",
    ),
]
