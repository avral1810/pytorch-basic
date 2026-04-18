from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from quiz.content.data import get_chapter
from quiz.questions.answer_keys import ANSWER_CODES
from quiz.questions.extra_questions import EXTRA_QUESTIONS
from quiz.questions.models import QuestionAnswer, QuestionTemplate, QuestionTests, QuestionText, Quiz, QuizQuestion

ROOT_DIR = Path(__file__).resolve().parents[2]
PLAYGROUND_STARTER = """import torch
from torch import nn


# Try small tensor experiments here.
# Example:
# x = torch.tensor([1.0, 2.0, 3.0])
# print(x.shape)
"""


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


QUESTIONS = [
    question("q30_create_tensor_from_list", "00", "Create A Tensor From A Python List", "00_tensors_and_autograd", "Write `make_tensor(values)` that returns `torch.tensor(values)`.", """import torch


def make_tensor(values) -> torch.Tensor:
    # Convert the Python values into a PyTorch tensor.
    raise NotImplementedError("Replace this line with your code.")
""", ["Input [1, 2, 3] should become a tensor with shape (3,).", "The values should stay in the same order."], "quiz.questions.tests.q30_create_tensor_from_list", symbol_name="make_tensor", hidden_binary_key="q30_create_tensor_from_list"),
    question("q31_tensor_dtype", "00", "Create A Float32 Tensor", "00_tensors_and_autograd", "Write `make_float_tensor(values)` that returns a tensor with dtype `torch.float32`.", """import torch


def make_float_tensor(values) -> torch.Tensor:
    # Return a tensor that stores the values as float32.
    raise NotImplementedError("Replace this line with your code.")
""", ["The tensor dtype should be torch.float32.", "Input [1, 2, 3] should become floating point values."], "quiz.questions.tests.q31_tensor_dtype", symbol_name="make_float_tensor", hidden_binary_key="q31_tensor_dtype"),
    question("q32_make_zeros_tensor", "00", "Make A Zeros Tensor", "00_tensors_and_autograd", "Write `make_zeros(rows, cols)` that returns a tensor of zeros with shape `(rows, cols)`.", """import torch


def make_zeros(rows: int, cols: int) -> torch.Tensor:
    # Return a 2D tensor full of zeros.
    raise NotImplementedError("Replace this line with your code.")
""", ["Input (2, 3) should return shape (2, 3).", "Every value in the tensor should be zero."], "quiz.questions.tests.q32_make_zeros_tensor", symbol_name="make_zeros", hidden_binary_key="q32_make_zeros_tensor"),
    question("q33_slice_first_row", "00", "Slice The First Row", "00_tensors_and_autograd", "Write `first_row(x)` that returns the first row of a 2D tensor using slicing.", """import torch


def first_row(x: torch.Tensor) -> torch.Tensor:
    # Return the first row of x.
    raise NotImplementedError("Replace this line with your code.")
""", ["If x has shape (2, 3), the result should have shape (3,).", "The returned values should be the first row of the input."], "quiz.questions.tests.q33_slice_first_row", symbol_name="first_row", hidden_binary_key="q33_slice_first_row"),
    question("q34_tensor_num_dims", "00", "Find Number Of Dimensions", "00_tensors_and_autograd", "Write `num_dims(x)` that returns how many dimensions the tensor has.", """import torch


def num_dims(x: torch.Tensor) -> int:
    # Return the number of dimensions in x.
    raise NotImplementedError("Replace this line with your code.")
""", ["A vector should have 1 dimension.", "A matrix should have 2 dimensions."], "quiz.questions.tests.q34_tensor_num_dims", symbol_name="num_dims", hidden_binary_key="q34_tensor_num_dims"),
    question("q35_sum_columns", "00", "Sum Along Dimension 0", "00_tensors_and_autograd", "Write `sum_columns(x)` that returns `x.sum(dim=0)`.", """import torch


def sum_columns(x: torch.Tensor) -> torch.Tensor:
    # Sum across rows so one value remains for each column.
    raise NotImplementedError("Replace this line with your code.")
""", ["If x is shape (2, 3), the result should be shape (3,).", "The output should contain the column sums."], "quiz.questions.tests.q35_sum_columns", symbol_name="sum_columns", hidden_binary_key="q35_sum_columns"),
    question("q00_add_tensors", "00", "Add Two Tensors", "00_tensors_and_autograd", "Write a function `add_tensors(a, b)` that returns the elementwise tensor sum.", """import torch


def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Return the elementwise sum of a and b.
    raise NotImplementedError("Replace this line with your code.")
""", ["Matching positions should be added together.", "A (2, 2) input pair should return shape (2, 2)."], "quiz.questions.tests.q00_add_tensors", symbol_name="add_tensors", hidden_binary_key="q00_add_tensors"),
    question("q01_trainable_scalar", "00", "Create A Trainable Scalar", "00_tensors_and_autograd", "Write `build_trainable_scalar(value)` so the returned scalar tensor has `requires_grad=True`.", """import torch


def build_trainable_scalar(value: float) -> torch.Tensor:
    # Return a scalar tensor that PyTorch can differentiate with respect to.
    raise NotImplementedError("Replace this line with your code.")
""", ["The result should be a scalar tensor.", "The tensor must allow gradient tracking."], "quiz.questions.tests.q01_trainable_scalar", symbol_name="build_trainable_scalar", hidden_binary_key="q01_trainable_scalar"),
    question("q08_matrix_multiply", "00", "Matrix Multiply Two Tensors", "00_tensors_and_autograd", "Write `matrix_multiply(a, b)` that returns `a @ b`.", """import torch


def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Return the matrix product of a and b.
    raise NotImplementedError("Replace this line with your code.")
""", ["A 2x2 matrix multiplied by another 2x2 matrix should return shape (2, 2).", "Use matrix multiplication, not elementwise multiplication."], "quiz.questions.tests.q08_matrix_multiply", symbol_name="matrix_multiply", hidden_binary_key="q08_matrix_multiply"),
    question("q09_tensor_mean", "00", "Mean Of A Tensor", "00_tensors_and_autograd", "Write `tensor_mean(x)` that returns the scalar mean of all values in the tensor.", """import torch


def tensor_mean(x: torch.Tensor) -> torch.Tensor:
    # Return the scalar mean of x.
    raise NotImplementedError("Replace this line with your code.")
""", ["The result should be a scalar tensor.", "A tensor [1.0, 2.0, 3.0] should have mean 2.0."], "quiz.questions.tests.q09_tensor_mean", symbol_name="tensor_mean", hidden_binary_key="q09_tensor_mean"),
    question("q10_gradient_step", "00", "One Gradient Descent Step", "00_tensors_and_autograd", "Write `gradient_step(weight, learning_rate)` that computes the loss `(3 * weight - 9) ** 2`, runs backward, applies one no-grad update, and returns the updated weight.", """import torch


def gradient_step(weight: torch.Tensor, learning_rate: float) -> torch.Tensor:
    # Compute the loss, run backward(), update weight once, and return it.
    raise NotImplementedError("Replace this line with your code.")
""", ["The updated weight should move closer to 3.0 when starting from 2.0.", "Use `torch.no_grad()` for the update."], "quiz.questions.tests.q10_gradient_step", symbol_name="gradient_step", hidden_binary_key="q10_gradient_step"),

    question("q02_reshape_to_column", "01", "Reshape Into A Column", "01_shapes_gradients_and_reshape", "Write `reshape_to_column(values)` so a flat input becomes a tensor with shape `(n, 1)`.", """import torch


def reshape_to_column(values) -> torch.Tensor:
    # Convert values into a torch tensor and return shape (n, 1).
    raise NotImplementedError("Replace this line with your code.")
""", ["Input [1, 2, 3] should become shape (3, 1).", "Value order should stay top-to-bottom."], "quiz.questions.tests.q02_reshape_to_column", symbol_name="reshape_to_column", hidden_binary_key="q02_reshape_to_column"),
    question("q11_add_batch_dim", "01", "Add A Batch Dimension", "01_shapes_gradients_and_reshape", "Write `add_batch_dim(x)` so it adds a batch dimension at the front.", """import torch


def add_batch_dim(x: torch.Tensor) -> torch.Tensor:
    # Return x with one new dimension at the front.
    raise NotImplementedError("Replace this line with your code.")
""", ["If x is shape (3, 4), the result should be shape (1, 3, 4).", "Use an operation like unsqueeze, not manual loops."], "quiz.questions.tests.q11_add_batch_dim", symbol_name="add_batch_dim", hidden_binary_key="q11_add_batch_dim"),
    question("q12_remove_singleton_dim", "01", "Remove A Singleton Dimension", "01_shapes_gradients_and_reshape", "Write `remove_singleton_dim(x)` that removes a size-1 dimension from `(n, 1)` inputs.", """import torch


def remove_singleton_dim(x: torch.Tensor) -> torch.Tensor:
    # Return x without the singleton dimension.
    raise NotImplementedError("Replace this line with your code.")
""", ["If x is shape (3, 1), the result should be shape (3,).", "The values should stay in the same order."], "quiz.questions.tests.q12_remove_singleton_dim", symbol_name="remove_singleton_dim", hidden_binary_key="q12_remove_singleton_dim"),
    question("q13_flatten_tensor", "01", "Flatten A Tensor", "01_shapes_gradients_and_reshape", "Write `flatten_tensor(x)` that returns a 1D version of the tensor.", """import torch


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    # Return a flattened 1D tensor.
    raise NotImplementedError("Replace this line with your code.")
""", ["A 2x3 tensor should become shape (6,).", "The values should keep their row-major order."], "quiz.questions.tests.q13_flatten_tensor", symbol_name="flatten_tensor", hidden_binary_key="q13_flatten_tensor"),
    question("q14_add_bias", "01", "Broadcast Add A Bias Vector", "01_shapes_gradients_and_reshape", "Write `add_bias(matrix, bias)` so the bias vector is added across each row of the matrix.", """import torch


def add_bias(matrix: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Use broadcasting to add the bias to each row.
    raise NotImplementedError("Replace this line with your code.")
""", ["If matrix is (2, 3) and bias is (3,), the result should be (2, 3).", "Do not manually loop over the rows."], "quiz.questions.tests.q14_add_bias", symbol_name="add_bias", hidden_binary_key="q14_add_bias"),

    question("q03_linear_model", "02", "Manual Linear Model", "02_linear_regression_from_scratch", "Write `linear_model(x, weight, bias)` using the formula `x @ weight + bias`.", """import torch


def linear_model(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Compute the model output using matrix multiplication and bias broadcasting.
    raise NotImplementedError("Replace this line with your code.")
""", ["If x is (batch, 1) and weight is (1, 1), the result should be (batch, 1).", "The bias should broadcast across the batch."], "quiz.questions.tests.q03_linear_model", symbol_name="linear_model", hidden_binary_key="q03_linear_model"),
    question("q04_mse_loss", "02", "Mean Squared Error", "02_linear_regression_from_scratch", "Write `mean_squared_error(preds, targets)` so it returns `((preds - targets) ** 2).mean()`.", """import torch


def mean_squared_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Return the scalar mean squared error.
    raise NotImplementedError("Replace this line with your code.")
""", ["The output should be a scalar tensor.", "Larger errors should contribute more to the loss."], "quiz.questions.tests.q04_mse_loss", symbol_name="mean_squared_error", hidden_binary_key="q04_mse_loss"),
    question("q15_make_linear_params", "02", "Make Linear Parameters", "02_linear_regression_from_scratch", "Write `make_linear_params()` that returns `(weight, bias)` for a single-feature linear model. Both should require gradients.", """import torch


def make_linear_params():
    # Return weight shape (1, 1) and bias shape (1,), both with requires_grad=True.
    raise NotImplementedError("Replace this line with your code.")
""", ["Weight should have shape (1, 1).", "Bias should have shape (1,) and both tensors should require gradients."], "quiz.questions.tests.q15_make_linear_params", symbol_name="make_linear_params", hidden_binary_key="q15_make_linear_params"),
    question("q16_compute_residuals", "02", "Compute Residuals", "02_linear_regression_from_scratch", "Write `compute_residuals(preds, targets)` that returns `preds - targets`.", """import torch


def compute_residuals(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Return prediction errors.
    raise NotImplementedError("Replace this line with your code.")
""", ["Residuals should keep the same shape as preds.", "Positive residual means the prediction is above the target."], "quiz.questions.tests.q16_compute_residuals", symbol_name="compute_residuals", hidden_binary_key="q16_compute_residuals"),
    question("q17_apply_sgd_update", "02", "Apply An SGD Update", "02_linear_regression_from_scratch", "Write `apply_sgd_update(weight, bias, learning_rate)` that uses the existing `.grad` fields to update both tensors once under `torch.no_grad()` and returns them.", """import torch


def apply_sgd_update(weight: torch.Tensor, bias: torch.Tensor, learning_rate: float):
    # Update weight and bias using their gradients, then return them.
    raise NotImplementedError("Replace this line with your code.")
""", ["Use the existing gradients already stored on weight.grad and bias.grad.", "Perform the update inside torch.no_grad()."], "quiz.questions.tests.q17_apply_sgd_update", symbol_name="apply_sgd_update", hidden_binary_key="q17_apply_sgd_update"),

    question("q05_tiny_linear_forward", "03", "Tiny nn.Module Forward", "03_linear_regression_with_nn_module", "Complete the `forward` method so it returns `self.linear(x)`.", """import torch
from torch import nn


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the model prediction using the linear layer above.
        raise NotImplementedError("Replace this line with your code.")
""", ["The output shape should match `(batch, 1)`.", "Use the layer defined in `__init__`, not a new one."], "quiz.questions.tests.q05_tiny_linear_forward"),
    question("q18_build_linear_layer", "03", "Build A Linear Layer", "03_linear_regression_with_nn_module", "Write `build_linear_layer()` that returns `nn.Linear(1, 1)`.", """from torch import nn


def build_linear_layer():
    # Return a linear layer for one input feature and one output value.
    raise NotImplementedError("Replace this line with your code.")
""", ["The return value should be an nn.Linear module.", "It should map 1 input feature to 1 output feature."], "quiz.questions.tests.q18_build_linear_layer", symbol_name="build_linear_layer", hidden_binary_key="q18_build_linear_layer"),
    question("q19_make_regression_loader", "03", "Build A Regression DataLoader", "03_linear_regression_with_nn_module", "Write `make_regression_loader(x, y)` that returns a `DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)`.", """from torch.utils.data import DataLoader, TensorDataset


def make_regression_loader(x, y):
    # Return a DataLoader with batch_size=32 and shuffle=True.
    raise NotImplementedError("Replace this line with your code.")
""", ["The loader should iterate over paired x/y examples.", "Use batch_size=32 and shuffle=True."], "quiz.questions.tests.q19_make_regression_loader", symbol_name="make_regression_loader", hidden_binary_key="q19_make_regression_loader"),
    question("q20_compute_mse_loss", "03", "Compute Loss With nn.MSELoss", "03_linear_regression_with_nn_module", "Write `compute_mse_loss(preds, targets)` using `nn.MSELoss()`.", """from torch import nn


def compute_mse_loss(preds, targets):
    # Return the scalar MSE loss using nn.MSELoss.
    raise NotImplementedError("Replace this line with your code.")
""", ["The result should be a scalar tensor.", "Use nn.MSELoss rather than manual loss math in this question."], "quiz.questions.tests.q20_compute_mse_loss", symbol_name="compute_mse_loss", hidden_binary_key="q20_compute_mse_loss"),
    question("q21_run_training_step", "03", "Run One Training Step", "03_linear_regression_with_nn_module", "Write `run_training_step(model, batch_x, batch_y, optimizer)` that computes MSE loss, clears gradients, runs backward, steps the optimizer, and returns the loss.", """from torch import nn


def run_training_step(model, batch_x, batch_y, optimizer):
    # Compute MSE loss, zero gradients, backward, step, and return the loss tensor.
    raise NotImplementedError("Replace this line with your code.")
""", ["Use nn.MSELoss inside the function.", "Return the loss after running the optimizer step."], "quiz.questions.tests.q21_run_training_step"),

    question("q06_logistic_probability", "04", "Logistic Probability", "04_logistic_regression_from_scratch", "Write `logistic_probability(x, weight, bias)` so it returns `sigmoid(x @ weight + bias)`.", """import torch


def logistic_probability(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # Return probabilities between 0 and 1.
    raise NotImplementedError("Replace this line with your code.")
""", ["Output values must stay between 0 and 1.", "For x shape (batch, 2) and weight shape (2, 1), output shape should be (batch, 1)."], "quiz.questions.tests.q06_logistic_probability", symbol_name="logistic_probability", hidden_binary_key="q06_logistic_probability"),
    question("q22_sigmoid_function", "04", "Implement Sigmoid", "04_logistic_regression_from_scratch", "Write `sigmoid(z)` using the formula `1 / (1 + torch.exp(-z))`.", """import torch


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    # Return the sigmoid of z.
    raise NotImplementedError("Replace this line with your code.")
""", ["The output should always be between 0 and 1.", "sigmoid(0) should be close to 0.5."], "quiz.questions.tests.q22_sigmoid_function", symbol_name="sigmoid", hidden_binary_key="q22_sigmoid_function"),
    question("q23_threshold_predictions", "04", "Threshold Probabilities", "04_logistic_regression_from_scratch", "Write `threshold_predictions(probs)` that returns `(probs >= 0.5).float()`.", """import torch


def threshold_predictions(probs: torch.Tensor) -> torch.Tensor:
    # Convert probabilities into 0/1 predictions.
    raise NotImplementedError("Replace this line with your code.")
""", ["Values >= 0.5 should become 1.0.", "Values < 0.5 should become 0.0."], "quiz.questions.tests.q23_threshold_predictions", symbol_name="threshold_predictions", hidden_binary_key="q23_threshold_predictions"),
    question("q24_make_binary_labels", "04", "Make Binary Labels", "04_logistic_regression_from_scratch", "Write `make_binary_labels(scores)` that returns `(scores > 0).float().unsqueeze(1)`.", """import torch


def make_binary_labels(scores: torch.Tensor) -> torch.Tensor:
    # Return binary labels with shape (n, 1).
    raise NotImplementedError("Replace this line with your code.")
""", ["Positive scores should map to label 1.", "The output should have shape (n, 1)."], "quiz.questions.tests.q24_make_binary_labels", symbol_name="make_binary_labels", hidden_binary_key="q24_make_binary_labels"),
    question("q25_binary_cross_entropy", "04", "Binary Cross Entropy", "04_logistic_regression_from_scratch", "Write `binary_cross_entropy(preds, targets)` that clamps predictions and returns the mean BCE loss.", """import torch


def binary_cross_entropy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Clamp predictions and compute mean BCE loss.
    raise NotImplementedError("Replace this line with your code.")
""", ["The result should be a scalar tensor.", "Clamp predictions away from exact 0 and 1 before taking logs."], "quiz.questions.tests.q25_binary_cross_entropy", symbol_name="binary_cross_entropy", hidden_binary_key="q25_binary_cross_entropy"),

    question("q07_logistic_module_forward", "05", "Logistic Module Forward", "05_logistic_regression_with_nn_module", "Finish the `forward` method. Return raw logits from `self.linear(x)` and do not apply sigmoid.", """import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return raw logits, not probabilities.
        raise NotImplementedError("Replace this line with your code.")
""", ["The output shape should be (batch, 1).", "The returned values are logits, so they are not forced into 0..1."], "quiz.questions.tests.q07_logistic_module_forward"),
    question("q26_build_logistic_layer", "05", "Build A Logistic Layer", "05_logistic_regression_with_nn_module", "Write `build_logistic_layer()` that returns `nn.Linear(2, 1)`.", """from torch import nn


def build_logistic_layer():
    # Return a linear layer for 2 inputs and 1 logit output.
    raise NotImplementedError("Replace this line with your code.")
""", ["The returned layer should map 2 features to 1 output.", "It should be an nn.Linear module."], "quiz.questions.tests.q26_build_logistic_layer", symbol_name="build_logistic_layer", hidden_binary_key="q26_build_logistic_layer"),
    question("q27_logits_to_probs", "05", "Convert Logits To Probabilities", "05_logistic_regression_with_nn_module", "Write `logits_to_probs(logits)` that returns `torch.sigmoid(logits)`.", """import torch


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    # Convert raw logits into probabilities.
    raise NotImplementedError("Replace this line with your code.")
""", ["Output values should lie in [0, 1].", "The output shape should match the logits shape."], "quiz.questions.tests.q27_logits_to_probs", symbol_name="logits_to_probs", hidden_binary_key="q27_logits_to_probs"),
    question("q28_make_bce_loss", "05", "Create BCEWithLogitsLoss", "05_logistic_regression_with_nn_module", "Write `make_bce_loss()` that returns `nn.BCEWithLogitsLoss()`.", """from torch import nn


def make_bce_loss():
    # Return BCEWithLogitsLoss.
    raise NotImplementedError("Replace this line with your code.")
""", ["The return value should be an nn.BCEWithLogitsLoss instance.", "Do not apply sigmoid inside this function."], "quiz.questions.tests.q28_make_bce_loss", symbol_name="make_bce_loss", hidden_binary_key="q28_make_bce_loss"),
    question("q29_predict_binary_classes", "05", "Predict Binary Classes", "05_logistic_regression_with_nn_module", "Write `predict_binary_classes(logits)` that applies sigmoid and thresholds at 0.5.", """import torch


def predict_binary_classes(logits: torch.Tensor) -> torch.Tensor:
    # Return float predictions 0.0 or 1.0.
    raise NotImplementedError("Replace this line with your code.")
""", ["The output should contain only 0.0 and 1.0.", "Values with sigmoid >= 0.5 should map to 1.0."], "quiz.questions.tests.q29_predict_binary_classes", symbol_name="predict_binary_classes", hidden_binary_key="q29_predict_binary_classes"),
]

QUESTIONS.extend(EXTRA_QUESTIONS)


QUESTIONS_BY_ID = {question.id: question for question in QUESTIONS}
QUIZZES_BY_CHAPTER = {
    chapter_id: Quiz(chapter_id=chapter_id, questions=tuple(question for question in QUESTIONS if question.chapter_id == chapter_id))
    for chapter_id in {question.chapter_id for question in QUESTIONS}
}


def get_quiz(chapter_id: str) -> Quiz:
    return QUIZZES_BY_CHAPTER.get(chapter_id, Quiz(chapter_id=chapter_id))


def public_question_payload(question_ids: list[str]) -> list[dict[str, object]]:
    payload = []
    for question_id in question_ids:
        item = QUESTIONS_BY_ID[question_id]
        payload.append(deepcopy(item.to_public_payload()))
        payload[-1]["answer_code"] = ANSWER_CODES.get(question_id, "")
    return payload


def get_public_chapter_detail(chapter_id: str) -> dict[str, object] | None:
    chapter = get_chapter(chapter_id)
    if chapter is None:
        return None
    detail = deepcopy(chapter)
    quiz = get_quiz(chapter_id)
    detail["question_ids"] = quiz.question_ids
    detail["questions"] = public_question_payload(quiz.question_ids)
    detail["quiz"] = quiz.to_serializable()
    script_path = ROOT_DIR / detail["script_path"]
    detail["script_content"] = script_path.read_text(encoding="utf-8") if script_path.exists() else ""
    detail["playground_starter"] = PLAYGROUND_STARTER
    return detail
