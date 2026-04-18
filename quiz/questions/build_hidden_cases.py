from __future__ import annotations

import json
import zlib
from pathlib import Path

from quiz.questions.extra_hidden_cases import EXTRA_HIDDEN_CASES


def tensor(value, dtype: str | None = None, *, requires_grad: bool = False, grad: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"__kind__": "tensor", "value": value}
    if dtype is not None:
        payload["dtype"] = dtype
    if requires_grad:
        payload["requires_grad"] = True
    if grad is not None:
        payload["grad"] = grad
    return payload


HIDDEN_CASES = {
    "q00_add_tensors": [
        {"name": "adds negative values too", "args": [tensor([[1, -2], [3, -4]]), tensor([[5, 6], [-1, 2]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([[6, 4], [2, -2]])}], "hint": "Use direct tensor addition so every position is added elementwise."},
        {"name": "supports float tensors", "args": [tensor([1.5, 2.5], "float32"), tensor([0.5, -1.0], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([2.0, 1.5], "float32")}], "hint": "The same elementwise add should work for float tensors too."},
    ],
    "q01_trainable_scalar": [
        {"name": "keeps the original value", "args": [2.5], "assertions": [{"kind": "requires_grad", "expected": True}, {"kind": "item_compare", "op": "eq", "expected": 2.5}], "hint": "Create a scalar tensor from the input value and enable gradients on it."},
        {"name": "backward works on the returned scalar", "args": [1.25], "assertions": [{"kind": "expr", "expression": "(actual * 3).sum().backward() is None and actual.grad is not None"}], "hint": "The returned tensor should be a real trainable tensor, not a detached copy."},
    ],
    "q02_reshape_to_column": [
        {"name": "works on tensor input", "args": [tensor([4, 5, 6])], "assertions": [{"kind": "tensor_equal", "expected": tensor([[4], [5], [6]])}], "hint": "Reshape into `(n, 1)` so the values stack vertically."},
        {"name": "single value still becomes a column", "args": [[9]], "assertions": [{"kind": "shape", "expected": [1, 1]}], "hint": "Even one value should still have two dimensions after reshaping."},
    ],
    "q03_linear_model": [
        {"name": "bias broadcasts across multiple rows", "args": [tensor([[1.0], [2.0], [3.0]], "float32"), tensor([[2.0]], "float32"), tensor([0.5], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[2.5], [4.5], [6.5]], "float32")}], "hint": "Use `x @ weight + bias` so the bias broadcasts over the batch."},
        {"name": "returns float output", "args": [tensor([[1.0]], "float32"), tensor([[3.0]], "float32"), tensor([1.0], "float32")], "assertions": [{"kind": "expr", "expression": "str(actual.dtype).startswith('torch.float')"}], "hint": "The model output should stay as a floating-point tensor."},
    ],
    "q04_mse_loss": [
        {"name": "matches manual computation", "args": [tensor([[1.0], [3.0]], "float32"), tensor([[2.0], [1.0]], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor(2.5, "float32")}], "hint": "Subtract, square, then take the mean over all values."},
        {"name": "larger errors raise the loss", "args": [tensor([[10.0]], "float32"), tensor([[0.0]], "float32")], "assertions": [{"kind": "item_compare", "op": "gt", "expected": 50.0}], "hint": "Squared error should become large for very wrong predictions."},
    ],
    "q06_logistic_probability": [
        {"name": "matches sigmoid of the linear formula", "args": [tensor([[1.0, 2.0]], "float32"), tensor([[2.0], [-1.0]], "float32"), tensor([0.5], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[0.62245935]], "float32"), "atol": 1e-5}], "hint": "Compute the logit first, then apply sigmoid to it."},
        {"name": "works for multiple rows", "args": [tensor([[0.0, 0.0], [1.0, 1.0]], "float32"), tensor([[1.0], [1.0]], "float32"), tensor([0.0], "float32")], "assertions": [{"kind": "shape", "expected": [2, 1]}], "hint": "The output should preserve the batch dimension."},
    ],
    "q08_matrix_multiply": [
        {"name": "handles rectangular matrix multiply", "args": [tensor([[1, 2, 3], [4, 5, 6]]), tensor([[1], [0], [1]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([[4], [10]])}], "hint": "Use `a @ b` rather than elementwise multiplication."},
        {"name": "identity matrix keeps values", "args": [tensor([[3, 1], [2, 4]]), tensor([[1, 0], [0, 1]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([[3, 1], [2, 4]])}], "hint": "Matrix multiply by the identity should preserve the original matrix."},
    ],
    "q09_tensor_mean": [
        {"name": "computes scalar mean", "args": [tensor([[1.0, 3.0], [5.0, 7.0]], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor(4.0, "float32")}], "hint": "Call `.mean()` on the whole tensor so the result is one scalar."},
        {"name": "output is scalar", "args": [tensor([1.0, 2.0, 3.0], "float32")], "assertions": [{"kind": "expr", "expression": "actual.ndim == 0"}], "hint": "The overall tensor mean should be a scalar, not a vector."},
    ],
    "q10_gradient_step": [
        {"name": "keeps requires_grad", "args": [tensor(2.5, "float32", requires_grad=True), 0.1], "assertions": [{"kind": "requires_grad", "expected": True}], "hint": "The updated tensor should still be differentiable afterward."},
        {"name": "moves downward when starting above target", "args": [tensor(4.0, "float32", requires_grad=True), 0.1], "assertions": [{"kind": "item_compare", "op": "lt", "expected": 4.0}], "hint": "If the starting weight is above 3.0, the update should move it down."},
    ],
    "q11_add_batch_dim": [
        {"name": "works on vectors", "args": [tensor([1, 2, 3])], "assertions": [{"kind": "shape", "expected": [1, 3]}], "hint": "Add the new dimension at the front of the tensor."},
        {"name": "works on 3D tensors too", "args": [tensor([[[1], [2]]])], "assertions": [{"kind": "shape", "expected": [1, 1, 2, 1]}], "hint": "The operation should generalize to tensors with any rank."},
    ],
    "q12_remove_singleton_dim": [
        {"name": "single value becomes 1D", "args": [tensor([[9]])], "assertions": [{"kind": "shape", "expected": [1]}], "hint": "Remove the size-1 second dimension from `(n, 1)` inputs."},
        {"name": "values stay in order", "args": [tensor([[4], [7], [9]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([4, 7, 9])}], "hint": "The structure should change, not the values."},
    ],
    "q13_flatten_tensor": [
        {"name": "flattens 3D input", "args": [tensor([[[1, 2], [3, 4]]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 2, 3, 4])}], "hint": "Return one 1D tensor containing the original values in row-major order."},
        {"name": "output is 1D", "args": [tensor([[1, 2], [3, 4]])], "assertions": [{"kind": "shape", "expected": [4]}], "hint": "Flatten should collapse all dimensions into a single axis."},
    ],
    "q14_add_bias": [
        {"name": "supports negative bias values", "args": [tensor([[3.0, 4.0], [5.0, 6.0]], "float32"), tensor([-1.0, 2.0], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[2.0, 6.0], [4.0, 8.0]], "float32")}], "hint": "Use broadcasting so the bias is added to each row."},
        {"name": "preserves matrix shape", "args": [tensor([[1, 2, 3], [4, 5, 6]]), tensor([10, 20, 30])], "assertions": [{"kind": "shape", "expected": [2, 3]}], "hint": "Adding the bias should not change the matrix dimensions."},
    ],
    "q15_make_linear_params": [
        {"name": "returns a pair of tensors", "args": [], "assertions": [{"kind": "sequence_length", "expected": 2}, {"kind": "expr", "expression": "tuple(actual[0].shape) == (1, 1) and tuple(actual[1].shape) == (1,)"}, {"kind": "expr", "expression": "actual[0].requires_grad and actual[1].requires_grad"}], "hint": "Return `(weight, bias)` with the expected shapes and enable gradients on both."}
    ],
    "q16_compute_residuals": [
        {"name": "handles negative residuals", "args": [tensor([[1.0], [3.0]], "float32"), tensor([[2.0], [1.0]], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[-1.0], [2.0]], "float32")}], "hint": "Residuals are `preds - targets`, so the sign matters."}
    ],
    "q17_apply_sgd_update": [
        {"name": "uses learning rate scale", "args": [tensor([[1.0]], "float32", requires_grad=True, grad=tensor([[2.0]], "float32")), tensor([0.5], "float32", requires_grad=True, grad=tensor([1.0], "float32")), 0.1], "assertions": [{"kind": "sequence_length", "expected": 2}, {"kind": "expr", "expression": "abs(actual[0].item() - 0.8) < 1e-6 and abs(actual[1].item() - 0.4) < 1e-6"}, {"kind": "expr", "expression": "actual[0].requires_grad and actual[1].requires_grad"}], "hint": "Subtract `learning_rate * grad` from both tensors inside a no-grad block."}
    ],
    "q18_build_linear_layer": [
        {"name": "uses the correct features", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.linear.Linear"}, {"kind": "attr_equals", "attr": "in_features", "expected": 1}, {"kind": "attr_equals", "attr": "out_features", "expected": 1}], "hint": "Return exactly `nn.Linear(1, 1)`."}
    ],
    "q19_make_regression_loader": [
        {"name": "shuffles data", "args": [tensor([[1.0], [2.0], [3.0]], "float32"), tensor([[2.0], [4.0], [6.0]], "float32")], "assertions": [{"kind": "type_name", "expected": "torch.utils.data.dataloader.DataLoader"}, {"kind": "attr_equals", "attr": "batch_size", "expected": 32}, {"kind": "expr", "expression": "actual.sampler.__class__.__name__ == 'RandomSampler'"}], "hint": "Build a DataLoader with `batch_size=32` and `shuffle=True`."}
    ],
    "q20_compute_mse_loss": [
        {"name": "zero error gives zero loss", "args": [tensor([[2.0], [4.0]], "float32"), tensor([[2.0], [4.0]], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor(0.0, "float32")}], "hint": "If predictions exactly match targets, MSE should be zero."},
        {"name": "returns a scalar tensor", "args": [tensor([[1.0], [2.0]], "float32"), tensor([[0.0], [0.0]], "float32")], "assertions": [{"kind": "expr", "expression": "actual.ndim == 0"}], "hint": "The loss should be one scalar tensor, not a vector."},
    ],
    "q22_sigmoid_function": [
        {"name": "vector stays in probability range", "args": [tensor([-3.0, 0.0, 3.0], "float32")], "assertions": [{"kind": "between_zero_and_one"}], "hint": "The sigmoid output must always stay between 0 and 1."},
        {"name": "zero maps to one half", "args": [tensor(0.0, "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor(0.5, "float32")}], "hint": "Check the formula at `z = 0`."},
    ],
    "q23_threshold_predictions": [
        {"name": "thresholds a full vector", "args": [tensor([0.2, 0.5, 0.9], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([0.0, 1.0, 1.0], "float32")}], "hint": "Values greater than or equal to 0.5 should become 1.0."},
        {"name": "shape is preserved", "args": [tensor([[0.2], [0.8]], "float32")], "assertions": [{"kind": "shape", "expected": [2, 1]}], "hint": "Thresholding should not change the batch structure."},
    ],
    "q24_make_binary_labels": [
        {"name": "keeps column shape", "args": [tensor([-2.0, 0.0, 4.0], "float32")], "assertions": [{"kind": "shape", "expected": [3, 1]}, {"kind": "tensor_equal", "expected": tensor([[0.0], [0.0], [1.0]], "float32")}], "hint": "The labels should be a column tensor after `unsqueeze(1)`."},
        {"name": "dtype is float", "args": [tensor([-1.0, 1.0], "float32")], "assertions": [{"kind": "expr", "expression": "str(actual.dtype).startswith('torch.float')"}], "hint": "Convert the boolean condition to floating-point labels."},
    ],
    "q25_binary_cross_entropy": [
        {"name": "good predictions give smaller loss", "args": [tensor([[0.95], [0.05]], "float32"), tensor([[1.0], [0.0]], "float32")], "assertions": [{"kind": "item_compare", "op": "lt", "expected": 0.2}], "hint": "Confident correct predictions should produce a small BCE loss."},
        {"name": "multi-example result is scalar", "args": [tensor([[0.8], [0.2], [0.6]], "float32"), tensor([[1.0], [0.0], [1.0]], "float32")], "assertions": [{"kind": "expr", "expression": "actual.ndim == 0"}], "hint": "Average across the examples so the result is a scalar."},
    ],
    "q26_build_logistic_layer": [
        {"name": "returns a 2-to-1 linear layer", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.linear.Linear"}, {"kind": "attr_equals", "attr": "in_features", "expected": 2}, {"kind": "attr_equals", "attr": "out_features", "expected": 1}], "hint": "The layer should map 2 inputs to 1 logit."}
    ],
    "q27_logits_to_probs": [
        {"name": "returns probabilities for logits", "args": [tensor([[-1.0], [0.0], [1.0]], "float32")], "assertions": [{"kind": "between_zero_and_one"}, {"kind": "tensor_close", "expected": tensor([[0.26894143], [0.5], [0.7310586]], "float32"), "atol": 1e-5}], "hint": "Apply `torch.sigmoid` directly to the logits."}
    ],
    "q28_make_bce_loss": [
        {"name": "returns BCEWithLogitsLoss", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.loss.BCEWithLogitsLoss"}, {"kind": "callable"}, {"kind": "expr", "expression": "actual(torch.tensor([[0.0]], dtype=torch.float32), torch.tensor([[1.0]], dtype=torch.float32)).ndim == 0"}], "hint": "Return the loss module itself, not a computed loss value."}
    ],
    "q29_predict_binary_classes": [
        {"name": "returns hard float predictions", "args": [tensor([[-2.0], [0.0], [3.0]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([[0.0], [1.0], [1.0]], "float32")}], "hint": "Apply sigmoid first, then threshold at 0.5 and keep float outputs."},
        {"name": "preserves shape", "args": [tensor([[-1.0], [1.0]], "float32")], "assertions": [{"kind": "shape", "expected": [2, 1]}], "hint": "The output should keep the same batch/column structure as the logits."},
    ],
    "q30_create_tensor_from_list": [
        {"name": "works on float values", "args": [[1.5, 2.5]], "assertions": [{"kind": "tensor_equal", "expected": tensor([1.5, 2.5], "float32")}], "hint": "Make sure the function passes the values directly into `torch.tensor(...)`."},
        {"name": "works on tuples too", "args": [[7, 8]], "assertions": [{"kind": "tensor_equal", "expected": tensor([7, 8])}], "hint": "The function should accept normal Python sequences, not only lists."},
        {"name": "returns a torch tensor", "args": [[1]], "assertions": [{"kind": "type_name", "expected": "torch.Tensor"}], "hint": "Return a real tensor object, not a Python list or number."},
    ],
    "q31_tensor_dtype": [
        {"name": "uses float32 dtype", "args": [[4, 5]], "assertions": [{"kind": "type_name", "expected": "torch.Tensor"}, {"kind": "expr", "expression": "str(actual.dtype) == 'torch.float32'"}], "hint": "Explicitly pass `dtype=torch.float32` when creating the tensor."},
        {"name": "works on decimal input", "args": [[1.25, 2.75]], "assertions": [{"kind": "expr", "expression": "str(actual.dtype) == 'torch.float32'"}], "hint": "Keep the dtype fixed to float32 even with decimal inputs."},
    ],
    "q32_make_zeros_tensor": [
        {"name": "works on another shape", "args": [1, 4], "assertions": [{"kind": "tensor_equal", "expected": tensor([[0.0, 0.0, 0.0, 0.0]])}], "hint": "Use `torch.zeros(rows, cols)` so shape and values both match."},
        {"name": "returns a tensor", "args": [2, 2], "assertions": [{"kind": "type_name", "expected": "torch.Tensor"}], "hint": "Return a real PyTorch tensor object."},
    ],
    "q33_slice_first_row": [
        {"name": "handles larger matrices", "args": [tensor([[2, 4], [6, 8], [10, 12]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([2, 4])}], "hint": "Slice the row directly instead of hardcoding a shape."},
        {"name": "keeps order on wider matrices", "args": [tensor([[9, 8, 7, 6], [1, 2, 3, 4]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([9, 8, 7, 6])}], "hint": "The returned row should preserve the left-to-right order."},
    ],
    "q34_tensor_num_dims": [
        {"name": "scalar has zero dims", "args": [tensor(5)], "assertions": [{"kind": "equals", "expected": 0}], "hint": "A scalar tensor has shape `()` and therefore zero dimensions."},
        {"name": "three dimensional tensor", "args": [tensor([[[1], [2]], [[3], [4]]])], "assertions": [{"kind": "equals", "expected": 3}], "hint": "Use `x.ndim` or `x.dim()` so the function generalizes."},
        {"name": "returns an int", "args": [tensor([[1, 2], [3, 4]])], "assertions": [{"kind": "expr", "expression": "isinstance(actual, int)"}], "hint": "Return the dimension count as a Python integer."},
    ],
    "q35_sum_columns": [
        {"name": "sums each column", "args": [tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([12, 15, 18])}], "hint": "Sum across rows with `dim=0`, not across columns with `dim=1`."},
        {"name": "handles negative values", "args": [tensor([[1, -2], [3, -4]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([4, -6])}], "hint": "Column sums should still work when some values are negative."},
    ],
}

HIDDEN_CASES.update(EXTRA_HIDDEN_CASES)


def main() -> None:
    target = Path(__file__).with_name("hidden_cases.bin")
    payload = json.dumps(HIDDEN_CASES).encode("utf-8")
    target.write_bytes(zlib.compress(payload, level=9))


if __name__ == "__main__":
    main()
