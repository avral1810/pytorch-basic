from __future__ import annotations


def tensor(value, dtype: str | None = None, *, requires_grad: bool = False, grad: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"__kind__": "tensor", "value": value}
    if dtype is not None:
        payload["dtype"] = dtype
    if requires_grad:
        payload["requires_grad"] = True
    if grad is not None:
        payload["grad"] = grad
    return payload


EXTRA_HIDDEN_CASES = {
    "q36_init_manual_mlp_params": [
        {
            "name": "returns four trainable tensors",
            "args": [7],
            "assertions": [
                {"kind": "sequence_length", "expected": 4},
                {"kind": "expr", "expression": "tuple(actual[0].shape) == (2, 16) and tuple(actual[1].shape) == (16,) and tuple(actual[2].shape) == (16, 2) and tuple(actual[3].shape) == (2,)"},
                {"kind": "expr", "expression": "all(item.requires_grad for item in actual)"},
            ],
        }
    ],
    "q37_manual_mlp_forward": [
        {
            "name": "returns manual logits",
            "args": [
                tensor([[1.0, -1.0], [0.5, 0.5]], "float32"),
                tensor([[1.0, 0.0], [0.0, 1.0]], "float32"),
                tensor([0.5, -0.5], "float32"),
                tensor([[2.0, -1.0], [1.0, 1.0]], "float32"),
                tensor([0.0, 0.5], "float32"),
            ],
            "assertions": [{"kind": "tensor_close", "expected": tensor([[3.0, -0.5], [2.5, 1.0]], "float32")}],
        }
    ],
    "q38_manual_predict_classes": [
        {"name": "uses argmax over classes", "args": [tensor([[0.1, 0.9], [3.0, 1.0]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q39_manual_mlp_loss": [
        {"name": "returns scalar cross entropy", "args": [tensor([[2.0, 0.5], [0.2, 1.5]], "float32"), tensor([0, 1])], "assertions": [{"kind": "expr", "expression": "actual.ndim == 0 and actual.item() > 0"}]}
    ],
    "q41_build_mlp_sequential": [
        {"name": "returns the expected sequential stack", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.container.Sequential"}, {"kind": "sequence_length", "expected": 5}, {"kind": "expr", "expression": "actual[0].in_features == 2 and actual[0].out_features == 16 and actual[2].in_features == 16 and actual[4].out_features == 2"}]}
    ],
    "q42_make_classification_loader": [
        {"name": "builds a shuffled loader", "args": [tensor([[1.0, 2.0], [3.0, 4.0]], "float32"), tensor([0, 1])], "assertions": [{"kind": "type_name", "expected": "torch.utils.data.dataloader.DataLoader"}, {"kind": "attr_equals", "attr": "batch_size", "expected": 64}, {"kind": "expr", "expression": "actual.sampler.__class__.__name__ == 'RandomSampler'"}]}
    ],
    "q43_compute_multiclass_accuracy": [
        {"name": "returns float accuracy", "args": [tensor([[2.0, 1.0], [0.1, 0.9], [0.7, 0.3]], "float32"), tensor([0, 1, 1])], "assertions": [{"kind": "expr", "expression": "isinstance(actual, float) and abs(actual - (2/3)) < 1e-8"}]}
    ],
    "q46_init_manual_cnn_params": [
        {"name": "returns eight trainable tensors", "args": [123], "assertions": [{"kind": "sequence_length", "expected": 8}, {"kind": "expr", "expression": "tuple(actual[0].shape) == (8, 1, 3, 3) and tuple(actual[2].shape) == (16, 8, 3, 3) and tuple(actual[4].shape) == (256, 32) and tuple(actual[6].shape) == (32, 3)"}, {"kind": "expr", "expression": "all(item.requires_grad for item in actual)"}]}
    ],
    "q47_manual_conv_relu_pool": [
        {"name": "shrinks spatial size after pooling", "args": [tensor([[[[1.0] * 16] * 16]], "float32"), tensor([[[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]], "float32"), tensor([0.0], "float32")], "assertions": [{"kind": "shape", "expected": [1, 1, 8, 8]}]}
    ],
    "q48_flatten_image_features": [
        {"name": "keeps batch dimension", "args": [tensor([[[[1.0] * 4] * 4] for _ in range(2)], "float32")], "assertions": [{"kind": "shape", "expected": [2, 16]}]}
    ],
    "q49_manual_cnn_classifier": [
        {"name": "returns class logits", "args": [tensor([[1.0, 2.0], [3.0, 4.0]], "float32"), tensor([[1.0, 0.0], [0.0, 1.0]], "float32"), tensor([0.0, 0.0], "float32"), tensor([[1.0, -1.0, 0.5], [0.5, 0.5, 1.0]], "float32"), tensor([0.0, 0.5, -0.5], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[2.0, 0.5, 2.0], [5.0, -0.5, 5.0]], "float32")}]} 
    ],
    "q50_predict_image_classes": [
        {"name": "predicts image classes", "args": [tensor([[0.1, 0.8, 0.3], [2.0, 1.0, -1.0]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q51_build_cnn_feature_extractor": [
        {"name": "returns the expected conv stack", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.container.Sequential"}, {"kind": "sequence_length", "expected": 6}, {"kind": "expr", "expression": "actual[0].in_channels == 1 and actual[0].out_channels == 8 and actual[3].in_channels == 8 and actual[3].out_channels == 16"}]}
    ],
    "q52_conv_output_size": [
        {"name": "preserves 28 with padding 1", "args": [28, 3, 1, 1], "assertions": [{"kind": "equals", "expected": 28}]}
    ],
    "q53_build_cnn_classifier_head": [
        {"name": "returns the expected classifier", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.container.Sequential"}, {"kind": "sequence_length", "expected": 4}, {"kind": "expr", "expression": "actual[1].in_features == 16 * 7 * 7 and actual[3].out_features == 3"}]}
    ],
    "q54_flatten_cnn_batch": [
        {"name": "flattens feature maps", "args": [tensor([[[[1.0] * 7] * 7] for _ in range(4)], "float32")], "assertions": [{"kind": "shape", "expected": [4, 49]}]}
    ],
    "q55_predict_cnn_classes": [
        {"name": "predicts cnn classes", "args": [tensor([[0.2, 1.0, -1.0], [3.0, 2.0, 1.0]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q56_add_channel_dim": [
        {"name": "adds a single channel axis", "args": [tensor([[1.0, 2.0], [3.0, 4.0]], "float32")], "assertions": [{"kind": "shape", "expected": [1, 2, 2]}]}
    ],
    "q57_build_pattern_loader": [
        {"name": "builds a shuffled vision loader", "args": [tensor([[[[1.0] * 4] * 4] for _ in range(3)], "float32"), tensor([0, 1, 2])], "assertions": [{"kind": "type_name", "expected": "torch.utils.data.dataloader.DataLoader"}, {"kind": "attr_equals", "attr": "batch_size", "expected": 64}, {"kind": "expr", "expression": "actual.sampler.__class__.__name__ == 'RandomSampler'"}]}
    ],
    "q59_build_vision_cnn": [
        {"name": "returns the expected vision cnn", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.container.Sequential"}, {"kind": "sequence_length", "expected": 10}, {"kind": "expr", "expression": "actual[0].in_channels == 1 and actual[0].out_channels == 8 and actual[7].in_features == 16 * 4 * 4 and actual[9].out_features == 3"}]}
    ],
    "q60_generate_diagonal_pattern": [
        {"name": "makes a diagonal image", "args": [4], "assertions": [{"kind": "shape", "expected": [1, 4, 4]}, {"kind": "tensor_equal", "expected": tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], "float32")}]}
    ],
    "q61_embedding_lookup": [
        {"name": "indexes the embedding table", "args": [tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "float32"), tensor([[0, 2], [1, 1]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [3.0, 4.0]]], "float32")}]}
    ],
    "q62_manual_rnn_step": [
        {"name": "computes a tanh recurrent update", "args": [tensor([[1.0, 0.0]], "float32"), tensor([[0.5, -0.5]], "float32"), tensor([[1.0, 0.0], [0.0, 1.0]], "float32"), tensor([[1.0, 0.0], [0.0, 1.0]], "float32"), tensor([0.0, 0.0], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[0.90514827, -0.46211717]], "float32"), "atol": 1e-5}]}
    ],
    "q63_run_manual_rnn": [
        {"name": "loops through time", "args": [tensor([[[1.0, 0.0], [0.0, 1.0]]], "float32"), tensor([[1.0, 0.0], [0.0, 1.0]], "float32"), tensor([[0.5, 0.0], [0.0, 0.5]], "float32"), tensor([0.0, 0.0], "float32")], "assertions": [{"kind": "shape", "expected": [1, 2]}, {"kind": "expr", "expression": "actual.abs().sum().item() > 0"}]}
    ],
    "q64_manual_sequence_logits": [
        {"name": "maps hidden states to logits", "args": [tensor([[1.0, 2.0]], "float32"), tensor([[1.0, 0.0], [0.0, 1.0]], "float32"), tensor([0.5, -0.5], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[1.5, 1.5]], "float32")}]}
    ],
    "q65_predict_manual_rnn_classes": [
        {"name": "predicts manual rnn classes", "args": [tensor([[1.0, 2.0], [3.0, 1.0]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q66_build_vanilla_rnn": [
        {"name": "returns the lesson rnn", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.rnn.RNN"}, {"kind": "attr_equals", "attr": "input_size", "expected": 32}, {"kind": "attr_equals", "attr": "hidden_size", "expected": 32}, {"kind": "expr", "expression": "actual.batch_first and actual.nonlinearity == 'tanh'"}]}
    ],
    "q67_take_last_hidden": [
        {"name": "takes the last recurrent layer", "args": [tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([[5.0, 6.0], [7.0, 8.0]], "float32")}]}
    ],
    "q68_make_rnn_labels": [
        {"name": "compares early and late token averages", "args": [tensor([[1, 1, 1, 1, 5, 5, 5, 5], [8, 8, 8, 8, 1, 1, 1, 1]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q69_build_rnn_classifier_head": [
        {"name": "returns a 32 to 2 head", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.linear.Linear"}, {"kind": "attr_equals", "attr": "in_features", "expected": 32}, {"kind": "attr_equals", "attr": "out_features", "expected": 2}]}
    ],
    "q70_predict_rnn_classes": [
        {"name": "predicts rnn classes", "args": [tensor([[0.1, 0.2], [5.0, 1.0]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q71_build_embedding_layer": [
        {"name": "returns the embedding layer", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.sparse.Embedding"}, {"kind": "attr_equals", "attr": "num_embeddings", "expected": 15}, {"kind": "attr_equals", "attr": "embedding_dim", "expected": 32}]}
    ],
    "q72_build_lstm_layer": [
        {"name": "returns the lesson lstm", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.rnn.LSTM"}, {"kind": "attr_equals", "attr": "input_size", "expected": 32}, {"kind": "attr_equals", "attr": "hidden_size", "expected": 32}, {"kind": "expr", "expression": "actual.batch_first"}]}
    ],
    "q73_take_lstm_final_hidden": [
        {"name": "takes the final lstm hidden state", "args": [tensor([[[1.0, 2.0]], [[3.0, 4.0]]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([[3.0, 4.0]], "float32")}]}
    ],
    "q74_build_lstm_classifier_head": [
        {"name": "returns a 32 to 2 head", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.linear.Linear"}, {"kind": "attr_equals", "attr": "in_features", "expected": 32}, {"kind": "attr_equals", "attr": "out_features", "expected": 2}]}
    ],
    "q75_predict_lstm_classes": [
        {"name": "predicts lstm classes", "args": [tensor([[0.5, 0.7], [2.0, 1.0]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q76_positional_encoding": [
        {"name": "returns the right positional shape", "args": [4, 6], "assertions": [{"kind": "shape", "expected": [4, 6]}, {"kind": "expr", "expression": "abs(actual[0, 0].item()) < 1e-6 and abs(actual[0, 1].item() - 1.0) < 1e-6"}]}
    ],
    "q77_scaled_attention_scores": [
        {"name": "returns a square score matrix", "args": [tensor([[[1.0, 0.0], [0.0, 1.0]]], "float32"), tensor([[[1.0, 0.0], [0.0, 1.0]]], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[[0.70710677, 0.0], [0.0, 0.70710677]]], "float32"), "atol": 1e-5}]}
    ],
    "q78_attention_weighted_values": [
        {"name": "mixes value vectors", "args": [tensor([[[1.0, 0.0], [0.5, 0.5]]], "float32"), tensor([[[2.0, 1.0], [0.0, 4.0]]], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[[2.0, 1.0], [1.0, 2.5]]], "float32")}]}
    ],
    "q79_mean_pool_sequence": [
        {"name": "mean pools over tokens", "args": [tensor([[[1.0, 3.0], [5.0, 7.0]]], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[3.0, 5.0]], "float32")}]}
    ],
    "q80_manual_transformer_logits": [
        {"name": "maps pooled vectors to logits", "args": [tensor([[1.0, 2.0]], "float32"), tensor([[1.0, 0.0], [0.0, 1.0]], "float32"), tensor([0.5, -0.5], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[1.5, 1.5]], "float32")}]}
    ],
    "q81_pool_encoder_output": [
        {"name": "pools encoder outputs", "args": [tensor([[[1.0, 2.0], [3.0, 4.0]]], "float32")], "assertions": [{"kind": "tensor_close", "expected": tensor([[2.0, 3.0]], "float32")}]}
    ],
    "q82_build_transformer_encoder_layer": [
        {"name": "returns the configured encoder layer", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.transformer.TransformerEncoderLayer"}, {"kind": "expr", "expression": "actual.self_attn.embed_dim == 32 and actual.self_attn.num_heads == 4 and actual.self_attn.batch_first"}]}
    ],
    "q83_build_transformer_classifier_head": [
        {"name": "returns a 32 to 2 head", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.linear.Linear"}, {"kind": "attr_equals", "attr": "in_features", "expected": 32}, {"kind": "attr_equals", "attr": "out_features", "expected": 2}]}
    ],
    "q84_make_transformer_labels": [
        {"name": "uses the token-sum threshold", "args": [tensor([[10, 10, 10, 10], [1, 1, 1, 1]])], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q85_predict_transformer_classes": [
        {"name": "predicts transformer classes", "args": [tensor([[0.3, 0.9], [5.0, 1.0]], "float32")], "assertions": [{"kind": "tensor_equal", "expected": tensor([1, 0])}]}
    ],
    "q86_build_generator": [
        {"name": "returns the generator stack", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.container.Sequential"}, {"kind": "sequence_length", "expected": 5}, {"kind": "expr", "expression": "actual[0].in_features == 3 and actual[4].out_features == 1"}]}
    ],
    "q87_build_discriminator": [
        {"name": "returns the discriminator stack", "args": [], "assertions": [{"kind": "type_name", "expected": "torch.nn.modules.container.Sequential"}, {"kind": "sequence_length", "expected": 5}, {"kind": "expr", "expression": "actual[0].in_features == 1 and actual[4].out_features == 1"}]}
    ],
    "q88_sample_gan_noise": [
        {"name": "samples 3D noise vectors", "args": [5], "assertions": [{"kind": "shape", "expected": [5, 3]}, {"kind": "expr", "expression": "str(actual.dtype).startswith('torch.float')"}]}
    ],
    "q89_make_real_and_fake_targets": [
        {"name": "returns ones and zeros like logits", "args": [tensor([[1.0], [2.0]], "float32")], "assertions": [{"kind": "sequence_length", "expected": 2}, {"kind": "expr", "expression": "torch.equal(actual[0], torch.ones(2, 1)) and torch.equal(actual[1], torch.zeros(2, 1))"}]}
    ],
    "q90_detach_fake_batch": [
        {"name": "detaches fake samples", "args": [tensor([[1.0], [2.0]], "float32", requires_grad=True)], "assertions": [{"kind": "tensor_equal", "expected": tensor([[1.0], [2.0]], "float32")}, {"kind": "requires_grad", "expected": False}]}
    ],
    "q91_get_device": [
        {"name": "returns a torch device", "args": [], "assertions": [{"kind": "expr", "expression": "actual.type in {'cpu', 'cuda', 'mps'}"}]}
    ],
    "q93_make_device_inputs": [
        {"name": "creates tensors on the requested device", "args": [4, "cpu"], "assertions": [{"kind": "sequence_length", "expected": 2}, {"kind": "expr", "expression": "tuple(actual[0].shape) == (4, 4) and tuple(actual[1].shape) == (4,) and actual[0].device.type == 'cpu' and actual[1].device.type == 'cpu'"}]}
    ],
    "q94_move_tensor_pair_to_device": [
        {"name": "moves both tensors to cpu", "args": [tensor([[1.0, 2.0]], "float32"), tensor([1]), "cpu"], "assertions": [{"kind": "sequence_length", "expected": 2}, {"kind": "expr", "expression": "actual[0].device.type == 'cpu' and actual[1].device.type == 'cpu' and torch.equal(actual[0], torch.tensor([[1.0, 2.0]])) and torch.equal(actual[1], torch.tensor([1]))"}]}
    ],
}
