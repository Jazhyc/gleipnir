"""Check opaque native operator metadata, gradients and full-graph capture."""

import pytest
import torch

from gleipnir.fouroversix_training import FrozenFourOverSixLinear, FrozenFp4Runtime
from gleipnir.fp4_compiler_ops import CompilerVisibleFourOverSixLinear


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compiler_visible_projection_preserves_exact_forward_and_backward(dtype):
    torch.manual_seed(17)
    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.requires_grad_(False)
    decoded = original.weight.detach().clone() + 0.125

    def runtime():
        return FrozenFp4Runtime(
            decoded,
            None,
            None,
            None,
            lambda inputs, weight, **kwargs: inputs @ weight.T,
            dequantized_weight=decoded,
        )

    reference = FrozenFourOverSixLinear(original, runtime())
    visible = CompilerVisibleFourOverSixLinear(original, runtime())
    values = torch.randn(2, 7, 16, dtype=dtype)
    inputs = values.clone().requires_grad_()
    expected_inputs = values.clone().requires_grad_()
    actual = torch.compile(visible, backend="aot_eager", fullgraph=True)(inputs)
    expected = reference(expected_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=0, atol=0)
    assert visible.runtime.forward_calls == visible.runtime.backward_calls == 1
    assert original.weight.grad is None


def test_visible_projection_requires_decoded_bf16_backward():
    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.requires_grad_(False)
    runtime = FrozenFp4Runtime(None, None, None, None, lambda *args: None)
    with pytest.raises(ValueError, match="BF16 backward"):
        CompilerVisibleFourOverSixLinear(original, runtime)


def test_runtime_keys_reuse_one_compiled_graph_across_projection_instances():
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    def action(projection, values):
        return projection(values)

    compiled = torch.compile(action, backend=backend, fullgraph=True)
    values = torch.randn(2, 7, 16, dtype=torch.bfloat16)
    for _ in range(3):
        original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
        original.requires_grad_(False)
        runtime = FrozenFp4Runtime(
            original.weight,
            None,
            None,
            None,
            lambda inputs, weight, **kwargs: inputs @ weight.T,
            dequantized_weight=original.weight,
        )
        projection = CompilerVisibleFourOverSixLinear(original, runtime)
        torch.testing.assert_close(
            compiled(projection, values), projection(values), rtol=0, atol=0
        )
        assert projection.native_runtime_key.device.type == "cpu"
    assert len(graphs) == 1
