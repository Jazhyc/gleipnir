"""Check end-to-end caller routing and retain mixed/speculative/decode copies."""

from types import SimpleNamespace

import pytest
import torch

from gleipnir.serving_gdn_direct_caller import build_core_source

SOURCE = """
def _forward_core(self, mixed_qkv, b, a, core_attn_out):
    attn_metadata = get_metadata()
    num_actual_tokens = attn_metadata.num_actual_tokens
    spec_sequence_masks = attn_metadata.spec_sequence_masks
    split_non_spec = (
        spec_sequence_masks is None
        and attn_metadata.num_prefills > 0
        and attn_metadata.num_decodes > 0
    )
    if attn_metadata.num_prefills > 0:
        core_attn_out_non_spec, state = self.chunk_gated_delta_rule(
            q=mixed_qkv, output_final_state=True,
        )
    else:
        core_attn_out_non_spec = torch.full((1, 5, 4, 128), 3.0)
    if spec_sequence_masks is not None:
        merged_out = core_attn_out_non_spec + 1
        core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
    else:
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
"""


@pytest.mark.parametrize(
    "prefills,decodes,spec,direct,value",
    [
        (1, 0, None, True, 1),
        (1, 1, None, False, 2),
        (1, 0, object(), False, 3),
        (0, 1, None, False, 3),
    ],
)
def test_caller_forwards_destination_and_removes_only_ordinary_prefill_copy(
    prefills, decodes, spec, direct, value
):
    metadata = SimpleNamespace(
        num_actual_tokens=5,
        num_prefills=prefills,
        num_decodes=decodes,
        spec_sequence_masks=spec,
    )
    namespace = {"torch": torch, "get_metadata": lambda: metadata}
    exec(compile(build_core_source(SOURCE), "<caller-test>", "exec"), namespace)

    class Destination:
        def __init__(self):
            self.tensor = torch.full((9, 4, 128), -123.0)
            self.copies = 0

        def __getitem__(self, index):
            return self.tensor[index]

        def __setitem__(self, index, value):
            self.copies += 1
            self.tensor[index] = value

    destination = Destination()
    calls = []

    def chunk(**kw):
        calls.append(kw)
        output = kw["core_attn_out"]
        if output is not None:
            assert output.data_ptr() == destination.tensor.data_ptr()
            assert output.shape == (5, 4, 128)
            output.fill_(1)
            return output.unsqueeze(0), None
        return torch.full((1, 5, 4, 128), 2.0), None

    layer = SimpleNamespace(chunk_gated_delta_rule=chunk)
    namespace["_forward_core"](layer, None, None, None, destination)
    assert destination.copies == (0 if direct else 1)
    assert torch.all(destination.tensor[:5] == value)
    assert torch.all(destination.tensor[5:] == -123)
    if prefills:
        assert (calls[0]["core_attn_out"] is not None) is direct


def test_caller_rejects_already_forwarded_or_changed_source():
    with pytest.raises(ValueError, match="already forwards"):
        build_core_source(
            SOURCE.replace("q=mixed_qkv,", "q=mixed_qkv, core_attn_out=None,")
        )
    with pytest.raises(ValueError, match="unsupported vLLM GDN caller"):
        build_core_source(SOURCE.replace("self.chunk_gated_delta_rule", "self.other"))
    with pytest.raises(ValueError, match="unsupported vLLM GDN caller"):
        build_core_source(
            SOURCE.replace("core_attn_out_non_spec.squeeze(0)", "other_output")
        )
