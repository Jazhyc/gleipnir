"""Check the new strided cache boundary without allocating or copying GPU cache."""

import pytest
import torch

from gleipnir.serving.mxfp8 import split_paged_cache


def test_content_packed_views_keep_original_storage_and_strides():
    content = torch.empty(3, 4, 16, 512, dtype=torch.bfloat16)
    key, value = content.split(256, dim=-1)
    observed = split_paged_cache((key, value))
    assert observed[0] is key and observed[1] is value
    assert key.stride(-2) == value.stride(-2) == 512
    assert key.untyped_storage().data_ptr() == content.untyped_storage().data_ptr()
    assert value.storage_offset() == 256


def test_original_combined_cache_retains_kv_views():
    combined = torch.empty(3, 2, 4, 16, 256, dtype=torch.bfloat16)
    key, value = split_paged_cache(combined)
    assert key.untyped_storage().data_ptr() == combined.untyped_storage().data_ptr()
    assert key.shape == value.shape == (3, 4, 16, 256)
    assert key.stride(0) == combined.stride(0)


@pytest.mark.parametrize(
    "bad",
    [
        (torch.empty(3, 4, 16, 256), torch.empty(3, 4, 16, 256)),
        (torch.empty(3, 4, 16, 256, dtype=torch.bfloat16),),
        (
            torch.empty(3, 4, 16, 256, dtype=torch.bfloat16),
            torch.empty(4, 4, 16, 256, dtype=torch.bfloat16),
        ),
    ],
)
def test_invalid_cache_views_fail_before_kernel_dispatch(bad):
    with pytest.raises(ValueError, match="MXFP8 requires"):
        split_paged_cache(bad)
