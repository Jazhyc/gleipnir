"""Native FP4 MLP with SDPA packing: a separately recorded composite recipe.

Requested by the user on 2026-10-16 for the teacher precision factorial,
extending the 2026-10-05 native FP4 finite acceptance. Startup validation
reuses the accepted BF16/SDPA receipt for attention packing and the
checksum-bound FP4 receipt for MLP arithmetic; see
``gleipnir.training.startup.composite_native_fp4_sdpa_reference``. The bound
FP4/FA4 module (``native_fp4.py``) is unchanged; every other envelope check
is delegated to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from gleipnir.training.backends import native_fp4

SDPA_REFERENCE = (
    "results/student_injection_awareness/4b/regular/causal_adapter/"
    "training_metadata.json"
)
SDPA_REFERENCE_SHA256 = (
    "91fadffbe9e69523c1c65cf485393ce2cc2f08aea8dafb2987f7c7663fa5f69f"
)
AUTHORITY_KEY = "native_fp4_sdpa_composite_authority"


def is_native_fp4_sdpa(training: Mapping[str, Any]) -> bool:
    """Route only explicitly marked native FP4 configs without FA4 settings."""
    return training.get("native_fp4_mlp") is True and AUTHORITY_KEY in training


def validate_native_fp4_sdpa_config(student: Mapping[str, Any]) -> bool:
    training = student["training"]
    authority = training.get(AUTHORITY_KEY)
    if not isinstance(authority, str) or not authority.strip():
        raise ValueError("native FP4 with SDPA requires a recorded user authority")
    if (
        training.get("packed_attention_backend") is not None
        or training.get("packed_attention_version") is not None
        or training.get("packing_learning_gradient_tolerance") is not None
        or training.get("startup_validation_reference_sha256") != SDPA_REFERENCE_SHA256
    ):
        raise ValueError("native FP4 with SDPA requires the accepted SDPA receipt")
    # Delegate the full FP4 envelope to the bound validator, substituting only
    # the attention fields this composite recipe replaces.
    shadow = deepcopy(dict(student))
    shadow["training"] = {
        **{k: v for k, v in training.items() if k != AUTHORITY_KEY},
        "packed_attention_backend": "flash_attention_4",
        "packed_attention_version": "4.0.0b33",
        "packing_learning_gradient_tolerance": 0.05,
        "startup_validation_reference_sha256": native_fp4.REFERENCE_SHA256,
    }
    return native_fp4.validate_native_fp4_config(shadow)
