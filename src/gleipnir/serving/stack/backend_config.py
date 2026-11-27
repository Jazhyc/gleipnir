"""Preserve explicit GDN selection with vLLM's custom configuration hash hook."""

import functools
import hashlib
import os
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gleipnir.serving.compile_cache import ServingCompileConfig

UPSTREAM_SHA256 = "3bd2f805f0f4b821e6902ef18da8ba66ad42a7e5a633b76abe96892532293d9c"
FLASHINFER_SHA256 = "3dc5005eded5c9e152ebf32359e7a3238e0c0924522a1a3482f19fb8138b4f5f"


def nonparallel_call(original: Callable[..., Any], *args, **kwargs) -> Any:
    """Pin the published non-CP route; reject an explicitly conflicting call."""
    if kwargs.get("backend") != "flashinfer":
        raise ValueError("migration non-CP adapter requires explicit FlashInfer")
    if kwargs.get("use_cp", False) is not False:
        raise ValueError("migration non-CP adapter received conflicting use_cp")
    kwargs["use_cp"] = False
    return original(*args, **kwargs)


def resolver_config(config: Any) -> Any:
    """Give only the backend resolver a dict; retain SupportsHash for the engine."""
    if isinstance(config.additional_config, ServingCompileConfig):
        return SimpleNamespace(
            additional_config=dict(config.additional_config),
            model_config=config.model_config,
        )
    return config


def install() -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    if version("vllm") != "0.31.0":
        raise ValueError("GDN configuration bridge requires vLLM 0.31")
    if hashlib.sha256(Path(gdn.__file__).read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("GDN backend resolver source changed")
    if getattr(gdn, "_gleipnir_mapping_bridge", False):
        return
    original = gdn._resolve_gdn_prefill_backend

    def resolve(config):
        return original(resolver_config(config))

    gdn._resolve_gdn_prefill_backend = resolve
    gdn._gleipnir_mapping_bridge = True
    mode = os.environ.get("GLEIPNIR_FLASHINFER_GDN_CP", "auto")
    if mode == "off":
        from flashinfer import gdn_prefill

        if version("flashinfer-python") != "0.7.0.post1" or (
            hashlib.sha256(Path(gdn_prefill.__file__).read_bytes()).hexdigest()
            != FLASHINFER_SHA256
        ):
            raise ValueError("FlashInfer GDN non-CP source changed")
        fi_original = gdn_prefill.chunk_gated_delta_rule

        @functools.wraps(fi_original)
        def nonparallel(*args, **kwargs):
            return nonparallel_call(fi_original, *args, **kwargs)

        nonparallel._gleipnir_nonparallel = True
        gdn_prefill.chunk_gated_delta_rule = nonparallel
    elif mode != "auto":
        raise ValueError("unknown FlashInfer GDN context-parallel mode")
