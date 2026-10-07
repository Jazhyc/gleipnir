"""Cache-free whole-prompt worker over the selected precision/host reference."""

import hashlib
import json
import os
from collections import Counter

import torch

from experiments.b200_attention_gdn_serving.swiglu_native_output_worker import (
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT, write


class PromptOnlyNativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker(
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker
):
    def prompt_only_state(self) -> dict:
        """Report actual allocated cache storage after runner initialization."""
        caches = self.model_runner.kv_caches

        def tensors(value):
            if isinstance(value, torch.Tensor):
                return [value]
            if isinstance(value, (list, tuple)):
                return [t for x in value for t in tensors(x)]
            return []

        runner_bytes = sum(t.numel() * t.element_size() for t in tensors(caches))
        layers = self.model_runner.compilation_config.static_forward_context
        layer_bytes = sum(
            t.numel() * t.element_size()
            for layer in layers.values()
            for t in tensors(getattr(layer, "kv_cache", None))
        )
        return {
            "worker_pid": os.getpid(),
            "runner_cache_bytes": runner_bytes,
            "layer_cache_bytes": layer_bytes,
            "cache_specs": list(self.model_runner.get_kv_cache_spec()),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        condition = self.vllm_config.additional_config["serving_condition"]
        path = ROOT / condition["prompt_only_validation"]
        receipt = json.loads(path.read_text())
        if (
            not receipt["passed"]
            or receipt["gpu"] != torch.cuda.get_device_name()
            or Counter(r["stage"] for r in receipt["checks"])
            != {"convolution": 2, "attention": 3, "gdn": 2}
            or any(not r["bitwise_equal"] for r in receipt["checks"])
            or any(
                r.get("independent_reference_relative_l2", float("inf")) > 0.01
                for r in receipt["checks"]
                if r["stage"] == "convolution"
            )
        ):
            raise ValueError("prompt-only native admission failed")
        for source, expected in receipt["sources"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
                raise ValueError(f"prompt-only admission source drift: {source}")
        config = self.vllm_config
        if config.scheduler_config.enable_chunked_prefill or (
            config.cache_config.enable_prefix_caching or config.speculative_config
        ):
            raise ValueError("prompt-only mode requires unchunked, fresh requests")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir.serving_prompt_only import install

        seen = set()
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "calls": [],
            "validation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

        def observed(name, rows):
            if name in seen or torch.cuda.is_current_stream_capturing():
                return
            seen.add(name)
            audit["calls"].append({"layer": name, "rows": rows})
            audit["passed"] = len(seen) == 32
            write("native_prompt_only.json", audit)

        audit["scope"] = install(self.model_runner, observed)
        self.precision["prompt_only"] = audit["scope"]
        write("loaded_precision.json", self.precision)
        write("native_prompt_only.json", audit)
        print(
            "prompt_only_installed layers=32 persistent_cache_bytes=0 causal=true",
            flush=True,
        )
