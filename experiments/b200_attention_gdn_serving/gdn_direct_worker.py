"""Install native-admitted direct GDN destinations on the Direct FP4 stack."""

import hashlib
import inspect
import json
import os
import re
from types import MethodType

import torch

from experiments.b200_attention_gdn_serving.swiglu_native_output_worker import (
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT, write
from gleipnir.serving.sources import recorded_source_path


class DirectGdnNativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker(
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker
):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.serving_gdn_direct_caller import build_core_source
        from gleipnir.serving_gdn_direct_output import make_forward, validate_native

        condition = self.vllm_config.additional_config["serving_condition"]
        path = condition["gdn_direct_output_validation"]
        raw = (ROOT / path).read_bytes()
        receipt = json.loads(raw)
        validate_native(receipt)
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("direct GDN output hardware changed")
        for source, digest in receipt["sources"].items():
            if (
                hashlib.sha256(
                    recorded_source_path(ROOT, source).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError(f"direct GDN output source drift: {source}")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            ChunkGatedDeltaRule,
            QwenGatedDeltaNetAttention,
            l2norm_fwd,
        )

        original_core = QwenGatedDeltaNetAttention._forward_core
        original_source = inspect.getsource(original_core)
        generated = build_core_source(original_source)
        generated_path = (
            ROOT / "results/b200_attention_gdn_serving/native_gdn_direct_caller.py"
        )
        generated_path.write_text(generated)
        namespace = dict(original_core.__globals__)
        exec(compile(generated, str(generated_path), "exec"), namespace)
        direct_core = namespace["_forward_core"]
        callers = [
            layer
            for layer in self.model_runner.get_model().modules()
            if isinstance(layer, QwenGatedDeltaNetAttention)
        ]
        if len(callers) != 24:
            raise ValueError("incomplete direct GDN caller scope")
        for layer in callers:
            if layer._forward_core.__func__ is not original_core:
                raise ValueError("unsupported GDN caller override")
            layer._forward_core = MethodType(direct_core, layer)

        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "validation_path": path,
            "validation_sha256": hashlib.sha256(raw).hexdigest(),
            "sources": receipt["sources"],
            "calls": [],
            "operators": {},
            "output_copy": False,
            "arithmetic_changed": False,
            "caller_count": len(callers),
            "caller_original_sha256": hashlib.sha256(
                original_source.encode()
            ).hexdigest(),
            "caller_generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "direct_scope": (
                "ordinary prefill; original mixed/speculative/decode behavior"
            ),
        }
        seen = set()

        def observer(index):
            def observed(rows, direct):
                if torch.cuda.is_current_stream_capturing() or (index, direct) in seen:
                    return
                seen.add((index, direct))
                audit["calls"].append({"layer": index, "rows": rows, "direct": direct})
                audit["passed"] = (
                    len({c["layer"] for c in audit["calls"] if c["direct"]}) == 24
                )
                write("native_gdn_direct_output.json", audit)

            return observed

        operators = [
            (name, layer)
            for name, layer in self.model_runner.get_model().named_modules()
            if isinstance(layer, ChunkGatedDeltaRule)
        ]
        if len(operators) != 24:
            raise ValueError("incomplete direct GDN output module scope")
        for index, (name, layer) in enumerate(operators):
            if layer.gdn_prefill_backend != "flashinfer":
                raise ValueError(f"direct GDN output requires FlashInfer: {name}")
            layer._forward_method = make_forward(
                chunk_gated_delta_rule, l2norm_fwd, observer(index)
            )
            audit["operators"][name] = {
                "ordinal": index,
                "decoder_layer": int(re.search(r"layers\.(\d+)", name)[1]),
                "backend": "flashinfer",
                "caller_buffer_output": True,
            }
        self.precision["gdn_direct_output"] = {
            "operators": audit["operators"],
            "validation_sha256": audit["validation_sha256"],
            "output_copy": False,
            "arithmetic_changed": False,
        }
        self.audit_serving_state()
        write("native_gdn_direct_output.json", audit)
        print(
            "direct_gdn_output_installed layers=24 arithmetic_changed=false", flush=True
        )
