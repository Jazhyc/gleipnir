"""Install native-admitted direct GDN destinations on the Direct FP4 stack."""

import hashlib
import json
import os
import re

import torch

from experiments.b200_attention_gdn_serving.swiglu_native_output_worker import (
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT, write


class DirectGdnNativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker(
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker
):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.serving_gdn_direct_output import make_forward, validate_native

        condition = self.vllm_config.additional_config["serving_condition"]
        path = condition["gdn_direct_output_validation"]
        raw = (ROOT / path).read_bytes()
        receipt = json.loads(raw)
        validate_native(receipt)
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("direct GDN output hardware changed")
        for source, digest in receipt["sources"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                raise ValueError(f"direct GDN output source drift: {source}")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            ChunkGatedDeltaRule,
            l2norm_fwd,
        )

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
