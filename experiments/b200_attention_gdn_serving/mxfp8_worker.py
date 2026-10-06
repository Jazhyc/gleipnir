"""Use native cuDNN MXFP8 prefill while retaining audited FP4 projections."""

import hashlib
import json
import os
from pathlib import Path

import torch

from experiments.b200_attention_gdn_serving.mixed_worker import MixedServingAuditWorker
from experiments.b200_attention_gdn_serving.worker import ROOT, write
from gleipnir.serving_mxfp8 import paged_forward


class Mxfp8ServingAuditWorker(MixedServingAuditWorker):
    def _install_attention_audit(self) -> None:
        import vllm.v1.attention.backends.flashinfer as backend

        path = ROOT / self.condition["mxfp8_validation"]
        receipt = json.loads(path.read_text())
        if (
            not receipt["arithmetic_passed"]
            or len(receipt["checks"]) != 7
            or receipt["batch_limit"] != 128
            or receipt["context_limit"] != 32768
            or any(
                not r["finite"] or not r["quantized_relative_l2"] <= 0.01
                for r in receipt["checks"]
            )
        ):
            raise ValueError("MXFP8 serving native validation failed")
        if (
            not receipt["passed"]
            and not self.condition["allow_finite_parity_diagnostic"]
        ):
            raise ValueError(
                "MXFP8 strict precision failed; diagnostic timing disabled"
            )
        for source, expected in receipt["sources"].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                raise ValueError(f"MXFP8 validation source drift: {source}")
        calls = []
        write(
            "native_attention.json",
            {"passed": False, "worker_pid": os.getpid(), "calls": []},
        )

        def forward(*args, **kwargs):
            if args:
                raise ValueError("MXFP8 serving expects the pinned keyword prefill API")
            value = None
            if len(calls) < 8 and not torch.cuda.is_current_stream_capturing():
                value = {
                    "query_shape": list(kwargs["query"].shape),
                    "cache_shape": list(kwargs["kv_cache"].shape),
                    "query_boundary_dtype": str(kwargs["query"].dtype),
                    "cache_dtype": str(kwargs["kv_cache"].dtype),
                    "core_payload_dtype": "torch.float8_e4m3fn",
                    "core_scale_format": "E8M0 per 32 elements",
                    "causal_alignment": "bottom_right",
                    "batch_size": kwargs["batch_size"],
                    "max_query_length": kwargs["max_q_len"],
                    "max_kv_length": kwargs["max_kv_len"],
                    "kernel": "cuDNN SdpaFwdDslSm100 D256 MXFP8 THD",
                }
            try:
                result = paged_forward(**kwargs)
            except BaseException as error:
                write(
                    "native_attention.json",
                    {
                        "passed": False,
                        "worker_pid": os.getpid(),
                        "failed_call": value,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                raise
            if value is not None:
                calls.append(value)
                if len(calls) == 8:
                    write(
                        "native_attention.json",
                        {
                            "passed": True,
                            "worker_pid": os.getpid(),
                            "serving_condition": self.condition,
                            "calls": calls,
                            "validation_sha256": hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest(),
                            "decode": "unchanged BF16 FlashInfer",
                            "conversion_cost_included": True,
                            "strict_native_precision_passed": receipt["passed"],
                            "diagnostic_only": not receipt["passed"],
                        },
                    )
            return result

        backend.trtllm_batch_context_with_kv_cache = forward
