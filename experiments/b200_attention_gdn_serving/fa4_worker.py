"""Bounded audit of vLLM's native causal, paged FlashAttention 4 dispatch."""

import functools
import os

import torch

from experiments.b200_attention_gdn_serving.worker import ServingAuditWorker, write


class Fa4ServingAuditWorker(ServingAuditWorker):
    def _install_attention_audit(self) -> None:
        import vllm.v1.attention.backends.flash_attn as backend

        original = backend.flash_attn_varlen_func
        calls = []
        if self.condition["attention_precision"] != "bf16":
            raise ValueError("this installed Blackwell FA4 backend supports BF16")
        write(
            "native_attention.json",
            {"passed": False, "worker_pid": os.getpid(), "calls": []},
        )

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                return original(*args, **kwargs)
            q, k, v = [
                kwargs[name] if name in kwargs else args[index]
                for index, name in enumerate(("q", "k", "v"))
            ]
            value = {
                "query_dtype": str(q.dtype),
                "cache_dtype": str(k.dtype),
                "value_dtype": str(v.dtype),
                "query_shape": list(q.shape),
                "cache_shape": list(k.shape),
                "value_shape": list(v.shape),
                "causal": kwargs.get("causal"),
                "fa_version": kwargs.get("fa_version"),
                "paged": kwargs.get("block_table") is not None,
                "kernel": "vllm.vllm_flash_attn.cute.interface._flash_attn_fwd",
            }
            if (
                value["fa_version"] != 4
                or not value["causal"]
                or not value["paged"]
                or any(x.dtype != torch.bfloat16 for x in (q, k, v))
                or q.shape[-2:] != (16, 256)
                or k.shape[-2:] != (4, 256)
            ):
                raise ValueError(f"native FA4 attention dispatch mismatch: {value}")
            result = original(*args, **kwargs)
            calls.append(value)
            if len(calls) == 8:
                backend.flash_attn_varlen_func = original
                write(
                    "native_attention.json",
                    {
                        "passed": True,
                        "worker_pid": os.getpid(),
                        "serving_condition": self.condition,
                        "calls": calls,
                    },
                )
            return result

        backend.flash_attn_varlen_func = observed
