"""Opt-in diagnostics around the unchanged cache-free mathematical helpers."""

from __future__ import annotations

import faulthandler
import functools
import hashlib
import json
import os
import signal
from pathlib import Path
from types import MethodType

from gleipnir.serving_operator_trace import OperatorTrace, PatchSet


def enable_trace(root: Path) -> None:
    """Initialize only in the GPU worker, after direct FROST bindings install."""
    from experiments.b200_attention_gdn_serving.prompt_only_worker import (
        PromptOnlyNativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker as Base,
    )

    original_load = Base.load_model

    def load(self, *, load_dummy_weights=False):
        original_load(self, load_dummy_weights=load_dummy_weights)
        directory = Path(os.environ["GLEIPNIR_PROMPT_ONLY_TRACE"]).resolve()
        if not directory.is_relative_to(root / "results"):
            raise ValueError("operator traces must be inside ignored project results")
        self._prompt_trace_directory = directory
        _install(self, root, os.environ.get("GLEIPNIR_PROMPT_ONLY_TRACE_MODE", "sync"))

    def control(self, mode=None):
        import torch

        if mode is not None:
            if mode not in {"off", "launch", "sync"}:
                raise ValueError("invalid operator trace mode")
            torch.cuda.synchronize()
            if self.model_runner.execute_model_state is not None:
                raise RuntimeError(
                    "drain model execution and sampling before toggling trace"
                )
            self._prompt_trace_patches.restore()
            if mode != "off":
                _install(self, root, mode)
            else:
                self._prompt_trace.record("disabled")
                self._prompt_trace.mode = "off"
        return {
            "pid": os.getpid(),
            "mode": self._prompt_trace.mode,
            "directory": str(self._prompt_trace_directory),
            "diagnostic_only": True,
        }

    Base.load_model = load
    Base.prompt_only_trace_state = control


def _install(worker, root: Path, mode: str) -> None:
    import flashinfer.gdn_prefill as gdn
    import torch
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qwen
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as conv

    import gleipnir.serving_prompt_only as prompt
    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.serving_fp4_swiglu_native_output import NativeOutputSwiGlu
    from gleipnir.serving_fp4_swiglu_overhead import OverheadSwiGlu

    trace = OperatorTrace(
        worker._prompt_trace_directory,
        mode,
        torch.cuda.synchronize,
        torch.cuda.is_current_stream_capturing,
    )
    patches = PatchSet()
    worker._prompt_trace, worker._prompt_trace_patches = trace, patches
    runner = worker.model_runner

    def describe(value):
        if isinstance(value, torch.Tensor):
            return {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "pointer": value.data_ptr(),
            }
        if hasattr(value, "codes"):
            return {
                k: describe(getattr(value, k)) for k in ("codes", "scales", "inverse")
            }
        return None

    def operator(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def call(*args, **kwargs):
            if trace.active.get() is None or trace.capturing():
                return original(*args, **kwargs)
            details = {str(i): d for i, v in enumerate(args) if (d := describe(v))}
            details.update({k: d for k, v in kwargs.items() if (d := describe(v))})
            return trace.invoke(
                label, lambda: original(*args, **kwargs), tensors=details
            )

        patches.set(owner, name, call)

    for owner, name, label in [
        (Nvfp4ScaledGemm, "__call__", "frost_gemm"),
        (NativeOutputSwiGlu, "__call__", "native_fp4_swiglu"),
        (OverheadSwiGlu, "__call__", "medium_swiglu"),
        (prompt, "packed_forward", "mxfp8_attention"),
        (conv, "causal_conv1d_fn", "causal_conv1d"),
        (qwen, "fused_post_conv_prep", "gdn_preparation"),
        (gdn, "chunk_gated_delta_rule", "gdn_prefill"),
    ]:
        operator(owner, name, label)

    original_build = prompt.PromptOnlyBuilder.build

    def build(builder, common_prefix_len, common_attn_metadata, fast_build=False):
        result = trace.invoke(
            "metadata",
            lambda: original_build(
                builder, common_prefix_len, common_attn_metadata, fast_build
            ),
        )
        if trace.active.get() is not None:
            lengths = common_attn_metadata.query_start_loc_cpu.diff().tolist()
            ids = list(runner.input_batch.req_ids)
            if len(ids) != len(lengths):
                raise ValueError("trace input ordering differs from attention metadata")
            trace.save_inputs(
                ids,
                [
                    runner.input_batch.token_ids_cpu[i, :n].tolist()
                    for i, n in enumerate(lengths)
                ],
            )
        return result

    patches.set(prompt.PromptOnlyBuilder, "build", build)
    for name, layer in runner.compilation_config.static_forward_context.items():
        if hasattr(layer, "_forward_core"):
            # Wrap the bound core without modifying the checksum-bound helper.
            original = layer._forward_core

            def core(this, *args, _original=original, _name=name, **kwargs):
                return trace.invoke(
                    _name + ".gdn_core", lambda: _original(*args, **kwargs)
                )

            patches.set(layer, "_forward_core", MethodType(core, layer))

    original_execute = runner.execute_model
    original_sample = runner.sample_tokens

    def execute(this, scheduler_output, *args, **kwargs):
        with trace.batch(scheduled=dict(scheduler_output.num_scheduled_tokens)):
            return trace.invoke(
                "model_execute",
                lambda: original_execute(scheduler_output, *args, **kwargs),
            )

    def sample(this, *args, **kwargs):
        with trace.batch(stage="sampling"):
            return trace.invoke("sampling", lambda: original_sample(*args, **kwargs))

    patches.set(runner, "execute_model", MethodType(execute, runner))
    patches.set(runner, "sample_tokens", MethodType(sample, runner))
    directory = worker._prompt_trace_directory
    worker._prompt_trace_stack = (directory / "python_stacks.txt").open("a")
    faulthandler.register(
        signal.SIGUSR2, file=worker._prompt_trace_stack, all_threads=True
    )
    receipt = {
        "pid": os.getpid(),
        "mode": mode,
        "diagnostic_only": True,
        "signal": "SIGUSR2",
        "sources": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                root / "src/gleipnir/__init__.py",
                root / "src/gleipnir/_compat.py",
                Path(__file__),
                root / "src/gleipnir/serving/operator_trace.py",
            ]
        },
    }
    (directory / "trace_ready.json").write_text(json.dumps(receipt, indent=2) + "\n")
    trace.record("ready", **receipt)
    print(f"prompt_only_trace_ready pid={os.getpid()} mode={mode}", flush=True)
