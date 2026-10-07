"""Correct the pinned Qwen kernel's Torch/Triton mutation-analysis ABI.

Runtime scalars stay runtime parameters in TTIR; only genuine constexpr/None
arguments are constants. Scalars are filtered out after mutation tracing.
The kernel definition, runtime launch and arithmetic remain unchanged.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

KERNEL = "_fused_qk_rmsnorm_rope_gate_kernel"
MODULE_SHA256 = "3703af53dfef0c7e53035da2bee03ecada7fbd7698b40b05d6fd1ef9cf5077a6"
TORCH_SHA256 = "0486ea520f0c5b012d8d27adc3e6957a7cf47ac217fc7c5fd24f8a7c30175e6c"
OUTPUTS = {"q_out_ptr", "k_out_ptr", "gate_out_ptr"}


def adapt_analysis_source(source: str) -> str:
    """Keep positional scalar slots and construct a truthful constexpr map."""
    old_names = "get_tensor_names(name, arg) for name, arg in ordered_args.items()"
    new_names = (
        "(get_tensor_names(name, arg) or [name])\n"
        "            for index, (name, arg) in enumerate(ordered_args.items())\n"
        "            if not kernel.params[index].is_constexpr and arg is not None"
    )
    old_constants = (
        "name: arg for name, arg in ordered_args.items() if not is_tensor_like_arg(arg)"
    )
    new_constants = (
        "name: arg\n"
        "        for index, (name, arg) in enumerate(ordered_args.items())\n"
        "        if kernel.params[index].is_constexpr or arg is None"
    )
    for old in [old_names, old_constants]:
        if source.count(old) != 1:
            raise ValueError("pinned Torch TTIR analysis source changed")
    return source.replace(old_names, new_names).replace(old_constants, new_constants)


def is_target(kernel: Any) -> bool:
    """Recognize the source-validated JIT kernel, including an autotuner wrapper."""
    value = kernel
    while hasattr(value, "fn"):
        value = value.fn
    return getattr(value, "__name__", None) == KERNEL


def install(root: Path, *, validation: str | None = None) -> dict[str, Any]:
    """Repair analysis only for the known kernel; preserve other fallbacks."""
    import os

    import torch
    import torch._higher_order_ops.triton_kernel_wrap as wrap
    import torch._inductor.ir
    import triton
    from vllm.model_executor.layers import fused_qk_norm_rope as module

    if torch.__version__.split("+")[0] != "2.11.0" or triton.__version__ != "3.7.1":
        raise ValueError("mutation fix requires validated Torch 2.11/Triton 3.7.1")
    if state := getattr(wrap, "_gleipnir_qk_mutation_analysis", None):
        return state["record"]
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != MODULE_SHA256:
        raise ValueError("pinned fused QK/RoPE source changed")
    if hashlib.sha256(Path(wrap.__file__).read_bytes()).hexdigest() != TORCH_SHA256:
        raise ValueError("pinned Torch mutation-analysis source changed")
    helper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if validation is not None:
        receipt = json.loads((root / validation).read_text())
        if not receipt.get("passed") or receipt["helper_sha256"] != helper_sha:
            raise ValueError("mutation validation/source drift")
    original_generate, original_identify = (
        wrap.generate_ttir,
        wrap.identify_mutated_tensors,
    )
    adapted = adapt_analysis_source(inspect.getsource(original_generate))
    digest = hashlib.sha256(adapted.encode()).hexdigest()
    out = root / "results/b200_mutation_analysis/generated"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"ttir_analysis_{digest}.py"
    target.write_text(adapted)
    namespace = dict(wrap.__dict__)
    exec(compile(adapted, str(target), "exec"), namespace)
    repaired_generate = namespace["generate_ttir"]

    def generate(kernel, kwargs, tma_descriptor_metadata):
        function = repaired_generate if is_target(kernel) else original_generate
        return function(kernel, kwargs, tma_descriptor_metadata)

    def identify(kernel, kwargs, tma_descriptor_metadata):
        result = original_identify(kernel, dict(kwargs), tma_descriptor_metadata)
        if not is_target(kernel):
            return result
        # Trace address operands with positional scalar slots intact, then
        # discard scalar results because their values cannot be mutated.
        return [
            name
            for name in result
            if name in kwargs
            and isinstance(kwargs[name], (torch.Tensor, torch._inductor.ir.TensorBox))
        ]

    wrap.generate_ttir, wrap.identify_mutated_tensors = generate, identify
    record = {
        "worker_pid": os.getpid(),
        "helper_sha256": helper_sha,
        "original_module_sha256": MODULE_SHA256,
        "torch_source_sha256": TORCH_SHA256,
        "adapted_analysis_sha256": digest,
        "kernel_runtime_unchanged": True,
        "scope": KERNEL,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "validation": validation,
    }
    wrap._gleipnir_qk_mutation_analysis = {
        "record": record,
        "original_generate": original_generate,
        "original_identify": original_identify,
        "generate": generate,
        "identify": identify,
    }
    (out / f"installed_{os.getpid()}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print("qk_mutation_analysis_installed", record, flush=True)
    return record


@contextmanager
def original_analysis():
    """Native-only ablation; never toggle analysis inside a serving process."""
    import torch._higher_order_ops.triton_kernel_wrap as wrap

    state = wrap._gleipnir_qk_mutation_analysis
    wrap.generate_ttir = state["original_generate"]
    wrap.identify_mutated_tensors = state["original_identify"]
    try:
        yield
    finally:
        wrap.generate_ttir = state["generate"]
        wrap.identify_mutated_tensors = state["identify"]
