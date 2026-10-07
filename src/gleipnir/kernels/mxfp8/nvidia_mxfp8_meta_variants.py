"""Scoped native scheduling and online dS experiments; never change defaults."""

from __future__ import annotations

import hashlib
import importlib
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch


@dataclass(frozen=True)
class NativeVariant:
    """Explicit compilation identity for each bounded native probe."""

    persistent_dq: bool = False
    persistent_dkdv: bool = False
    ds_warp_amax: bool = False
    dq_store_bits: int = 16
    forward_p_scale_log2: int = 0

    def __post_init__(self) -> None:
        if self.dq_store_bits not in (16, 128):
            raise ValueError("dQ store width must be 16 or 128")
        if self.forward_p_scale_log2 not in (0, 4, 8):
            raise ValueError("forward P scale must be 0, 4 or 8")


def probability_source(source: str, log2_scale: int) -> str:
    """Static P scale with zero rescale threshold to keep payloads below 448.

    Reduction and LSE still see unscaled probabilities. Scaling only the MMA
    payload and its E8M0 metadata preserves the mathematical PV contraction.
    """
    if log2_scale not in (4, 8):
        raise ValueError("probability source expects log2 scale 4 or 8")
    init = "CFG, _TMA = make_cfg_d256_mxfp8(PARAMS)"
    scale = "SF_CONST_VALUE = 0x7F"
    if source.count(init) != 1 or source.count(scale) != 1:
        raise ValueError("forward probability source changed")
    source = source.replace(
        init,
        init
        + "\nfrom dataclasses import replace as _gleipnir_replace"
        + "\nCFG = _gleipnir_replace(CFG, RESCALE_THRESHOLD=0.0)",
    )
    source = source.replace(scale, f"SF_CONST_VALUE = {127 - log2_scale}")
    source, count = re.subn(
        r"(chunk_P_[A-Za-z0-9_]+)\.to\(STORAGE_DTYPE\)",
        rf"(\1 * cutlass.Float32({float(2**log2_scale)})).to(STORAGE_DTYPE)",
        source,
    )
    if count != 16:
        raise ValueError(
            f"forward P conversion source changed: found {count}, expected 16"
        )
    return source


@lru_cache(maxsize=8)
def forward_variant(hq, hk, device, log2_scale):
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl

    from gleipnir.nvidia_mxfp8_varlen_attention import forward_plan

    original_loader = api_dsl._load_sm100_kernel_module

    def loader(flavor, params, **kwargs):
        original = original_loader(flavor, params, **kwargs)
        if flavor != (256, 256) or kwargs.get("pertensor") or kwargs.get("rubin"):
            raise ValueError("probability variant requires SM100 block-scaled D256")
        source = probability_source(Path(original.__file__).read_text(), log2_scale)
        digest = hashlib.sha256(source.encode()).hexdigest()
        path = (
            Path(".cache/kernels/nvidia_mxfp8/generated") / f"probability_{digest}.py"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(source)
        return load_template(
            str(path.resolve()), params, tag="gleipnir_probability_" + digest
        )

    with patch.object(api_dsl, "_load_sm100_kernel_module", loader):
        return forward_plan.__wrapped__(hq, hk, device)


def warp_amax_source(source: str, *, dq: bool) -> str:
    """Add warp-shared maxima before scale conversion; fail on source drift.

    This probes Meta's warp reduction principle independently in NVIDIA's two
    backward kernels. It does not claim their fragments are Meta's exact 32x32
    geometry or share a dS payload between the separate launches.
    """
    coordinate = "tTR_cDP" if dq else "tTR_cVDO"
    marker = f"                dS_row = cute.get({coordinate}[0], mode=[0])"
    if source.count(marker) != 1:
        raise ValueError("online dS source changed; refuse ambiguous replacement")
    names = ("group_amax_0", "group_amax_1") if dq else ("group_amax", "group_amax_1")
    lines = [
        f'                {name} = cute.arch.warp_redux_sync({name}, kind="fmax")'
        for name in names
    ]
    return source.replace(marker, "\n".join(lines) + "\n" + marker)


@lru_cache(maxsize=1)
def patched_classes() -> tuple:
    """Keep immutable upstream files; materialize checksum-named derived modules."""
    classes = []
    for dq, stem, name in (
        (True, "bprop_dq_d256_mxfp8", "BlackwellFmhaBackwardDQ256"),
        (False, "bprop_dkdv_d256_mxfp8", "BlackwellFmhaBackwardDKDV256"),
    ):
        original = importlib.import_module("cudnn.sdpa.bwd.kernels.sm100." + stem)
        source = warp_amax_source(Path(original.__file__).read_text(), dq=dq)
        digest = hashlib.sha256(source.encode()).hexdigest()
        path = Path(".cache/kernels/nvidia_mxfp8/generated") / f"{stem}_{digest}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != source:
            raise ValueError("derived native source checksum collision")
        if not path.exists():
            path.write_text(source)
        module_name = "cudnn.sdpa.bwd.kernels.sm100.gleipnir_" + stem + "_" + digest
        spec = importlib.util.spec_from_file_location(module_name, path.resolve())
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        classes.append(getattr(module, name))
    return tuple(classes)


@lru_cache(maxsize=32)
def backward_variant(heads: tuple[int, int], variant: NativeVariant):
    from cudnn.frost.compiled_cache import positional_entry

    from gleipnir.nvidia_mxfp8_attention import UPSTREAM_REVISION
    from gleipnir.nvidia_mxfp8_varlen_host import compile_packed_backward

    classes = patched_classes() if variant.ds_warp_amax else None
    paths = [Path(__file__), Path(__file__).with_name("nvidia_mxfp8_varlen_host.py")]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
    identity = (
        variant.persistent_dq,
        variant.persistent_dkdv,
        variant.ds_warp_amax,
        variant.dq_store_bits,
    )
    key = f"{UPSTREAM_REVISION}:{digest}:{heads}:{identity}"
    owner = compile_packed_backward(
        heads,
        key,
        persistent_dq=variant.persistent_dq,
        persistent_dkdv=variant.persistent_dkdv,
        dq_store_bits=variant.dq_store_bits,
        kernel_classes=classes,
    )
    fn = positional_entry(owner)
    if fn is None:
        raise RuntimeError("native variant requires TVM FFI")
    return owner, fn


@contextmanager
def native_variant(variant: NativeVariant):
    """Patch only the explicit experimental backend for the context lifetime."""
    from contextlib import ExitStack

    import gleipnir.nvidia_mxfp8_fused_attention as attention

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                attention,
                "backward_plan",
                lambda hq, hk: backward_variant((hq, hk), variant),
            )
        )
        if variant.forward_p_scale_log2:
            stack.enter_context(
                patch.object(
                    attention,
                    "forward_plan",
                    lambda hq, hk, device: forward_variant(
                        hq, hk, device, variant.forward_p_scale_log2
                    ),
                )
            )
        yield asdict(variant)
