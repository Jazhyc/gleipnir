"""Checksum-bound audit of every Meta blog technique and actual source guards."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta", type=Path, default=Path("/tmp/gleipnir-ads-kernels/lp_fa4")
    )
    parser.add_argument(
        "--nvidia", type=Path, default=Path("/tmp/gleipnir-cudnn-survey/python/cudnn")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.meta / "src/lp_fa4/cute/flash_bwd_sm100.py",
        args.meta / "src/lp_fa4/cute/flash_fwd_sm100.py",
        args.nvidia / "sdpa/fwd/kernels/sm100/prefill_d256_mxfp8.py",
        args.nvidia / "sdpa/bwd/kernels/sm100/bprop_dq_d256_mxfp8.py",
        args.nvidia / "sdpa/bwd/kernels/sm100/bprop_dkdv_d256_mxfp8.py",
        args.nvidia / "gated_attention_block/kernels/proj_gemm.py",
        args.nvidia / "gated_attention_block/api.py",
    ]
    sources = {p.name: p.read_text() for p in paths}
    tree = ast.parse(sources["flash_bwd_sm100.py"])
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FlashAttentionBackwardSm100"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    block = next(
        n
        for n in init.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "blockscaled"
    )
    actual_guards = ast.Module(body=block.body, type_ignores=[])
    code = compile(ast.fix_missing_locations(actual_guards), str(paths[0]), "exec")
    defaults = dict(
        head_dim=128,
        head_dim_v=128,
        tile_m=128,
        tile_n=128,
        qhead_per_kvhead=1,
        cluster_size=1,
        use_2cta_instrs=False,
        is_persistent=False,
        is_causal=False,
        is_local=False,
        deterministic=False,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        has_aux_tensors=False,
    )
    guards = []
    for name, changes in [
        ("supported_control", {}),
        ("d256", {"head_dim": 256, "head_dim_v": 256}),
        ("gqa4", {"qhead_per_kvhead": 4}),
        ("causal", {"is_causal": True}),
        (
            "qwen",
            {
                "head_dim": 256,
                "head_dim_v": 256,
                "qhead_per_kvhead": 4,
                "is_causal": True,
            },
        ),
    ]:
        scope = {**defaults, **changes}
        try:
            exec(code, scope)
            guards.append({"case": name, "accepted": True})
        except AssertionError:
            guards.append({"case": name, "accepted": False})
    assert [g["accepted"] for g in guards] == [True, False, False, False, False]
    techniques = [
        (
            "persistent/jagged forward scheduling",
            "already_present",
            "THD_PERSISTENT = True",
        ),
        (
            "KV pipeline reordering/unroll",
            "different native pipeline; Meta port blocked by D256/GQA/causal guards",
            "mb_bmm2_ready",
        ),
        (
            "online probability conversion",
            "fixed P scales present; test scaled-forward variants",
            "SF_CONST_VALUE = 0x7F",
        ),
        (
            "compact jagged data / aligned scales",
            "already_present",
            "_thd_sf_tile_bases",
        ),
        ("TMEM allocation/scale aliasing", "already_present", "SF_TMEM_COLS"),
        (
            "square online dS quantization",
            "warp-amax prototype; separate native kernels cannot reuse one dS payload",
            "group_amax_0",
        ),
        (
            "FP16 global dQ reduction",
            "inapplicable: native dQ accumulates in TMEM and stores once",
            "store_num_bits_per_copy",
        ),
        ("square Q/K/dO payload", "implemented_previous_screen", ""),
        (
            "RMSNorm plus quantization",
            "new training-safe head-norm/RoPE producer pilot",
            "",
        ),
        (
            "GEMM plus quantization",
            "new BF16/MXFP8 projection epilogue pilots with FP32 LoRA add",
            "",
        ),
        (
            "jagged projection/norm backward with FP8 gradients",
            "head norm backward fused; native FP8 projection backward unavailable",
            "",
        ),
        (
            "custom end-to-end module",
            "test Qwen producer integration; Meta cross-attention graph differs",
            "",
        ),
    ]
    checks = []
    for technique, status, token in techniques:
        matches = [
            {"file": p.name, "line": i + 1}
            for p in paths
            for i, line in enumerate(p.read_text().splitlines())
            if token and token in line
        ]
        if token and not matches:
            raise ValueError(f"missing source evidence: {technique}")
        checks.append(
            {"technique": technique, "status": status, "source_evidence": matches[:8]}
        )
    # Execute the actual pure-Python fused-projection validator without imports.
    ptree = ast.parse(sources["proj_gemm.py"])
    validator = next(
        n
        for n in ptree.body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_norm_rope_params"
    )
    validator.args.args[0].annotation = None
    scope = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[validator], type_ignores=[])),
            "projection_guard",
            "exec",
        ),
        scope,
    )
    from types import SimpleNamespace

    params = SimpleNamespace(
        d_head=256,
        h_q=16,
        h_kv=4,
        rope_dim=64,
        eps=1e-6,
        norm_source="ldg_early",
        quant_fp8=False,
        quant_mxfp8=True,
        want_rstd=True,
        qk_norm=True,
    )
    try:
        scope["validate_norm_rope_params"](params)
        raise AssertionError(
            "upstream unexpectedly supports fused MXFP8 training stats"
        )
    except ValueError as error:
        projection_decline = str(error)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "complete",
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "meta_constructor_guards": guards,
        "upstream_projection_training_decline": projection_decline,
        "techniques": checks,
        "gpu_execution": False,
    }
    for p in paths:
        (args.output / p.name).write_bytes(p.read_bytes())
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "guards": guards,
                "projection_decline": projection_decline,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
