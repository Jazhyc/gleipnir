"""Validate config-derived MXFP8 serving attention at a served model's heads.

The 4B canary (`experiments/b200_attention_gdn_serving/mxfp8_canary.py`) with
query/KV head counts read from ``--model``: producer exactness against the
training producer, the FP32 arithmetic oracle (MXFP8 Q/K/V, online FP8 P)
and a BF16 FP32 reference on the same five batch/length mixes, plus CUDA-graph
capture and replay with updated lengths and pages (7 checks, same receipt
schema). It also requires ``mxfp8_geometry`` at the 4B shape (16/4) to be
bitwise equal to the original hash-bound module, so the copy is exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LOG2E = 1.4426950408889634


def attention_shape(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    if int(text["head_dim"]) != 256:
        raise ValueError("MXFP8 serving attention requires head dimension 256")
    return {
        "query_heads": int(text["num_attention_heads"]),
        "kv_heads": int(text["num_key_value_heads"]),
        "head_dim": 256,
    }


def quantized_attention(q, k, v):
    """The 4B FP32 oracle (`mxfp8_reference.attention`) for any GQA ratio."""
    import torch

    from experiments.b200_attention_gdn_serving.mxfp8_reference import columns, rows

    heads, group = q.shape[1], q.shape[1] // k.shape[1]
    query, key, value = rows(q), rows(k), columns(v)
    key = key.repeat_interleave(group, 1)
    value = value.repeat_interleave(group, 1)
    score = torch.einsum("qhd,khd->hqk", query, key) * (LOG2E / 16)
    qn, kn = q.shape[0], k.shape[0]
    mask = (
        torch.arange(kn, device=q.device)[None, :]
        > torch.arange(qn, device=q.device)[:, None] + kn - qn
    )
    score.masked_fill_(mask[None], float("-inf"))
    m = torch.full((heads, qn), float("-inf"), device=q.device)
    denominator = torch.zeros_like(m)
    accumulator = torch.zeros((heads, qn, 256), device=q.device)
    for start in range(0, kn, 128):
        tile = score[:, :, start : start + 128]
        current = tile.amax(-1)
        next_m = torch.where(current - m > 4.0, current, m)
        alpha = torch.where(torch.isfinite(next_m), torch.exp2(m - next_m), 1.0)
        p = torch.where(
            torch.isfinite(next_m[..., None]), torch.exp2(tile - next_m[..., None]), 0.0
        )
        fp8_p = p.to(torch.float8_e4m3fn).float()
        accumulator = accumulator * alpha[..., None] + torch.einsum(
            "hqk,khd->hqd", fp8_p, value[start : start + 128]
        )
        denominator = denominator * alpha + p.sum(-1)
        m = next_m
    return (accumulator / denominator[..., None]).permute(1, 0, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from gleipnir.nvidia_mxfp8_fused_quantize import prepare
    from gleipnir.serving import mxfp8 as base
    from gleipnir.serving.mxfp8 import produce
    from gleipnir.serving.mxfp8_geometry import forward_plan, paged_forward

    shape = attention_shape(args.model)
    heads, kv_heads = shape["query_heads"], shape["kv_heads"]
    group = heads // kv_heads
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__),
        Path("src/gleipnir/serving/mxfp8.py"),
        Path("src/gleipnir/serving/mxfp8_geometry.py"),
        Path("src/gleipnir/serving/mxfp8_source.py"),
        Path("src/gleipnir/kernels/mxfp8/nvidia_mxfp8_attention.py"),
        Path("src/gleipnir/kernels/mxfp8/nvidia_mxfp8_fused_quantize.py"),
        Path("experiments/b200_attention_gdn_serving/mxfp8_reference.py"),
    ]
    native = Path(".cache/kernels/nvidia_mxfp8/frontend/cudnn/sdpa/fwd")
    sources.extend(
        [
            native / "api_dsl.py",
            native / "config_sm100.py",
            native / "kernels/sm100/prefill_d256_mxfp8.py",
        ]
    )
    sources = [p for p in sources if p.exists()]
    receipt = {
        "passed": False,
        "relative_l2_limit": 0.05,
        "quantized_relative_l2_limit": 0.01,
        "arithmetic_passed": False,
        "precision": "mxfp8",
        "attention_shape": shape,
        "checks": [],
        "base_equivalence": None,
        "sources": {
            str(p.relative_to(Path.cwd()) if p.is_absolute() else p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        "gpu": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "batch_limit": base.MAX_BATCH,
        "context_limit": base.MAX_LENGTH,
        "kv_metadata_contract": "TRTLLM cumulative pages plus exact token lengths",
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")

    def cumulative(values: list[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *torch.tensor(values).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

    def case(queries, histories, q_heads, k_heads):
        batch = len(queries)
        pages_needed = [(n + 15) // 16 for n in histories]
        pages = sum(pages_needed)
        cache = torch.randn(
            (pages, 2, k_heads, 16, 256), dtype=torch.bfloat16, device="cuda"
        )
        table_cpu = torch.zeros((batch, max(pages_needed)), dtype=torch.int32)
        permutation = torch.randperm(pages)
        cursor = 0
        for row, count in enumerate(pages_needed):
            table_cpu[row, :count] = permutation[cursor : cursor + count]
            cursor += count
        q = torch.randn(
            (sum(queries), q_heads, 256), dtype=torch.bfloat16, device="cuda"
        )
        return {
            "query": q,
            "kv_cache": cache,
            "block_tables": table_cpu.cuda(),
            "cum_seq_lens_q": cumulative(queries),
            "cum_seq_lens_kv": cumulative(pages_needed),
            "seq_lens": torch.tensor(histories, dtype=torch.int32, device="cuda"),
            "max_q_len": max(queries),
            "max_kv_len": max(histories),
            "batch_size": batch,
            "out": torch.empty_like(q),
        }

    def packed(which, lengths, table):
        return torch.cat(
            [
                which[table[i, : (n + 15) // 16].long()]
                .permute(0, 2, 1, 3)
                .flatten(0, 1)[:n]
                for i, n in enumerate(lengths)
            ]
        )

    def oracle(kwargs, queries, histories, quantized=False):
        key, value = kwargs["kv_cache"].unbind(1)
        table, q = kwargs["block_tables"], kwargs["query"]
        result, offset = [], 0
        for row, (qn, kn) in enumerate(zip(queries, histories, strict=True)):
            indexes = table[row, : (kn + 15) // 16].long()
            k = key[indexes].permute(0, 2, 1, 3).flatten(0, 1)[:kn]
            v = value[indexes].permute(0, 2, 1, 3).flatten(0, 1)[:kn]
            if quantized:
                result.append(quantized_attention(q[offset : offset + qn], k, v))
            else:
                k = k.repeat_interleave(group, 1).float()
                v = v.repeat_interleave(group, 1).float()
                score = torch.einsum("qhd,khd->hqk", q[offset : offset + qn].float(), k)
                score = score / 16
                mask = (
                    torch.arange(kn, device="cuda")[None, :]
                    > torch.arange(qn, device="cuda")[:, None] + kn - qn
                )
                score.masked_fill_(mask[None], float("-inf"))
                result.append(torch.einsum("hqk,khd->qhd", score.softmax(-1), v))
            offset += qn
        return torch.cat(result)

    def errors(kwargs, queries, histories):
        output = kwargs["out"].float()
        expected = oracle(kwargs, queries, histories)
        quantized = oracle(kwargs, queries, histories, quantized=True)
        return {
            "relative_l2": ((output - expected).norm() / expected.norm()).item(),
            "quantized_relative_l2": (
                (output - quantized).norm() / quantized.norm()
            ).item(),
            "finite": bool(torch.isfinite(kwargs["out"]).all()),
        }

    def forward(kwargs):
        return paged_forward(query_heads=heads, kv_heads=kv_heads, **kwargs)

    save()
    precision_ok = True
    try:
        # The copy at the 4B shape must reproduce the hash-bound module exactly.
        equivalence = []
        for queries, histories in (([1, 17, 129], [511, 129, 4097]), ([129], [8192])):
            kwargs = case(queries, histories, 16, 4)
            base.paged_forward(**kwargs)
            original = kwargs["out"].clone()
            paged_forward(query_heads=16, kv_heads=4, **kwargs)
            torch.cuda.synchronize()
            equivalence.append(bool(torch.equal(original, kwargs["out"])))
        receipt["base_equivalence"] = {"shape": [16, 4], "bitwise": equivalence}
        save()
        assert all(equivalence), receipt["base_equivalence"]
        for queries, histories in (
            ([1, 3, 127, 129], [1, 3, 127, 129]),
            ([1, 17, 129], [511, 129, 4097]),
            (
                [1 if i % 2 else 17 for i in range(128)],
                [17 + i % 4 * 16 for i in range(128)],
            ),
            ([129], [32768]),
            ([17, 1], [129, 511]),
        ):
            kwargs = case(queries, histories, heads, kv_heads)
            key, value = kwargs["kv_cache"].unbind(1)
            table = kwargs["block_tables"]
            cuk = cumulative(histories)
            # Payload and every live packed SF atom against the validated
            # training producer, at this model's head counts.
            for source, cu, maximum, column, paged in (
                (kwargs["query"], kwargs["cum_seq_lens_q"], max(queries), False, False),
                (key, cuk, max(histories), False, True),
                (value, cuk, max(histories), True, True),
            ):
                lengths = histories if paged else queries
                dense = packed(source, histories, table) if paged else source
                payload, sf = produce(
                    source,
                    cu,
                    maximum,
                    table=table if paged else None,
                    total=len(queries) * maximum if paged else None,
                    column=column,
                )
                ref = prepare(dense, cu, maximum)
                expected_payload, expected_sf = (
                    (ref[1], ref[6]) if column else (ref[0], ref[5])
                )
                tiles = sum((n + 127) // 128 for n in lengths)
                count = dense.shape[1]
                assert torch.equal(payload[: dense.shape[0]], expected_payload)
                assert torch.equal(
                    sf.reshape(count, -1, 1024)[:, :tiles],
                    expected_sf.reshape(count, -1, 1024)[:, :tiles],
                )
            forward(kwargs)
            torch.cuda.synchronize()
            check = {
                "queries": queries,
                "histories": histories,
                "mode": "eager",
                "producer_exact": True,
                **errors(kwargs, queries, histories),
            }
            receipt["checks"].append(check)
            save()
            precision_ok = precision_ok and check["relative_l2"] <= 0.05
            assert check["finite"] and check["quantized_relative_l2"] <= 0.01, check
            if queries == [17, 1]:
                forward(kwargs)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward(kwargs)
                for mode in ("capture", "updated_lengths_and_pages"):
                    q_lengths, k_lengths = queries, histories
                    if mode == "updated_lengths_and_pages":
                        q_lengths, k_lengths = [1, 17], [511, 129]
                        kwargs["cum_seq_lens_q"].copy_(cumulative(q_lengths))
                        kwargs["seq_lens"].copy_(
                            torch.tensor(k_lengths, dtype=torch.int32, device="cuda")
                        )
                        kwargs["cum_seq_lens_kv"].copy_(cumulative([32, 9]))
                        table.copy_(table.flip(0))
                    graph.replay()
                    torch.cuda.synchronize()
                    check = {"mode": mode, **errors(kwargs, q_lengths, k_lengths)}
                    receipt["checks"].append(check)
                    save()
                    precision_ok = precision_ok and check["relative_l2"] <= 0.05
                    assert check["finite"] and check["quantized_relative_l2"] <= 0.01, (
                        check
                    )
        receipt["passed"] = precision_ok
        receipt["arithmetic_passed"] = True
        receipt["diagnostic_only"] = not precision_ok
        receipt["workspace_bytes"] = forward_plan(
            torch.device("cuda:0"), heads, kv_heads, base.MAX_LENGTH
        ).scratch_workspace_bytes()
        save()
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
