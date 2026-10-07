"""Validate native MXFP8 serving, cache gather, asymmetric lengths and replay."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from experiments.b200_attention_gdn_serving.mxfp8_reference import attention
    from gleipnir.nvidia_mxfp8_fused_quantize import prepare
    from gleipnir.serving_mxfp8 import forward_plan, paged_forward, produce

    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__),
        Path("src/gleipnir/serving/mxfp8.py"),
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
        "checks": [],
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "gpu": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "batch_limit": 128,
        "context_limit": 32768,
        "kv_metadata_contract": "TRTLLM cumulative pages plus exact token lengths",
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        archive = args.output.parent / (args.output.stem + "_sources")
        for p in sources:
            target = archive / p.resolve().relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())

    save()
    precision_ok = True
    try:
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
            batch = len(queries)
            pages_needed = [(n + 15) // 16 for n in histories]
            pages = sum(pages_needed)
            cache = torch.randn(
                (pages, 2, 4, 16, 256), dtype=torch.bfloat16, device="cuda"
            )
            key, value = cache.unbind(1)
            table_cpu = torch.zeros((batch, max(pages_needed)), dtype=torch.int32)
            permutation = torch.randperm(pages)
            cursor = 0
            for row, count in enumerate(pages_needed):
                table_cpu[row, :count] = permutation[cursor : cursor + count]
                cursor += count
            table = table_cpu.cuda()
            q = torch.randn(
                (sum(queries), 16, 256), dtype=torch.bfloat16, device="cuda"
            )
            cuq = torch.tensor(
                [0, *torch.tensor(queries).cumsum(0).tolist()],
                dtype=torch.int32,
                device="cuda",
            )
            cuk = torch.tensor(
                [0, *torch.tensor(histories).cumsum(0).tolist()],
                dtype=torch.int32,
                device="cuda",
            )
            lengths = torch.tensor(histories, dtype=torch.int32, device="cuda")
            page_offsets = torch.tensor(
                [0, *torch.tensor(pages_needed).cumsum(0).tolist()],
                dtype=torch.int32,
                device="cuda",
            )
            output = torch.empty_like(q)
            kwargs = dict(
                query=q,
                kv_cache=cache,
                block_tables=table,
                cum_seq_lens_q=cuq,
                cum_seq_lens_kv=page_offsets,
                seq_lens=lengths,
                max_q_len=max(queries),
                max_kv_len=max(histories),
                batch_size=batch,
                out=output,
            )

            def packed_cache(which, lengths=histories, pages_table=table):
                return torch.cat(
                    [
                        which[pages_table[i, : (n + 15) // 16].long()]
                        .permute(0, 2, 1, 3)
                        .flatten(0, 1)[:n]
                        for i, n in enumerate(lengths)
                    ]
                )

            def oracle(
                q_lengths=queries,
                k_lengths=histories,
                pages_table=table,
                q_tensor=q,
                key_tensor=key,
                value_tensor=value,
                quantized=False,
            ):
                result = []
                offset = 0
                for row, (qn, kn) in enumerate(zip(q_lengths, k_lengths, strict=True)):
                    indexes = pages_table[row, : (kn + 15) // 16].long()
                    k = key_tensor[indexes].permute(0, 2, 1, 3).flatten(0, 1)[:kn]
                    v = value_tensor[indexes].permute(0, 2, 1, 3).flatten(0, 1)[:kn]
                    if quantized:
                        result.append(attention(q_tensor[offset : offset + qn], k, v))
                        offset += qn
                        continue
                    k, v = (
                        k.repeat_interleave(4, 1).float(),
                        v.repeat_interleave(4, 1).float(),
                    )
                    score = (
                        torch.einsum(
                            "qhd,khd->hqk", q_tensor[offset : offset + qn].float(), k
                        )
                        / 16
                    )
                    mask = (
                        torch.arange(kn, device="cuda")[None, :]
                        > torch.arange(qn, device="cuda")[:, None] + kn - qn
                    )
                    score.masked_fill_(mask[None], float("-inf"))
                    result.append(torch.einsum("hqk,khd->qhd", score.softmax(-1), v))
                    offset += qn
                return torch.cat(result)

            # Compare payload and every live packed SF atom against the already
            # validated training producer. Its backward buffers are diagnostic only.
            valid_tiles_q = sum((n + 127) // 128 for n in queries)
            valid_tiles_k = sum((n + 127) // 128 for n in histories)
            for source, cu, maximum, column, cache_source, tiles in (
                (q, cuq, max(queries), False, False, valid_tiles_q),
                (key, cuk, max(histories), False, True, valid_tiles_k),
                (value, cuk, max(histories), True, True, valid_tiles_k),
            ):
                dense = packed_cache(source) if cache_source else source
                payload, sf = produce(
                    source,
                    cu,
                    maximum,
                    table=table if cache_source else None,
                    total=batch * maximum if cache_source else None,
                    column=column,
                )
                ref = prepare(dense, cu, maximum)
                expected_payload, expected_sf = (
                    (ref[1], ref[6]) if column else (ref[0], ref[5])
                )
                assert torch.equal(payload[: dense.shape[0]], expected_payload)
                heads = dense.shape[1]
                assert torch.equal(
                    sf.reshape(heads, -1, 1024)[:, :tiles],
                    expected_sf.reshape(heads, -1, 1024)[:, :tiles],
                )
            paged_forward(**kwargs)
            torch.cuda.synchronize()
            expected = oracle()
            relative = ((output.float() - expected).norm() / expected.norm()).item()
            quantized_expected = oracle(quantized=True)
            arithmetic_error = (
                (output.float() - quantized_expected).norm() / quantized_expected.norm()
            ).item()
            finite = torch.isfinite(output).all().item()
            receipt["checks"].append(
                {
                    "queries": queries,
                    "histories": histories,
                    "mode": "eager",
                    "relative_l2": relative,
                    "finite": finite,
                    "producer_exact": True,
                    "quantized_relative_l2": arithmetic_error,
                }
            )
            save()
            precision_ok = precision_ok and relative <= 0.05
            assert finite and arithmetic_error <= 0.01, receipt["checks"][-1]
            if queries == [17, 1]:
                paged_forward(**kwargs)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    paged_forward(**kwargs)
                for mode in ("capture", "updated_lengths_and_pages"):
                    if mode == "updated_lengths_and_pages":
                        cuq.copy_(
                            torch.tensor([0, 1, 18], dtype=torch.int32, device="cuda")
                        )
                        cuk.copy_(
                            torch.tensor(
                                [0, 511, 640], dtype=torch.int32, device="cuda"
                            )
                        )
                        lengths.copy_(
                            torch.tensor([511, 129], dtype=torch.int32, device="cuda")
                        )
                        page_offsets.copy_(
                            torch.tensor([0, 32, 41], dtype=torch.int32, device="cuda")
                        )
                        table.copy_(table.flip(0))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = (
                        oracle([1, 17], [511, 129])
                        if mode == "updated_lengths_and_pages"
                        else oracle()
                    )
                    relative = (
                        (output.float() - expected).norm() / expected.norm()
                    ).item()
                    finite = torch.isfinite(output).all().item()
                    quantized_expected = (
                        oracle([1, 17], [511, 129], quantized=True)
                        if mode == "updated_lengths_and_pages"
                        else oracle(quantized=True)
                    )
                    arithmetic_error = (
                        (output.float() - quantized_expected).norm()
                        / quantized_expected.norm()
                    ).item()
                    receipt["checks"].append(
                        {
                            "mode": mode,
                            "relative_l2": relative,
                            "finite": finite,
                            "quantized_relative_l2": arithmetic_error,
                        }
                    )
                    save()
                    precision_ok = precision_ok and relative <= 0.05
                    assert finite and arithmetic_error <= 0.01, receipt["checks"][-1]
        receipt["passed"] = precision_ok
        receipt["arithmetic_passed"] = True
        receipt["diagnostic_only"] = not precision_ok
        receipt["workspace_bytes"] = forward_plan(
            torch.device("cuda:0")
        ).scratch_workspace_bytes()
        save()
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
