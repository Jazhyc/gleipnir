"""Native D256 FA4 checks for zero-copy hybrid paged KV subdivision."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from flash_attn.cute.interface import _flash_attn_fwd

    from gleipnir.serving_fa4 import make_paged_fa4_forward, subdivide_paged_kv

    source_paths = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__),
        Path("src/gleipnir/serving/fa4.py"),
        Path(_flash_attn_fwd.__code__.co_filename),
    ]
    receipt = {
        "passed": False,
        "version": version("flash-attn-4"),
        "device_capability": list(torch.cuda.get_device_capability()),
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "checks": [],
    }
    assert receipt["version"] == "4.0.0b33" and receipt["device_capability"] == [10, 0]

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        archive = args.output.parent / (args.output.stem + "_sources")
        for path in source_paths:
            target = archive / path.resolve().relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())

    save()
    try:
        torch.manual_seed(17)
        cache = torch.randn((16, 2, 640, 4, 256), dtype=torch.bfloat16, device="cuda")
        k, v = cache.unbind(1)
        table = torch.tensor(
            [[7, 2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [3, 4, 5, 6, 8, 9, 10]],
            dtype=torch.int32,
            device="cuda",
        )
        forward = make_paged_fa4_forward(_flash_attn_fwd)
        for q_lengths, k_lengths in (
            ([1, 17, 129], [1271, 511, 4097]),
            ([129, 17, 513], [1271, 511, 4097]),
        ):
            q = torch.randn(
                (sum(q_lengths), 16, 256), dtype=torch.bfloat16, device="cuda"
            )
            cu = torch.tensor(
                [0, q_lengths[0], sum(q_lengths[:2]), sum(q_lengths)],
                dtype=torch.int32,
                device="cuda",
            )
            seq = torch.tensor(k_lengths, dtype=torch.int32, device="cuda")
            kwargs = dict(
                block_table=table,
                seqused_k=seq,
                cu_seqlens_q=cu,
                max_seqlen_q=max(q_lengths),
                max_seqlen_k=max(k_lengths),
                causal=True,
                softmax_scale=1 / 16,
                fa_version=4,
                num_splits=1,
            )

            def oracle(
                query_tensor=q, queries=q_lengths, keys=k_lengths
            ) -> torch.Tensor:
                outputs = []
                offset = 0
                for row, (qn, kn) in enumerate(zip(queries, keys, strict=True)):
                    pages = table[row, : (kn + 639) // 640].long()
                    key = (
                        k[pages].flatten(0, 1)[:kn].repeat_interleave(4, dim=1).float()
                    )
                    value = (
                        v[pages].flatten(0, 1)[:kn].repeat_interleave(4, dim=1).float()
                    )
                    query = query_tensor[offset : offset + qn].float()
                    score = torch.einsum("qhd,khd->hqk", query, key) / 16
                    mask = torch.arange(kn, device="cuda")[None, :] > (
                        torch.arange(qn, device="cuda")[:, None] + kn - qn
                    )
                    score.masked_fill_(mask[None], float("-inf"))
                    outputs.append(
                        torch.einsum("hqk,khd->qhd", score.softmax(-1), value)
                    )
                    offset += qn
                return torch.cat(outputs)

            def check(
                output: torch.Tensor,
                mode: str,
                queries=q_lengths,
                keys=k_lengths,
                compute=oracle,
            ) -> None:
                expected = compute()
                relative = ((output.float() - expected).norm() / expected.norm()).item()
                finite = torch.isfinite(output).all().item()
                receipt["checks"].append(
                    {
                        "q_lengths": queries,
                        "k_lengths": keys,
                        "mode": mode,
                        "relative_l2": relative,
                        "finite": finite,
                    }
                )
                save()
                assert finite and relative <= 0.01, receipt["checks"][-1]

            small_k, small_v, mapped = subdivide_paged_kv(k, v, table)
            assert (
                small_k.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
            )
            assert (
                small_v.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
            )
            assert torch.equal(k[table].flatten(1, 2), small_k[mapped].flatten(1, 2))
            assert torch.equal(v[table].flatten(1, 2), small_v[mapped].flatten(1, 2))
            output = forward(q, k, v, **kwargs)
            torch.cuda.synchronize()
            check(output, "eager")
            forward(q, k, v, **kwargs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward(q, k, v, **kwargs)
            graph.replay()
            torch.cuda.synchronize()
            check(captured, "capture")
            table.copy_(table.roll(1, dims=1))
            graph.replay()
            torch.cuda.synchronize()
            check(captured, "capture_updated_page_table")
        receipt["passed"] = True
        save()
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
