"""Extract frozen Qwen3.5 activations for the probe studies (one GPU).

For every row of the requested splits, one BF16 Transformers forward captures:

- the residual stream after decoder block ``--layer`` (zero-based) at every
  trajectory token (prompted backbone: the trajectory plus the suffix after it;
  NTP: the whole bare trajectory), appended to an activation store
  (:mod:`gleipnir.probes.store`) in the requested formats;
- the last token's residual stream after every decoder block (pre final norm),
  saved as BF16 bits in ``last_layers.npy`` for the last-token probe;
- under the prompted backbone, the decision margin ``logit(1) - logit(0)`` from
  the final norm and tied LM head in FP32, for parity with the stock-vLLM
  benchmark. The ``parity`` split stores margins only.

The pinned FLA and causal-conv1d fast paths are installed, probed and bound
before model import; the run fails closed without them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Backbones: (model id, revision, decoder blocks, hidden size), checked at load.
MODELS = {
    "qwen35_4b": (
        "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", 32, 2560
    ),
    "qwen38_27b": (
        "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", 64, 5120
    ),
}  # fmt: skip
SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
DECISION_PREFIX = "Prediction:"
KERNEL_FLAG = "GLEIPNIR_PROBE_KERNELS"


def render(tokenizer, row: dict, backbone: str) -> tuple[str, int]:
    """Return the model input and the character offset where kept tokens begin."""
    if backbone == "ntp":
        return row["trajectory"], 0
    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        render_margin_prompt,
    )

    text = render_margin_prompt(
        tokenizer,
        row["prompt"],
        enable_thinking=False,
        assistant_suffix=SUFFIX,
        decision_prefix=DECISION_PREFIX,
    )
    # The chat template trims the message, so anchor on the instruction prefix
    # and verify the trajectory follows it byte for byte.
    prefix = row["prompt"][: row["trajectory_start"]]
    if text.count(prefix) != 1:
        raise ValueError(f"instruction prefix not unique in row {row['id']}")
    start = text.index(prefix) + len(prefix)
    if not text.startswith(row["trajectory"].rstrip(), start):
        raise ValueError(f"trajectory not embedded after prefix in row {row['id']}")
    return text, start


def tokenize(tokenizer, text: str, keep_from_char: int) -> tuple[list[int], int]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = [end for _, end in encoded["offset_mapping"]]
    keep_from = next(i for i, end in enumerate(ends) if end > keep_from_char)
    return encoded["input_ids"], keep_from


def load_model(model_id: str, revision: str):
    import importlib.metadata

    import torch
    import transformers
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5

    versions = {
        package: importlib.metadata.version(package)
        for package in ("flash-linear-attention", "fla-core", "causal-conv1d")
    }
    if versions != {
        "flash-linear-attention": "0.5.2",
        "fla-core": "0.5.2",
        "causal-conv1d": "1.6.2.post1",
    }:
        raise RuntimeError(f"unpinned Qwen3.5 fast kernels: {versions}")
    if getattr(modeling_qwen3_5, "is_fast_path_available", True) is not True:
        raise RuntimeError("Transformers did not bind the Qwen3.5 fast path")
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "kernels": versions,
        "gpu": torch.cuda.get_device_name(),
    }
    return model, runtime


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    from gleipnir.probes.store import StoreWriter

    data = ROOT / "data/probe_generalization" / args.population
    model_id, revision, depth, width = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.convert_tokens_to_ids(["0", "1"]) != [15, 16]:
        raise ValueError("decision tokens are not ids 15/16")
    model, runtime = load_model(model_id, revision)
    layers = model.model.layers
    hidden = model.config.hidden_size
    if (len(layers), hidden) != (depth, width):
        raise ValueError(
            f"{model_id}: {len(layers)} blocks x {hidden}, expected {depth} x {width}"
        )
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"layer {args.layer} outside 0..{len(layers) - 1}")
    head = model.lm_head.weight[15:17].detach().float()
    capture: dict = {}

    def hook(index: int):
        def save(_module, _inputs, output):
            states = output[0] if isinstance(output, tuple) else output
            capture["last"][index] = states[0, -1].clone()
            if index == args.layer:
                capture["span"] = states[0, capture["keep_from"] :].clone()

        return save

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    for split in args.splits:
        rows = [json.loads(line) for line in (data / f"{split}.jsonl").open()]
        rows = rows[: args.limit] if args.limit else rows
        if split == "parity" and args.backbone != "prompted":
            raise ValueError("parity margins exist only for the prompted backbone")
        out = Path(args.out_root) / args.population / args.backbone / split
        store_activations = split != "parity"
        writer = (
            StoreWriter(out, hidden, tuple(args.formats)) if store_activations else None
        )
        if not store_activations:
            out.mkdir(parents=True, exist_ok=False)
        last_layers = np.zeros((len(rows), len(layers), hidden), dtype=np.int16)
        records, started, tokens = [], time.time(), 0
        for i, row in enumerate(rows):
            text, keep_char = render(tokenizer, row, args.backbone)
            ids, keep_from = tokenize(tokenizer, text, keep_char)
            capture.update(last={}, keep_from=keep_from)
            with torch.inference_mode():
                output = model.model(
                    input_ids=torch.tensor([ids], device="cuda"), use_cache=False
                )
            final = output.last_hidden_state[0, -1].float()
            if not bool(torch.isfinite(final).all()):
                raise ValueError(f"nonfinite final state in row {row['id']}")
            logits = head @ final
            last = torch.stack([capture["last"][j] for j in range(len(layers))])
            last_layers[i] = last.to(torch.bfloat16).view(torch.int16).cpu().numpy()
            record = {
                key: row[key]
                for key in (
                    "id",
                    "split",
                    "source",
                    "label",
                    "soft_target",
                    "reference_margin",
                )
                if key in row
            }
            record.update(
                input_tokens=len(ids),
                kept_from=keep_from,
                rendered_sha256=hashlib.sha256(text.encode()).hexdigest(),
                margin=float(logits[1] - logits[0])
                if args.backbone == "prompted"
                else None,
            )
            if writer is not None:
                writer.append(capture["span"], record)
            records.append(record)
            tokens += len(ids)
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                rate = tokens / (time.time() - started)
                print(
                    f"{args.backbone}:{split} {i + 1}/{len(rows)} {rate:.0f} tok/s",
                    flush=True,
                )
        np.save(out / "last_layers.npy", last_layers)
        (out / "rows.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
        )
        meta = {
            "model": model_id,
            "revision": revision,
            "backbone": args.backbone,
            "split": split,
            "layer": args.layer,
            "rows": len(records),
            "input_tokens": tokens,
            "seconds": time.time() - started,
            "input_sha256": hashlib.sha256(
                (data / f"{split}.jsonl").read_bytes()
            ).hexdigest(),
            "last_layers": "post-block residual, pre final norm, BF16 bits as int16",
            "runtime": runtime,
        }
        if writer is not None:
            writer.close(meta)
        else:
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("split_complete", json.dumps(meta), flush=True)
    for handle in handles:
        handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        choices=("pilot", "full", "sleightbench", "optimized_diag"),
        required=True,
    )
    parser.add_argument("--model", choices=tuple(MODELS), default="qwen35_4b")
    parser.add_argument("--backbone", choices=("prompted", "ntp"), required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--formats", nargs="+", default=["fp8"])
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--out-root", required=True, help="activation store root")
    parser.add_argument("--limit", type=int, help="smoke test: first N rows")
    args = parser.parse_args()
    if os.environ.get(KERNEL_FLAG) != "1":
        from gleipnir.training.backends.qwen35 import ensure_qwen35_fast_kernels

        environment = ensure_qwen35_fast_kernels()
        environment[KERNEL_FLAG] = "1"
        command = [sys.executable, "-m", "experiments.probe_generalization.extract"]
        os.execve(sys.executable, [*command, *sys.argv[1:]], environment)
    run(args)


if __name__ == "__main__":
    main()
