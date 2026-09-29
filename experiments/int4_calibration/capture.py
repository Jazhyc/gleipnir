"""Bounded, disjoint real MLP activation capture; no model modifications."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.local_inference.core import DATA, ROOT, digest, read_rows, write_json
from gleipnir.qwen35_adapter_rebase import sha256_file


def select(rows: list[dict], excluded: list[dict]) -> list[dict]:
    blocked = {r["id"] for r in excluded}
    hashes = {r["original_trajectory_sha256"] for r in excluded}
    chosen = []
    for source, label in sorted({(r["source"], r["label"]) for r in rows}):
        candidates = sorted(
            (r for r in rows if (r["source"], r["label"]) == (source, label)),
            key=lambda r: (r["tokens"], r["id"]),
        )
        group = []
        for row in candidates:
            if row["id"] in blocked or row["original_trajectory_sha256"] in hashes:
                continue
            hashes.add(row["original_trajectory_sha256"])
            group.append(
                dict(row, split="heldout" if len(group) == 1 else "calibration")
            )
            if len(group) == 3:
                break
        if len(group) != 3:
            raise ValueError("Insufficient independent rows")
        chosen.extend(group)
    return chosen


def parse_layers(specification: str, count: int) -> list[int]:
    """Select explicit or all decoder layers without changing trajectory selection."""
    if specification == "all":
        return list(range(count))
    values = [int(v) for v in specification.split(",")]
    if (
        not values
        or len(values) != len(set(values))
        or any(v < 0 or v >= count for v in values)
    ):
        raise ValueError("Invalid or duplicate capture layers")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,16,31")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = read_rows()
    reference = json.loads((ROOT / "reference.json").read_text())
    selected = select(
        rows,
        read_rows(DATA / "iteration32.jsonl")
        + [r for r in rows if r["id"] in reference["ids"]],
    )
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert sha256_file(DATA / "subset.jsonl") == manifest["subset_sha256"]
    merge = json.loads((ROOT / "merged_bf16/merge_manifest.json").read_text())
    for name, expected in merge["files"].items():
        assert sha256_file(ROOT / "merged_bf16" / name) == expected
    model_config = json.loads((ROOT / "merged_bf16/config.json").read_text())
    text_config = model_config.get("text_config", model_config)
    layers = parse_layers(args.layers, text_config["num_hidden_layers"])
    record = {
        "subset_sha256": sha256_file(DATA / "subset.jsonl"),
        "iteration32_sha256": sha256_file(DATA / "iteration32.jsonl"),
        "merge_manifest_sha256": sha256_file(ROOT / "merged_bf16/merge_manifest.json"),
        "rows": [{k: v for k, v in r.items() if k != "prompt"} for r in selected],
        "layers": layers,
        "source_sha256": sha256_file(Path(__file__)),
        "positions_per_row": 256,
        "backend": "Transformers BF16 SDPA causal-LM; bounded hooks only",
        "torch": torch.__version__,
        "completed": [],
    }
    write_json(args.output / "manifest.json", record)
    print(
        "selection", [(r["split"], r["id"], r["tokens"]) for r in selected], flush=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / "merged_bf16",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "merged_bf16", local_files_only=True
    )
    captured = {}
    handles = []
    weights = {}
    for index in record["layers"]:
        # Causal-LM loader strips the multimodal language_model nesting.
        mlp = model.model.layers[index].mlp
        weights[f"{index}_gate_up"] = torch.cat(
            [mlp.gate_proj.weight.detach().cpu(), mlp.up_proj.weight.detach().cpu()]
        )
        weights[f"{index}_down"] = mlp.down_proj.weight.detach().cpu()
        for name, module in (("gate_up", mlp.gate_proj), ("down", mlp.down_proj)):
            key = f"{index}_{name}"

            def hook(module, inputs, key=key):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                positions = torch.linspace(0, len(x) - 1, 256, device=x.device).long()
                values = x[positions].cpu()
                assert torch.isfinite(values).all()
                captured[key] = values

            handles.append(module.register_forward_pre_hook(hook))
    torch.save(weights, args.output / "weights.pt")
    for i, row in enumerate(selected):
        assert digest(row["prompt"]) == row["prompt_sha256"]
        inputs = tokenizer(
            row["prompt"], return_tensors="pt", add_special_tokens=False
        ).to("cuda")
        assert inputs.input_ids.shape[1] == row["tokens"]
        captured.clear()
        start = time.perf_counter()
        with torch.inference_mode():
            result = model(**inputs, use_cache=False, logits_to_keep=1)
        assert len(captured) == 2 * len(layers)
        assert torch.isfinite(result.logits).all()
        torch.save(captured, args.output / f"row_{i}.pt")
        record["completed"].append(
            {
                "row": i,
                "seconds": time.perf_counter() - start,
                "sha256": sha256_file(args.output / f"row_{i}.pt"),
            }
        )
        write_json(args.output / "manifest.json", record)
        print("captured", i, row["split"], row["tokens"], flush=True)
    for handle in handles:
        handle.remove()
    record["weights_sha256"] = sha256_file(args.output / "weights.pt")
    record["state"] = "complete"
    write_json(args.output / "manifest.json", record)


if __name__ == "__main__":
    main()
