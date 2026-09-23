"""Bounded eager canary: base, untouched FP32 master, and BF16 merged model."""

from __future__ import annotations

import gc
import json
import time

from experiments.local_inference.core import (
    CONFIG,
    DATA,
    ROOT,
    canary_rows,
    compare,
    parity_passes,
    read_rows,
    write_json,
)
from gleipnir.binary_evaluation import binary_token_ids
from gleipnir.qwen35_adapter_rebase import sha256_file


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    config = json.loads(CONFIG.read_text())
    if (
        sha256_file(DATA / "subset.jsonl")
        != json.loads((DATA / "manifest.json").read_text())["subset_sha256"]
    ):
        raise ValueError("subset changed")
    existing = ROOT / "reference.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        if (
            previous.get("passed")
            and previous["subset_sha256"] == sha256_file(DATA / "subset.jsonl")
            and previous["merge_manifest_sha256"]
            == sha256_file(ROOT / "merged_bf16/merge_manifest.json")
            and parity_passes(previous["merge_parity"], config["parity"])
            and previous["adapter_effect"]["max_absolute_error"] > 1e-5
        ):
            print("Reusing passed reference canary", flush=True)
            return
        raise ValueError(
            "retain prior reference.json before retrying changed/failed canary"
        )
    rows = canary_rows(read_rows())
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "base", local_files_only=True)
    ids = binary_token_ids(tokenizer)
    result = {
        "ids": [r["id"] for r in rows],
        "tokens": [r["tokens"] for r in rows],
        "subset_sha256": sha256_file(DATA / "subset.jsonl"),
        "merge_manifest_sha256": sha256_file(ROOT / "merged_bf16/merge_manifest.json"),
        "backend": "Transformers SDPA; bounded eager canary only; logits_to_keep=1",
        "config_sha256": sha256_file(CONFIG),
    }

    def score(model, name):
        scores = []
        model.eval()
        for row in rows:
            started = time.perf_counter()
            inputs = tokenizer(
                row["prompt"], return_tensors="pt", add_special_tokens=False
            ).to("cuda")
            with torch.inference_mode():
                logits = (
                    model(**inputs, use_cache=False, logits_to_keep=1)
                    .logits[0, -1, ids]
                    .float()
                )
            scores.append(
                {
                    "id": row["id"],
                    "logits": logits.tolist(),
                    "score": torch.softmax(logits, -1)[1].item(),
                    "seconds": time.perf_counter() - started,
                }
            )
            print(f"{name}: {row['id']} score={scores[-1]['score']:.8f}", flush=True)
        result[name] = scores
        write_json(ROOT / "reference.json", result)

    model = AutoModelForCausalLM.from_pretrained(
        ROOT / "base",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    score(model, "base")
    model = PeftModel.from_pretrained(model, ROOT / "adapter", local_files_only=True)
    score(model, "master")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / "merged_bf16",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    score(model, "merged")
    master = [r["score"] for r in result["master"]]
    result["merge_parity"] = compare(master, [r["score"] for r in result["merged"]])
    result["adapter_effect"] = compare([r["score"] for r in result["base"]], master)
    result["passed"] = (
        parity_passes(result["merge_parity"], config["parity"])
        and result["adapter_effect"]["max_absolute_error"] > 1e-5
    )
    write_json(ROOT / "reference.json", result)
    if not result["passed"]:
        raise RuntimeError("master/merge canary failed; see reference.json")


if __name__ == "__main__":
    main()
