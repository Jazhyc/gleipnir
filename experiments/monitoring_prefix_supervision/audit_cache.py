"""Compare completed-cache targets with fresh scoring on fixed long prefixes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from gleipnir.prefix_audit import select_cache_audit
from gleipnir.prefix_cache import binary_score, validate_resume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/monitoring_prefix_supervision/prefix_references.jsonl"),
    )
    args = parser.parse_args()
    output = args.cache_dir / "fresh_audit.json"
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads((args.cache_dir / "contract.json").read_text())
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    complete = json.loads((args.cache_dir / "complete.json").read_text())
    manifest = contract["manifest"]
    for path, expected in (
        (args.references, manifest["references_sha256"]),
        (Path(manifest["source"]), manifest["source_sha256"]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("audit input drift")
    refs = {r["id"]: r for r in map(json.loads, args.references.open())}
    cached_path = args.cache_dir / "logits.jsonl"
    completed = validate_resume(cached_path, refs, contract_hash)
    if completed != set(refs) or complete["contract_sha256"] != contract_hash:
        raise ValueError("cache incomplete")
    selected = select_cache_audit(list(refs.values()))
    selected_ids = {r["id"] for r in selected}
    cached = {
        r["id"]: r
        for r in map(json.loads, cached_path.open())
        if r["id"] in selected_ids
    }
    parents = {r["parent_prompt_id"] for r in selected}
    trajectories = {
        r["prompt_id"]: r["student_prompt"]
        .split("<agent_trajectory>\n", 1)[1]
        .rsplit("</agent_trajectory>", 1)[0]
        for r in map(json.loads, Path(manifest["source"]).open())
        if r["prompt_id"] in parents
    }
    instruction = Path(__file__).with_name("teacher_prefix.txt").read_text().rstrip()
    if (
        hashlib.sha256(
            Path(__file__).with_name("teacher_prefix.txt").read_bytes()
        ).hexdigest()
        != manifest["instruction_sha256"]
    ):
        raise ValueError("instruction drift")
    for name, version in contract["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"audit runtime mismatch: {name}")
    memory = [
        int(v)
        for v in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
    ]
    if len(memory) != 2 or max(memory) > 1024:
        raise RuntimeError("audit requires two idle GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["PATH"] = (
        str(Path(sys.executable).parent.absolute()) + ":" + os.environ.get("PATH", "")
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model, engine = contract["model"], contract["engine"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    ids = contract["sampling"]["decision_token_ids"]
    if [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)] != [
        [i] for i in ids
    ]:
        raise ValueError("decision token drift")
    llm = LLM(
        model=model["id"],
        revision=model["revision"],
        tokenizer_revision=model["revision"],
        dtype=engine["dtype"],
        quantization=engine["quantization"],
        tensor_parallel_size=2,
        language_model_only=True,
        max_model_len=engine["max_model_len"],
        max_num_seqs=engine["max_num_seqs"],
        max_num_batched_tokens=engine["max_num_batched_tokens"],
        gpu_memory_utilization=engine["gpu_memory_utilization"],
        enable_prefix_caching=True,
        mamba_cache_mode=engine["mamba_cache_mode"],
        mamba_ssm_cache_dtype=engine["mamba_ssm_cache_dtype"],
        gdn_prefill_backend=engine["gdn_prefill_backend"],
        seed=engine["seed"],
        enforce_eager=engine["enforce_eager"],
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=2,
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    scored = []
    for ref in selected:
        text = trajectories[ref["parent_prompt_id"]][: ref["end_character"]]
        user = (
            f"{instruction}\n<agent_trajectory>\n{text}"
            + ("" if text.endswith("\n") else "\n")
            + "</agent_trajectory>\n"
        )
        if (
            hashlib.sha256(user.encode()).hexdigest()
            != ref["rendered_user_prompt_sha256"]
        ):
            raise ValueError("audit prompt drift")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not prompt.endswith(contract["prompt"]["assistant_suffix"]):
            raise ValueError("audit wrapper drift")
        if not llm.reset_prefix_cache():
            raise RuntimeError("fresh cache reset failed")
        prediction = llm.generate([prompt + "Prediction:"], sampling, use_tqdm=False)[0]
        probs = prediction.outputs[0].logprobs[0]
        lp0, lp1 = [float(probs[t].logprob) for t in ids]
        scored.append(
            {
                "id": ref["id"],
                "source": ref["source"],
                "logprob_0": lp0,
                "logprob_1": lp1,
                "score": binary_score(lp0, lp1),
                "cached_score": cached[ref["id"]]["score"],
                "prompt_tokens": len(prediction.prompt_token_ids),
            }
        )
    errors = [abs(r["score"] - r["cached_score"]) for r in scored]
    result = {
        "contract_sha256": contract_hash,
        "selection": "prefix-audit-v1",
        "rows": scored,
        "mean_absolute_error": sum(errors) / len(errors),
        "max_absolute_error": max(errors),
        "passed": max(errors) <= 0.05 and sum(errors) / len(errors) <= 0.02,
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError("full-workload audit failed; do not train")


if __name__ == "__main__":
    main()
