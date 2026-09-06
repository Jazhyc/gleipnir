"""Measure cross-prefix cache reuse and score agreement on training prefixes."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    # Match the ID runner's environment before any model/backend import.
    os.environ["PATH"] = (
        f"{Path(sys.executable).parent.absolute()}:{os.environ.get('PATH', '')}"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    for variable, subdir in (
        ("TILELANG_CACHE_DIR", "tilelang"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TVM_CACHE_DIR", "tvm"),
    ):
        os.environ[variable] = f"/tmp/gleipnir-gpu-0,1/{subdir}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path)
    parser.add_argument("--parallel-trajectories", type=int, choices=(1, 8), default=8)
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument(
        "--gdn-prefill-backend", choices=("triton", "flashinfer"), default="triton"
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype", choices=("auto", "float32"), default="auto"
    )
    args = parser.parse_args()
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if args.batch_invariant else "0"
    if args.output.exists():
        raise FileExistsError(args.output)
    memory = [
        int(value)
        for value in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
    ]
    if len(memory) != 2 or max(memory) > 1024:
        raise RuntimeError(f"both GPUs must be idle before canary: {memory}")
    root = Path("data/monitoring_prefix_supervision")
    manifest = json.loads((root / "manifest.json").read_text())
    source = Path(manifest["source"])
    refs = root / "prefix_references.jsonl"
    instruction_path = Path(__file__).with_name("teacher_prefix.txt")
    for path, expected in (
        (source, manifest["source_sha256"]),
        (refs, manifest["references_sha256"]),
        (instruction_path, manifest["instruction_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError(f"artifact drift: {path}")
    instruction = instruction_path.read_text().rstrip()
    grouped = defaultdict(list)
    for line in refs.open():
        row = json.loads(line)
        grouped[row["parent_prompt_id"]].append(row)
    chosen = defaultdict(list)
    for parent, rows in grouped.items():
        if len(rows) >= 3 and len(chosen[rows[0]["source"]]) < 2:
            chosen[rows[0]["source"]].append(parent)
    parents = {p for values in chosen.values() for p in values}
    if set(chosen) != {"stride", "gloom", "bash_arena", "bash_bench"}:
        raise ValueError("canary requires expanding prefixes from all four sources")
    if len(parents) != 8:
        raise ValueError("canary requires eight independent trajectories")
    trajectories = {}
    for line in source.open():
        row = json.loads(line)
        if row["prompt_id"] in parents:
            trajectories[row["prompt_id"]] = (
                row["student_prompt"]
                .split("<agent_trajectory>\n", 1)[1]
                .rsplit("</agent_trajectory>", 1)[0]
            )
    # Three expanding prefixes per source, no labels or teacher targets.
    selected = [
        row
        for values in chosen.values()
        for parent in values
        for row in grouped[parent][:3]
    ]
    config = json.loads(args.benchmark_config.read_text())
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("0", "1")]
    if any(len(x) != 1 for x in ids):
        raise ValueError("decision labels must each be one token")
    ids = [x[0] for x in ids]
    prompts = []
    for row in selected:
        text = trajectories[row["parent_prompt_id"]][: row["end_character"]]
        user = (
            f"{instruction}\n<agent_trajectory>\n{text}"
            + ("" if text.endswith("\n") else "\n")
            + "</agent_trajectory>\n"
        )
        if (
            hashlib.sha256(user.encode()).hexdigest()
            != row["rendered_user_prompt_sha256"]
        ):
            raise ValueError("rendered prompt drift")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not rendered.endswith(config["prompt"]["assistant_suffix"]):
            raise ValueError("assistant wrapper drift")
        prompts.append(rendered + "Prediction:")
    llm = LLM(
        model=model["id"],
        revision=model["revision"],
        tokenizer_revision=model["revision"],
        dtype="bfloat16",
        quantization="fp8",
        tensor_parallel_size=2,
        language_model_only=True,
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        mamba_cache_mode="all",
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        gdn_prefill_backend=args.gdn_prefill_backend,
        seed=20260906,
        enforce_eager=bool(config["engine"].get("enforce_eager", False)),
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=2,
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    results = {}
    # Warm kernels before timing; reset between cold requests, not growing prefixes.
    llm.generate(prompts[:1], sampling, use_tqdm=False)
    for mode in ("cold", "cold_repeat", "growing"):
        if not llm.reset_prefix_cache():
            raise RuntimeError("prefix cache reset failed")
        rows = []
        started = time.perf_counter()
        for index, prompt in enumerate(prompts):
            if mode.startswith("cold") and not llm.reset_prefix_cache():
                raise RuntimeError("cold cache reset failed")
            out = llm.generate([prompt], sampling, use_tqdm=False)[0]
            probs = out.outputs[0].logprobs[0]
            lp = [float(probs[token].logprob) for token in ids]
            if not all(math.isfinite(x) for x in lp):
                raise ValueError("nonfinite decision logprobs")
            maximum = max(lp)
            score = math.exp(lp[1] - maximum) / sum(math.exp(x - maximum) for x in lp)
            rows.append(
                {
                    "id": selected[index]["id"],
                    "score": score,
                    "logprob_0": lp[0],
                    "logprob_1": lp[1],
                    "prompt_tokens": len(out.prompt_token_ids),
                    "cached_tokens": out.num_cached_tokens,
                }
            )
        results[mode] = {"seconds": time.perf_counter() - started, "rows": rows}
    if not llm.reset_prefix_cache():
        raise RuntimeError("batched cache reset failed")
    batched = {}
    started = time.perf_counter()
    for offset in range(3):
        indices = list(range(offset, len(prompts), 3))
        outputs = llm.generate([prompts[i] for i in indices], sampling, use_tqdm=False)
        for index, out in zip(indices, outputs, strict=True):
            probs = out.outputs[0].logprobs[0]
            lp = [float(probs[token].logprob) for token in ids]
            if not all(math.isfinite(x) for x in lp):
                raise ValueError("nonfinite batched logprobs")
            maximum = max(lp)
            batched[index] = {
                "id": selected[index]["id"],
                "score": math.exp(lp[1] - maximum)
                / sum(math.exp(x - maximum) for x in lp),
                "logprob_0": lp[0],
                "logprob_1": lp[1],
                "prompt_tokens": len(out.prompt_token_ids),
                "cached_tokens": out.num_cached_tokens,
            }
    results["batched_growing"] = {
        "seconds": time.perf_counter() - started,
        "rows": [batched[i] for i in range(len(prompts))],
    }
    errors = [
        abs(a["score"] - b["score"])
        for mode in (
            ("growing",)
            if args.parallel_trajectories == 1
            else ("growing", "batched_growing")
        )
        for a, b in zip(results["cold"]["rows"], results[mode]["rows"], strict=True)
    ]
    hits = sum(row["cached_tokens"] or 0 for row in results["growing"]["rows"])
    rubric_tokens = len(tokenizer.encode(instruction, add_special_tokens=False)) + 64
    trajectory_hits = [
        row["cached_tokens"] or 0
        for mode in ("growing", "batched_growing")
        for index, row in enumerate(results[mode]["rows"])
        if index % 3 != 0
    ]
    selected_mode = "growing" if args.parallel_trajectories == 1 else "batched_growing"
    source_reuse = {
        source_name: any(
            (scored["cached_tokens"] or 0) > rubric_tokens
            for reference, scored in zip(
                selected, results[selected_mode]["rows"], strict=True
            )
            if reference["source"] == source_name
        )
        for source_name in chosen
    }
    results.update(
        model=model,
        runtime_versions={
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers")
        },
        manifest_sha256=sha(root / "manifest.json"),
        mean_absolute_error=sum(errors) / len(errors),
        max_absolute_error=max(errors),
        cached_tokens=hits,
        passed=all(source_reuse.values())
        and max(errors) <= 0.05
        and sum(errors) / len(errors) <= 0.02,
        purpose="cache-enabled engine: reset-per-request versus growing-prefix reuse",
        trajectory_reuse_cached_tokens=trajectory_hits,
        rubric_only_token_ceiling=rubric_tokens,
        source_trajectory_reuse=source_reuse,
        parallel_trajectories=args.parallel_trajectories,
        batch_invariant=args.batch_invariant,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        gdn_prefill_backend=args.gdn_prefill_backend,
        cold_repeat_max_absolute_error=max(
            abs(a["score"] - b["score"])
            for a, b in zip(
                results["cold"]["rows"], results["cold_repeat"]["rows"], strict=True
            )
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if not results["passed"]:
        raise RuntimeError("cache canary failed; do not launch annotation")
    if args.cache_output:
        from experiments.monitoring_prefix_supervision.cache import run_cache

        run_cache(
            llm,
            tokenizer,
            sampling,
            ids,
            grouped,
            source,
            instruction,
            config,
            manifest,
            args.cache_output,
            results,
            parallel_trajectories=args.parallel_trajectories,
        )


if __name__ == "__main__":
    main()
