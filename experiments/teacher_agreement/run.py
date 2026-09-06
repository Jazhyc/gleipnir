"""Frozen matched-prompt local-teacher agreement diagnostic."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.special import expit
from scipy.stats import spearmanr

from gleipnir.monitoring_systems_screen import atomic_write_json, atomic_write_jsonl

ROOT = Path("results/teacher_agreement")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def agreement(rows: list[dict]) -> dict:
    """Describe probability and log-odds disagreement; never transform targets."""
    if not rows:
        raise ValueError("empty teacher scores")
    q = np.array([r["qwen_score"] for r in rows])
    k = np.array([r["kimi_score"] for r in rows])
    qm = np.array([r["qwen_margin"] for r in rows])
    km = np.array([r["kimi_margin"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    if not all(np.isfinite(a).all() for a in (q, k, qm, km)):
        raise ValueError("nonfinite teacher scores")
    if not all(np.all((a >= 0) & (a <= 1)) for a in (q, k)):
        raise ValueError("invalid probabilities")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid labels")
    slope, intercept = np.linalg.lstsq(
        np.column_stack([km, np.ones(len(km))]), qm, rcond=None
    )[0]
    return {
        "n": len(rows),
        "probability_mae": float(np.mean(abs(q - k))),
        "qwen_minus_kimi_bias": float(np.mean(q - k)),
        "max_probability_error": float(np.max(abs(q - k))),
        "decision_disagreement": float(np.mean((q >= 0.5) != (k >= 0.5))),
        "spearman": (
            float(spearmanr(q, k).statistic)
            if np.ptp(q) > 0 and np.ptp(k) > 0
            else None
        ),
        "qwen_mean_confidence": float(np.mean(np.maximum(q, 1 - q))),
        "kimi_mean_confidence": float(np.mean(np.maximum(k, 1 - k))),
        "qwen_brier": float(np.mean((q - labels) ** 2)),
        "kimi_brier": float(np.mean((k - labels) ** 2)),
        "descriptive_qwen_margin_on_kimi_slope": float(slope),
        "descriptive_qwen_margin_on_kimi_intercept": float(intercept),
    }


def prepare() -> None:
    from transformers import AutoTokenizer

    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        render_margin_prompt,
    )

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        settings = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    root = Path(settings["output"])
    if root.exists():
        raise FileExistsError(root)
    prompts = [json.loads(s) for s in Path(settings["prompts"]).open()]
    target_rows = list(map(json.loads, Path(settings["kimi"]).open()))
    targets = {(r["dataset"], r["index"]): r for r in target_rows}
    if len(targets) != len(target_rows) or len({r["id"] for r in prompts}) != len(
        prompts
    ):
        raise ValueError("duplicate source identities")
    selected, used = [], set()
    sources = sorted({r["metadata"]["source_dataset"] for r in prompts})
    for source in sources:
        for label in (0, 1):
            eligible = [
                r
                for r in prompts
                if r["metadata"]["source_dataset"] == source
                and r["metadata"]["ground_truth"] == label
            ]
            eligible.sort(
                key=lambda r: hashlib.sha256(
                    (settings["seed"] + r["id"]).encode()
                ).digest()
            )
            chosen = []
            for row in eligible:
                trajectory = row["metadata"]["trajectory_sha256"]
                if trajectory not in used:
                    chosen.append(row)
                    used.add(trajectory)
                if len(chosen) == settings["per_source_label"]:
                    break
            if len(chosen) != settings["per_source_label"]:
                raise ValueError("insufficient unique trajectories")
            selected.extend(chosen)
    matched = []
    for r in selected:
        k = targets[("tool_trajectory/" + r["metadata"]["source_dataset"], r["id"])]
        if (
            hashlib.sha256(r["prompt"].encode()).hexdigest()
            != k["rendered_prompt_sha256"]
            or k["label"] != r["metadata"]["ground_truth"]
        ):
            raise ValueError("Kimi prompt/label mismatch")
        margin = k["target_logprobs"]["positive"] - k["target_logprobs"]["negative"]
        if not np.isclose(expit(margin), k["soft_target"], rtol=0, atol=1e-10):
            raise ValueError("Kimi raw-score mismatch")
        matched.append({"id": r["id"], **k})
    config = json.loads(Path(settings["template"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    lengths = [
        len(
            tokenizer.encode(
                render_margin_prompt(
                    tokenizer,
                    r["prompt"],
                    enable_thinking=False,
                    assistant_suffix=config["prompt"]["assistant_suffix"],
                    decision_prefix="Prediction:",
                ),
                add_special_tokens=False,
            )
        )
        for r in selected
    ]
    if max(lengths) >= 32768:
        raise ValueError("context overflow")
    atomic_write_jsonl(root / "prompts.jsonl", selected)
    atomic_write_jsonl(root / "kimi.jsonl", matched)
    config.update(
        campaign_id="matched-qwen-kimi-v1",
        hypothesis="Matched full-trajectory teacher agreement",
        intervention="Original teacher rubric; no prefix-cache reuse",
        baselines=["Existing Kimi K3 scores on identical rendered prompts"],
    )
    config["scope"] = {
        "input": str(root / "prompts.jsonl"),
        "input_sha256": sha(root / "prompts.jsonl"),
        "manifest": str(root / "prompts.manifest.json"),
        "rows": len(selected),
        "sources": sources,
        "role": "training_population_diagnostic",
        "selection_rule": (
            "64 unique-trajectory hash-selected rows per source/label; "
            "no score selection"
        ),
    }
    config["engine"].update(
        batch_rows=1,
        enable_prefix_caching=False,
        audited_max_prompt_tokens=max(lengths),
        audited_total_prompt_tokens=sum(lengths),
    )
    atomic_write_json(
        root / "prompts.manifest.json",
        {
            "output": {"rows": len(selected), "sha256": sha(root / "prompts.jsonl")},
            "prompt": config["prompt"],
        },
    )
    atomic_write_json(root / "benchmark.json", config)
    atomic_write_json(
        root / "manifest.json",
        {
            "settings": settings,
            "source_sha256": sha(Path(settings["prompts"])),
            "kimi_source_sha256": sha(Path(settings["kimi"])),
            "files": {
                str(p): sha(p)
                for p in [
                    root / "prompts.jsonl",
                    root / "kimi.jsonl",
                    root / "benchmark.json",
                    root / "prompts.manifest.json",
                ]
            },
        },
    )
    print(
        json.dumps(
            {"rows": len(selected), "tokens": sum(lengths), "max_tokens": max(lengths)}
        )
    )


def analyze() -> None:
    kimi = {r["id"]: r for r in map(json.loads, (ROOT / "kimi.jsonl").open())}
    qwen = list(map(json.loads, (ROOT / "qwen/predictions.jsonl").open()))
    if len(qwen) != len(kimi) or {r["id"] for r in qwen} != set(kimi):
        raise ValueError("incomplete paired scoring")
    pairs = []
    for q in qwen:
        k = kimi[q["id"]]
        if q["source_prompt_sha256"] != k["rendered_prompt_sha256"]:
            raise ValueError("scored prompt drift")
        if (
            q["label"] != k["label"]
            or "tool_trajectory/" + q["source"] != k["dataset"]
            or q["config_sha256"] != sha(ROOT / "benchmark.json")
        ):
            raise ValueError("scored identity/config drift")
        if not np.isclose(
            expit(q["logprob_1"] - q["logprob_0"]),
            q["score"],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Qwen raw-score mismatch")
        pairs.append(
            {
                "id": q["id"],
                "source": q["source"],
                "kimi_provider": k["provider"],
                "label": q["label"],
                "qwen_score": q["score"],
                "kimi_score": k["soft_target"],
                "qwen_margin": q["logit_margin_1_minus_0"],
                "kimi_margin": k["target_logprobs"]["positive"]
                - k["target_logprobs"]["negative"],
            }
        )
    atomic_write_jsonl(ROOT / "paired_scores.jsonl", pairs)
    atomic_write_json(
        ROOT / "agreement.json",
        {
            "pooled": agreement(pairs),
            "sources": {
                s: agreement([r for r in pairs if r["source"] == s])
                for s in sorted({r["source"] for r in pairs})
            },
            "kimi_providers": {
                p: agreement([r for r in pairs if r["kimi_provider"] == p])
                for p in sorted({r["kimi_provider"] for r in pairs})
            },
            "input_sha256": {
                name: sha(ROOT / name)
                for name in ("kimi.jsonl", "qwen/predictions.jsonl", "benchmark.json")
            },
            "interpretation": (
                "Descriptive matched-prompt training-population diagnostic; "
                "not causal evidence or target recalibration."
            ),
        },
    )


def run() -> None:
    from experiments.adapter_capacity_scaling.run_lambda import runtime_environment

    manifest = json.loads((ROOT / "manifest.json").read_text())
    for p, h in manifest["files"].items():
        if sha(Path(p)) != h:
            raise ValueError("frozen input drift")
    memory = [
        int(v)
        for v in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
    ]
    if len(memory) != 2 or max(memory) > 1024:
        raise RuntimeError("both GPUs must be idle")
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.tool_trajectory_monitoring.benchmark_qwen_ood",
            "--config",
            str(ROOT / "benchmark.json"),
            "--output",
            str(ROOT / "qwen"),
        ],
        env=runtime_environment("0,1"),
        check=True,
    )
    analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "analyze"])
    {"prepare": prepare, "run": run, "analyze": analyze}[parser.parse_args().action]()
