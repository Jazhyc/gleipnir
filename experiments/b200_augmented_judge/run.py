"""Freeze and run a matched A/B comparison without training or model selection."""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.metadata
import inspect
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import yaml

from gleipnir.campaigns.monitoring.evaluation import agreement
from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    write_json,
    write_rows,
)
from gleipnir.evaluation.http_score import score_batch
from gleipnir.evaluation.preferences import summarize_preferences
from gleipnir.serving.bf16_monitor import command as bf16_command
from gleipnir.serving.reference import selected_serving_default

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CONFIG = HERE / "config.yaml"
C = yaml.safe_load(CONFIG.read_text())
OUT = ROOT / "results" / C["campaign_id"]
DATA = ROOT / "data" / C["campaign_id"]
LOGS = ROOT / "logs/runpod" / C["campaign_id"]
ACTIVE = ROOT / "results/b200_attention_gdn_serving"


def source(name: str) -> Path:
    return ROOT / C["inputs"][name]["path"]


def check(*, model: bool = False) -> dict:
    """Require the original contract and executed files, including model weights."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    if file_hash(CONFIG) != manifest["config_sha256"]:
        raise ValueError("JudgeDeceiver configuration changed")
    for relative, expected in {**manifest["files"], **manifest["sources"]}.items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"JudgeDeceiver input/source changed: {relative}")
    if model:
        merge = json.loads(source("merge").read_text())
        for name, expected in merge["files_sha256"].items():
            if file_hash(Path(C["merged_model"]) / name) != expected:
                raise ValueError(f"merged weights changed: {name}")
        for name, expected in merge["source_files_sha256"].items():
            if file_hash(ROOT / C["base_model"] / name) != expected:
                raise ValueError(f"base weights changed: {name}")
    return manifest


def judge_command(parent: list[str], precision: str, sources: dict) -> list[str]:
    """Change only precision recipe and the explicit cached A/B decision rows."""
    command = bf16_command(parent, sources) if precision == "bf16" else parent.copy()
    command[command.index("-m") + 1] = "experiments.b200_augmented_judge.server"
    worker = "JudgeBf16Worker" if precision == "bf16" else "JudgeOptimizedWorker"
    command[command.index("--worker-cls") + 1] = (
        "experiments.b200_augmented_judge.worker." + worker
    )
    command[command.index("--model") + 1] = C["merged_model"]
    i = command.index("--hf-overrides") + 1
    overrides = json.loads(command[i])
    overrides["classifier_from_token"] = ["A", "B"]
    overrides["text_config"]["classifier_from_token"] = ["A", "B"]
    command[i] = json.dumps(overrides, sort_keys=True)
    i = command.index("--additional-config") + 1
    additional = json.loads(command[i])
    additional["monitor_score"]["token_ids"] = [32, 33]
    additional["serving_condition"]["merged_model"] = C["merged_model"]
    additional["judge_surface"] = {"tokens": ["A", "B"], "ids": [32, 33]}
    command[i] = json.dumps(additional, sort_keys=True)
    return command


def prepare() -> None:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    for spec in C["inputs"].values():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError(f"frozen source drift: {spec['path']}")
    selection, parent = selected_serving_default(ROOT)
    assets = ROOT / "data/judge_injection_continuation/tokenizer"
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    template = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string((assets / "chat_template.jinja").read_text())
    if tokenizer.encode("A", add_special_tokens=False).ids != [32] or tokenizer.encode(
        "B", add_special_tokens=False
    ).ids != [33]:
        raise ValueError("A/B token identity changed")
    test, canary = read_rows(source("test")), read_rows(source("canary"))
    if (
        len(test) != C["evaluation"]["rows"]
        or len(canary) != C["evaluation"]["canary_rows"]
        or len({r["lineage_group"] for r in test}) != 6
        or len({r["pair_id"] for r in test}) != 252
        or {r["lineage_group"] for r in test} & {r["lineage_group"] for r in canary}
    ):
        raise ValueError("held-out coverage/grouping changed")
    DATA.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=False)
    original = read_rows(source("historical_augmented"))
    if [r["id"] for r in original] != [r["id"] for r in test]:
        raise ValueError("historical ordered population changed")
    for split, rows in (("test", test), ("canary", canary)):
        workload = []
        for i, row in enumerate(rows):
            prompt = template.render(
                messages=[{"role": "user", "content": row["student_prompt"]}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            if not 0 < tokens < 32768:
                raise ValueError("JudgeDeceiver prompt would truncate")
            if split == "test" and (
                digest(prompt) != original[i]["prompt_sha256"]
                or tokens != original[i]["prompt_tokens"]
            ):
                raise ValueError("historical rendered prompt changed")
            workload.append(
                {
                    "id": row["id"],
                    "prompt": prompt,
                    "prompt_sha256": digest(prompt),
                    "prompt_tokens": tokens,
                }
            )
        write_rows(DATA / f"{split}_workload.jsonl", workload)
    lengths = [r["prompt_tokens"] for r in read_rows(DATA / "test_workload.jsonl")]
    if (sum(lengths), max(lengths)) != (1521530, 594):
        raise ValueError("frozen tokenizer totals changed")
    paths = set(ROOT.glob("src/**/*.py"))
    paths.update(HERE.glob("*.py"))
    paths.add(HERE / "README.md")
    additional = json.loads(parent[parent.index("--additional-config") + 1])
    paths.update(
        ROOT / p
        for p in additional["gleipnir_frost_fp4"]
        if not Path(p).name.endswith(("_canary.py", "_compare.py"))
        and Path(p).name != "fp4_gemm_tune.py"
    )
    paths.update(ROOT.glob("experiments/b200_vllm031/*.py"))
    paths.update(ROOT.glob("experiments/b200_attention_precision/*.py"))
    paths.update(ROOT.glob("experiments/b200_monitor_score/*.py"))
    paths.update(
        ROOT / p
        for p in (
            "experiments/training_procedure_screen/evaluate_causal.py",
            "experiments/deception_distillation/train_student_sft.py",
        )
    )
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}
    files = {s["path"]: s["sha256"] for s in C["inputs"].values()}
    files.update(
        {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in assets.iterdir()
            if p.is_file()
        }
    )
    files.update({str(p.relative_to(ROOT)): file_hash(p) for p in DATA.glob("*.jsonl")})
    for relative in sources:
        target = OUT / "executed_sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write_json(
        DATA / "manifest.json",
        {
            "config_sha256": file_hash(CONFIG),
            "files": files,
            "sources": sources,
            "selection": selection,
            "surface": "AB",
            "rows": 4188,
            "commands": {
                p: judge_command(parent, p, sources) for p in ("bf16", "optimized")
            },
        },
    )
    write_json(OUT / "status.json", {"stage": "prepared"})
    print("judge_prepared", len(test), sum(lengths), len(sources), flush=True)


def reference() -> None:
    """Bounded base/FP32-master/merged A/B references; no holdout scoring."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.deception_distillation.train_student_sft import (
        gated_delta_kernel_modules,
    )
    from experiments.training_procedure_screen.evaluate_causal import score_adapter

    check(model=True)
    if (OUT / "reference.json").exists():
        raise ValueError("A/B reference already attempted")
    if importlib.metadata.version("fla-core") != "0.5.2":
        raise ValueError("original reference FLA changed")
    tokenizer = AutoTokenizer.from_pretrained(C["merged_model"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    rows = read_rows(DATA / "canary_workload.jsonl")
    tokenized = [tokenizer.encode(r["prompt"], add_special_tokens=False) for r in rows]
    if [len(x) for x in tokenized] != [r["prompt_tokens"] for r in rows]:
        raise ValueError("reference tokenizer changed")
    cells, timings = {}, {}
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / C["base_model"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    kernels = gated_delta_kernel_modules(model)
    if not kernels or any(not k.startswith("fla.ops.") for k in kernels):
        raise ValueError("reference needs original FLA")
    for name in ("base", "master"):
        if name == "master":
            model = PeftModel.from_pretrained(
                model, ROOT / C["adapter"] / "causal_adapter", is_trainable=False
            )
        cells[name], timings[name] = score_adapter(
            model,
            tokenizer,
            tokenized,
            [32, 33],
            batch_size=1,
            decision_head_mode="token_logits",
        )
        print("judge_reference", name, timings[name], flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        C["merged_model"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    cells["merged"], timings["merged"] = score_adapter(
        model,
        tokenizer,
        tokenized,
        [32, 33],
        batch_size=1,
        decision_head_mode="token_logits",
    )
    gate = agreement(cells["merged"], cells["master"], cells["base"], C["parity"])
    write_json(
        OUT / "reference.json",
        {
            "cells": cells,
            "timings": timings,
            "kernels": kernels,
            "ids": [r["id"] for r in rows],
            "merged_parity": gate,
            "manifest_sha256": file_hash(DATA / "manifest.json"),
        },
    )
    print("judge_merged_parity", gate, flush=True)
    if not gate["passed"]:
        raise ValueError("A/B merged/master gate failed")


def retire(name: str, expected: dict) -> None:
    if json.loads((ACTIVE / "server.json").read_text()) != expected:
        raise ValueError("resident scorer identity changed before retirement")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.b200_vllm031.stop",
            "--archive-name",
            C["campaign_id"] + "_" + name,
        ],
        cwd=ROOT,
        check=True,
    )


def evaluation_gate(gate: dict, *, diagnostic: bool) -> None:
    """The explicit A/B exception permits MAE and correlation drift, not NaNs."""
    for value in gate.values():
        if value["passed"]:
            continue
        if (
            not diagnostic
            or not value["finite"]
            or value["correlation"] is None
            or not np.isfinite(value["correlation"])
            or not np.isfinite(value["mean_absolute_difference"])
            or value["adapter_effect"] <= value["limits"]["min_adapter_effect"]
        ):
            raise ValueError("A/B score gate failed outside authorized finite scope")


async def score(precision: str) -> None:
    from vllm.v1.core.sched import scheduler

    from experiments.b200_vllm031.run import archive_audits
    from experiments.b200_vllm031.runtime import candidate_environment
    from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources

    manifest = check(model=True)
    selection, _ = selected_serving_default(ROOT)
    if {p: importlib.metadata.version(p) for p in selection["runtime"]} != selection[
        "runtime"
    ]:
        raise ValueError("selected serving runtime changed")
    binding = {
        "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
        "integration_sha256": file_hash(ROOT / "experiments/b200_vllm031/scheduler.py"),
    }
    if binding != selection["scheduler_binding"]:
        raise ValueError("pooling scheduler source changed")
    if (ACTIVE / "server.json").exists():
        raise ValueError("retire the prior model before scoring")
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).split()
    if pids:
        raise ValueError("unrelated GPU process remains")
    destination = OUT / precision
    destination.mkdir(exist_ok=False)
    command = manifest["commands"][precision].copy()
    command[0] = sys.executable
    condition = json.loads(command[command.index("--additional-config") + 1])[
        "serving_condition"
    ]
    restore_diagnostic_sources(
        ROOT, condition, ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz"
    )
    parent = json.loads((ROOT / selection["host_parent"]).read_text())
    env = candidate_environment(ROOT)
    env.update(
        GLEIPNIR_JUDGE_PRECISION=precision,
        GLEIPNIR_GIGATOKEN_RECEIPT=str(destination / "frontend.json"),
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(binding),
    )
    if precision == "bf16":
        env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
        env["GLEIPNIR_BF16_AUDIT"] = str(destination / "bf16_audit.json")
    else:
        env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"]["validation"]
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{precision}_server.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    active = {
        "pid": process.pid,
        "command": command,
        "status": "starting",
        "runtime_migration": selection["runtime"],
        "serving_precision": precision,
        "surface": "AB",
        "log": str(LOGS / f"{precision}_server.log"),
        "adapter_sha256": json.loads(source("completion").read_text())[
            "serving_sha256"
        ],
    }
    write_json(ACTIVE / "server.json", active)
    write_json(destination / "server.json", active)
    try:
        started = time.perf_counter()
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010", timeout=10, trust_env=False
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"JudgeDeceiver server exited: {process.returncode}"
                    )
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("A/B serving startup exceeded twenty minutes")
                await asyncio.sleep(2)
        canary, _ = await score_batch(
            read_rows(DATA / "canary_workload.jsonl"),
            {**C["evaluation"], "concurrency": 4},
        )
        reference = json.loads((OUT / "reference.json").read_text())["cells"]
        gate = {
            name: agreement(
                [r["score"] for r in canary],
                reference[name],
                reference["base"],
                C["parity"],
            )
            for name in ("master", "merged")
        }
        write_json(destination / "parity.json", gate)
        write_json(destination / "canary_predictions.json", canary)
        archive_audits(destination / "native_audits")
        if precision == "bf16":
            audit = json.loads((destination / "bf16_audit.json").read_text())
            if (
                not audit["passed"]
                or len(audit["attention_calls"]) != 8
                or audit["head"]["decision_ids"] != [32, 33]
            ):
                raise ValueError("BF16 A/B native audit incomplete")
        else:
            audits = destination / "native_audits"
            audit = json.loads(
                (audits / "native_attention_projections.json").read_text()
            )
            head = json.loads((audits / "monitor_score.json").read_text())
            if (
                not audit["passed"]
                or len(audit["calls"]) != 16
                or not head["passed"]
                or head["decision_token_ids"] != [32, 33]
            ):
                raise ValueError("optimized A/B native audit incomplete")
        print("judge_serving_parity", precision, gate, flush=True)
        diagnostic = C["evaluation"]["failed_parity_diagnostic"]
        evaluation_gate(gate, diagnostic=diagnostic)
        active.update(status="ready", ready_at_unix=time.time())
        write_json(ACTIVE / "server.json", active)
        write_json(destination / "server.json", active)
        workload = read_rows(DATA / "test_workload.jsonl")
        originals = read_rows(source("test"))
        values, times = [], []
        for start in range(0, len(workload), C["evaluation"]["batch_rows"]):
            batch, seconds = await score_batch(
                workload[start : start + C["evaluation"]["batch_rows"]], C["evaluation"]
            )
            write_json(destination / "batches" / f"{start:05d}.json", batch)
            values.extend(batch)
            times.append(seconds)
            write_json(
                OUT / "status.json",
                {
                    "stage": "scoring",
                    "precision": precision,
                    "rows": len(values),
                    "total": 4188,
                },
            )
            print("judge_progress", precision, len(values), 4188, flush=True)
        predictions = bind_predictions(values, originals)
        write_rows(destination / "preferences.jsonl", predictions)
        write_json(destination / "summary.json", summarize_preferences(predictions))
        latencies = [r["latency_seconds"] for r in values]
        write_json(
            destination / "timing.json",
            {
                "seconds": sum(times),
                "batch_seconds": times,
                "rows": len(values),
                "tokens": 1521530,
                "input_tokens_per_second": 1521530 / sum(times),
                "requests_per_second": len(values) / sum(times),
                "latency_p50_p95": np.percentile(latencies, [50, 95]).tolist(),
            },
        )
        write_json(
            destination / "complete.json",
            {
                "passed": True,
                "rows": 4188,
                "scope": "failed_parity_diagnostic" if diagnostic else "parity_gated",
                "parity_passed": all(g["passed"] for g in gate.values()),
                "surface": "AB",
                "decision_ids": [32, 33],
                "prediction_sha256": file_hash(destination / "preferences.jsonl"),
                "manifest_sha256": file_hash(DATA / "manifest.json"),
            },
        )
        check()
        print("judge_complete", precision, flush=True)
    except BaseException as error:
        write_json(destination / "failure.json", {"error": str(error)})
        retire(precision + "_failed", json.loads((ACTIVE / "server.json").read_text()))
        raise


def bind_predictions(values: list[dict], inputs: list[dict]) -> list[dict]:
    """Never lose preference labels, grouping or paired clean counterparts."""
    if len(values) != len(inputs) or len({v["id"] for v in values}) != len(values):
        raise ValueError("JudgeDeceiver output coverage changed")
    if [v["id"] for v in values] != [r["id"] for r in inputs]:
        raise ValueError("JudgeDeceiver output identity/order changed")
    return [
        {
            **{k: v for k, v in row.items() if k != "student_prompt"},
            **value,
            "p_B": value["score"],
        }
        for row, value in zip(inputs, values, strict=True)
    ]


def report() -> None:
    populations = {
        p: read_rows(OUT / p / "preferences.jsonl") for p in ("bf16", "optimized")
    }
    x, y = [
        np.array([r["score"] for r in populations[p]]) for p in ("bf16", "optimized")
    ]
    paired = {}
    for condition in ("all", "clean", "preferred_injected", "disfavored_injected"):
        mask = np.array(
            [
                condition == "all" or r["condition"] == condition
                for r in populations["bf16"]
            ]
        )
        if not mask.any():
            continue
        a, b = x[mask], y[mask]
        paired[condition] = {
            "rows": len(a),
            "mae": float(np.abs(b - a).mean()),
            "mean_score_delta": float((b - a).mean()),
            "correlation": float(np.corrcoef(a, b)[0, 1]),
            "decision_flips": int(((a >= 0.5) != (b >= 0.5)).sum()),
        }
    write_json(
        OUT / "summary.json",
        {
            "metrics": {p: summarize_preferences(r) for p, r in populations.items()},
            "paired_backends": paired,
            "surface": "AB",
            "rows_per_backend": 4188,
            "historical_context": {
                name: summarize_preferences(read_rows(source(name)))
                for name in ("historical_regular", "historical_augmented")
            },
            "master_sha256": C["inputs"]["master"]["sha256"],
            "manifest_sha256": file_hash(DATA / "manifest.json"),
        },
    )
    write_json(OUT / "status.json", {"stage": "complete", "rows": 8376})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "reference", "all", "report", "diagnostic"),
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    try:
        if args.stage == "reference":
            reference()
        elif args.stage == "report":
            report()
        elif args.stage == "diagnostic":
            check(model=True)
            if not C["evaluation"]["failed_parity_diagnostic"]:
                raise ValueError("diagnostic scope was not frozen")
            if (ACTIVE / "server.json").exists():
                raise ValueError("retire the failed scorer before diagnostic startup")
            retired = json.loads(source("retired_candidate").read_text())
            if retired["status"] != "retired" or any(
                Path(f"/proc/{retired[k]}").exists() for k in ("pid", "worker_pid")
            ):
                raise ValueError("failed scorer retirement is unverified")
            original = ROOT / C["reuse_evaluation_root"]
            old_manifest = json.loads(source("reuse_manifest").read_text())
            if any(
                old_manifest["files"][C["inputs"][name]["path"]]
                != C["inputs"][name]["sha256"]
                for name in ("test", "canary", "master", "master_config", "merge")
            ):
                raise ValueError("reused A/B control model/population changed")
            saved_reference = json.loads(source("completed_reference").read_text())
            if (
                not saved_reference["merged_parity"]["passed"]
                or saved_reference["manifest_sha256"]
                != file_hash(source("reuse_manifest"))
                or saved_reference["ids"]
                != [r["id"] for r in read_rows(DATA / "canary_workload.jsonl")]
            ):
                raise ValueError("reused reference identity/gate changed")
            shutil.copyfile(source("completed_reference"), OUT / "reference.json")
            shutil.copytree(original / "bf16", OUT / "bf16")
            write_json(
                OUT / "reused_controls.json",
                {
                    "source": C["reuse_evaluation_root"],
                    "manifest_sha256": file_hash(source("reuse_manifest")),
                    "reference_sha256": file_hash(source("completed_reference")),
                    "original_failed_gate_sha256": file_hash(
                        source("original_failed_gate")
                    ),
                    "scope": "user_authorized_failed_parity_diagnostic",
                },
            )
            asyncio.run(score("optimized"))
            report()
        else:
            check(model=True)
            if (ACTIVE / "server.json").exists():
                retire("parent", json.loads(source("resident").read_text()))
            else:
                previous = json.loads(source("retired_parent").read_text())
                resident = json.loads(source("resident").read_text())
                if (
                    previous["status"] != "retired"
                    or previous["pid"] != resident["pid"]
                    or Path(f"/proc/{previous['pid']}").exists()
                    or Path(f"/proc/{previous['worker_pid']}").exists()
                ):
                    raise ValueError("previous parent retirement is unverified")
            write_json(OUT / "status.json", {"stage": "references"})
            from gleipnir.campaigns.runtime import training_environment

            env = training_environment(ROOT, C["campaign_id"])
            env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:{env.get('PYTHONPATH', '')}"
            subprocess.run(
                [
                    C["training_python"],
                    "-m",
                    "experiments.b200_augmented_judge.run",
                    "--stage",
                    "reference",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            asyncio.run(score("bf16"))
            retire("bf16", json.loads((OUT / "bf16/server.json").read_text()))
            asyncio.run(score("optimized"))
            report()
    except BaseException as error:
        write_json(OUT / "status.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
