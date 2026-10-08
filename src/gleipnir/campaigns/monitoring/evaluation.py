"""Adapter-specific merged parity and fixed optimized ID/APPS evaluation."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    write_json,
    write_rows,
)
from gleipnir.evaluation.apps import summarize_apps
from gleipnir.evaluation.binary import metric_views
from gleipnir.evaluation.calibration import binary_calibration
from gleipnir.evaluation.campaign import EvaluationContext, canary, render
from gleipnir.evaluation.http_score import score_batch

from .contract import Campaign


def agreement(
    scores: list[float], reference: list[float], base: list[float], limits: dict
) -> dict:
    """Require a new adapter's predeclared numerical gate, preserving failures."""
    actual, expected, original = map(
        lambda x: np.asarray(x, dtype=float), (scores, reference, base)
    )
    if (
        actual.ndim != 1
        or actual.shape != expected.shape
        or actual.shape != original.shape
        or not actual.size
    ):
        raise ValueError("new-adapter parity coverage changed")
    finite = bool(
        all(
            np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
            for a in (actual, expected, original)
        )
    )
    mae = float(np.abs(actual - expected).mean()) if finite else None
    correlation = (
        float(np.corrcoef(actual, expected)[0, 1])
        if finite and actual.std() > 0 and expected.std() > 0
        else None
    )
    effect = float(np.abs(actual - original).max()) if finite else None
    return {
        "passed": bool(
            finite
            and mae <= limits["max_mean_absolute_difference"]
            and correlation is not None
            and np.isfinite(correlation)
            and correlation >= limits["min_correlation"]
            and effect > limits["min_adapter_effect"]
        ),
        "finite": finite,
        "mean_absolute_difference": mae,
        "correlation": correlation
        if correlation is not None and np.isfinite(correlation)
        else None,
        "maximum_absolute_difference": float(np.abs(actual - expected).max())
        if finite
        else None,
        "adapter_effect": effect,
        "limits": limits,
    }


def canary_workload(ctx: Campaign) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    config = ctx.config
    context = EvaluationContext(ctx.data, ctx.output, ("monitor",), {})
    rows = canary(context, "monitor", config["parity"]["rows_per_source_label"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["merged_model"], local_files_only=True
    )
    prompts = render(tokenizer, rows)
    master = json.loads((ctx.adapter / "parity_reference.json").read_text())
    if [digest(p) for p in prompts] != master["prompt_sha256"] or master[
        "master_sha256"
    ] != file_hash(ctx.adapter / "causal_adapter/adapter_model.safetensors"):
        raise ValueError("new-adapter canary prompt/master identity changed")
    workload = [
        {
            "id": f"canary-{i}",
            "prompt": text,
            "prompt_sha256": digest(text),
            "prompt_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
        }
        for i, text in enumerate(prompts)
    ]
    return workload, master


def merged_reference(ctx: Campaign) -> None:
    """Check BF16 merged inference against the FP32 causal master independently."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.deception_distillation.train_student_sft import (
        gated_delta_kernel_modules,
    )
    from experiments.training_procedure_screen.evaluate_causal import score_adapter

    config = ctx.config
    ctx.check()
    workload, master = canary_workload(ctx)
    tokenizer = AutoTokenizer.from_pretrained(
        config["merged_model"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["merged_model"],
        local_files_only=True,
        dtype="bfloat16",
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    kernels = gated_delta_kernel_modules(model)
    if not kernels or any(not name.startswith("fla.ops.") for name in kernels):
        raise ValueError("merged reference requires original pinned FLA kernels")
    tokenized = [
        tokenizer.encode(r["prompt"], add_special_tokens=False) for r in workload
    ]
    scores, details = score_adapter(
        model,
        tokenizer,
        tokenized,
        [15, 16],
        batch_size=1,
        decision_head_mode="token_logits",
    )
    receipt = agreement(scores, master["adapter"], master["base"], config["parity"])
    receipt.update(
        scores=scores,
        master_sha256=master["master_sha256"],
        details=details,
        merged_artifact_sha256=file_hash(ctx.output / "merged_artifact.json"),
    )
    write_json(ctx.output / "merged_parity.json", receipt)
    write_rows(ctx.output / "canary_workload.jsonl", workload)
    print(
        "campaign_merged_parity",
        receipt["passed"],
        receipt["mean_absolute_difference"],
        receipt["correlation"],
        flush=True,
    )
    if not receipt["passed"]:
        raise ValueError("new-adapter master/merged BF16 parity failed")


def id_metrics(rows: list[dict]) -> dict:
    """Use canonical source-macro raw partial AUC with calibration diagnostics."""
    frame = pd.DataFrame(
        {
            "dataset": [r["source_dataset"] for r in rows],
            "label": [r["ground_truth"] for r in rows],
            "score": [r["score"] for r in rows],
        }
    )
    return {
        "metrics": metric_views(frame),
        "calibration": {
            name: binary_calibration(list(group["label"]), list(group["score"]))
            for name, group in [
                ("pooled", frame),
                *[(name, group) for name, group in frame.groupby("dataset")],
            ]
        },
    }


def attach_inputs(values: list[dict], inputs: list[dict]) -> list[dict]:
    if len(values) != len(inputs) or len({r["id"] for r in values}) != len(values):
        raise ValueError("prediction population/uniqueness changed")
    output = []
    for value, row in zip(values, inputs, strict=True):
        if value["id"] != row["id"]:
            raise ValueError("prediction order/identity changed")
        if any(k in value and value[k] != v for k, v in row["metadata"].items()):
            raise ValueError("saved prediction label/source metadata changed")
        output.append({**value, **row["metadata"]})
    return output


def frozen_workloads(ctx: Campaign) -> dict[str, list[dict]]:
    """Reuse complete checked rendered prompts, never reselect evaluation cases."""
    workloads = validate_workloads(ctx)
    # Frozen prompts were rendered from these exact tokenizer assets. New
    # adapters change weights, not the pinned model's tokenizer/chat template.
    selection = json.loads(ctx.input("serving_selection").read_text())
    old_merge = json.loads((ctx.root / selection["merged_artifact"]).read_text())
    merged = json.loads((ctx.output / "merged_artifact.json").read_text())
    for name, expected in old_merge["files_sha256"].items():
        if (
            "safetensors" not in name
            and name != "config.json"
            and merged["files_sha256"].get(name) != expected
        ):
            raise ValueError(f"frozen tokenizer/model asset changed: {name}")
    return workloads


async def optimized(ctx: Campaign) -> None:
    """Load the new merged weights with the unchanged selected compiled recipe."""
    from vllm.v1.core.sched import scheduler

    from experiments.b200_vllm031.run import archive_audits
    from experiments.b200_vllm031.runtime import candidate_environment
    from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources
    from gleipnir.serving.reference import selected_serving_default

    config = ctx.config
    diagnostic = config["evaluation"].get("failed_parity_diagnostic", False)
    bf16 = config["evaluation"].get("serving_precision", "optimized") == "bf16"
    ctx.check()
    merged = json.loads((ctx.output / "merged_parity.json").read_text())
    if not merged["passed"]:
        raise ValueError("optimized evaluation requires passed merged/master parity")
    selection, command = selected_serving_default(ctx.root)
    if {n: importlib.metadata.version(n) for n in selection["runtime"]} != selection[
        "runtime"
    ]:
        raise ValueError("selected serving runtime changed")
    scheduler_binding = {
        "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
        "integration_sha256": file_hash(
            ctx.root / "experiments/b200_vllm031/scheduler.py"
        ),
    }
    if scheduler_binding != selection["scheduler_binding"]:
        raise ValueError("selected pooling scheduler changed")
    if (ctx.serving / "server.json").exists():
        raise ValueError("retire old scorer before loading the new adapter")
    driver = int(os.environ.get("GLEIPNIR_CAMPAIGN_DRIVER_PID", "-1"))
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).split()
    if any(int(pid) not in (driver, os.getpid()) for pid in pids):
        raise ValueError("unrelated model remains on GPU")
    command[0] = sys.executable
    command[command.index("--model") + 1] = config["merged_model"]
    if bf16:
        from gleipnir.serving.bf16_monitor import command as bf16_command

        command = bf16_command(command, ctx.sources())
    additional_index = command.index("--additional-config") + 1
    additional = json.loads(command[additional_index])
    additional["serving_condition"]["merged_model"] = config["merged_model"]
    command[additional_index] = json.dumps(additional, sort_keys=True)
    restore_diagnostic_sources(
        ctx.root,
        additional["serving_condition"],
        ctx.root / "results/b200_vllm031/pre_migration_sources.tar.gz",
    )
    env = candidate_environment(ctx.root)
    parent = json.loads((ctx.root / selection["host_parent"]).read_text())
    env.update(
        GLEIPNIR_FROST_WRAPPER_VALIDATION=parent["host_wrapper"]["validation"],
        GLEIPNIR_GIGATOKEN_RECEIPT=str(ctx.output / "frontend.json"),
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(scheduler_binding),
    )
    if bf16:
        env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
        env["GLEIPNIR_BF16_AUDIT"] = str(ctx.output / "bf16_audit.json")
    ctx.logs.mkdir(parents=True, exist_ok=True)
    with (ctx.logs / "server.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ctx.root,
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
        "log": str(ctx.logs / "server.log"),
        "serving_default": selection["name"],
        "adapter_sha256": json.loads((ctx.adapter / "complete.json").read_text())[
            "serving_sha256"
        ],
        "frontend": parent["frontend"],
        "host_wrapper": None if bf16 else parent["host_wrapper"],
        "serving_precision": "bf16" if bf16 else "optimized",
    }
    write_json(ctx.serving / "server.json", active)
    write_json(ctx.output / "server.json", active)
    try:
        started = time.perf_counter()
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{config['evaluation']['port']}",
            trust_env=False,
            timeout=30,
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"campaign serving process exited: {process.returncode}"
                    )
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError(
                        "new-adapter serving startup exceeded twenty minutes"
                    )
                await asyncio.sleep(2)
        workload = read_rows(ctx.output / "canary_workload.jsonl")
        values, _ = await score_batch(
            workload, {**config["evaluation"], "concurrency": 4}
        )
        master = json.loads((ctx.adapter / "parity_reference.json").read_text())
        gate = agreement(
            [v["score"] for v in values],
            master["adapter"],
            master["base"],
            config["parity"],
        )
        gate["versus_merged_bf16"] = agreement(
            [v["score"] for v in values],
            merged["scores"],
            master["base"],
            config["parity"],
        )
        gate["serving_precision"] = "bf16" if bf16 else "optimized"
        gate["evaluation_scope"] = (
            "failed_parity_diagnostic" if diagnostic else "parity_gated"
        )
        write_json(ctx.output / "optimized_parity.json", gate)
        write_json(ctx.output / "optimized_canary_predictions.json", values)
        if bf16:
            native = json.loads((ctx.output / "bf16_audit.json").read_text())
            if not native["passed"] or len(native["attention_calls"]) != 8:
                raise ValueError("BF16 model/native attention audit incomplete")
        else:
            archive_audits(ctx.output / "native_audits")
            native = json.loads(
                (
                    ctx.output / "native_audits/native_attention_projections.json"
                ).read_text()
            )
            if (
                not native["passed"]
                or native["precision"] != "fp8"
                or len(native["calls"]) != 16
            ):
                raise ValueError("actual FP8 projection dispatch changed")
        require_evaluation_gate(gate, diagnostic=diagnostic)
        active.update(status="ready", ready_at_unix=time.time())
        write_json(ctx.serving / "server.json", active)
        write_json(ctx.output / "server.json", active)
        print(
            "campaign_optimized_parity",
            gate["mean_absolute_difference"],
            gate["correlation"],
            flush=True,
        )
        workloads = frozen_workloads(ctx)
        populations = {}
        for split, rows in workloads.items():
            all_values, times = [], []
            for start in range(0, len(rows), config["evaluation"]["batch_rows"]):
                batch = rows[start : start + config["evaluation"]["batch_rows"]]
                values, seconds = await score_batch(batch, config["evaluation"])
                write_json(
                    ctx.output / "evaluation/batches" / split / f"{start:05d}.json",
                    values,
                )
                all_values.extend(values)
                times.append(seconds)
                print(
                    "campaign_evaluation_progress",
                    split,
                    len(all_values),
                    len(rows),
                    flush=True,
                )
            scored = attach_inputs(all_values, read_rows(ctx.input(split)))
            write_rows(ctx.output / "evaluation" / f"{split}.jsonl", scored)
            populations[split] = scored
            write_json(
                ctx.output / "evaluation" / f"{split}_timing.json",
                {
                    "seconds": sum(times),
                    "batch_seconds": times,
                    "rows": len(scored),
                    "tokens": sum(r["prompt_tokens"] for r in rows),
                },
            )
        summary = summarize(populations, ctx)
        write_json(ctx.output / "summary.json", summary)
        write_json(
            ctx.output / "evaluation/complete.json",
            {
                "rows": sum(len(v) for v in populations.values()),
                "evaluation_scope": "failed_parity_diagnostic"
                if diagnostic
                else "parity_gated",
                "parity_passed": gate["passed"]
                and gate["versus_merged_bf16"]["passed"],
                "config_sha256": file_hash(ctx.config_path),
                "master_sha256": master["master_sha256"],
                "files_sha256": {
                    s: file_hash(ctx.output / "evaluation" / f"{s}.jsonl")
                    for s in populations
                },
            },
        )
        print(
            "campaign_evaluation_complete",
            list(summary["evaluations"]),
            flush=True,
        )
    except BaseException as error:
        write_json(
            ctx.output / "evaluation_failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        if process.poll() is None:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.b200_vllm031.stop",
                    "--archive-name",
                    config["campaign_id"] + "_failed",
                ],
                cwd=ctx.root,
                env=env,
                check=True,
            )
        raise


def validate_workloads(ctx: Campaign) -> dict[str, list[dict]]:
    """Check ordered identity, metadata and cached rendered prompts on CPU."""
    workloads = {
        s: (
            json.loads(ctx.input(s + "_workload").read_text())
            if s == "id"
            else read_rows(ctx.input(s + "_workload"))
        )
        for s in ctx.splits
    }
    for split, rows in workloads.items():
        canonical = read_rows(ctx.input(split))
        if (
            len(canonical) != ctx.config["evaluation"]["populations"][split]
            or len(rows) != len(canonical)
            or len({r["id"] for r in rows}) != len(rows)
        ):
            raise ValueError("frozen workload coverage changed")
        for row, source in zip(rows, canonical, strict=True):
            if (
                row["id"] != source["id"]
                or digest(row["prompt"]) != row["prompt_sha256"]
                or not 0 < row["prompt_tokens"] < 32768
            ):
                raise ValueError("frozen rendered prompt/context identity changed")
            if split == "id":
                meta = source["metadata"]
                if (
                    digest(source["prompt"]) != row["source_prompt_sha256"]
                    or row["dataset"] != meta["source_dataset"]
                    or row["label"] != meta["ground_truth"]
                ):
                    raise ValueError("canonical ID lineage/source/label drift")
            elif row["metadata"] != source["metadata"]:
                raise ValueError("APPS payload/label metadata drift")
    train = read_rows(ctx.input("training"))
    seen = {r["trajectory_sha256"] for r in train}
    if ctx.config.get("augmentation_audit", False):
        seen |= {r["trajectory_sha256"] for r in read_rows(ctx.input("clean_training"))}
    for split in ctx.splits:
        for row in read_rows(ctx.input(split)):
            meta = row["metadata"]
            candidates = [
                meta.get("trajectory_sha256"),
                meta.get("original_trajectory_sha256"),
                meta.get("original_metadata", {}).get("trajectory_sha256"),
            ]
            if seen.intersection(x for x in candidates if x):
                raise ValueError(
                    f"training/held-out trajectory overlap: {split}/{row['id']}"
                )
    return workloads


def summarize(populations: dict[str, list[dict]], ctx: Campaign) -> dict:
    """Registered held-out metrics; baselines remain explicit optional controls."""
    ctx.check()
    evaluations = {}
    if "id" in ctx.config["evaluations"]:
        evaluations["id"] = id_metrics(populations["id"])
    if "apps" in ctx.config["evaluations"]:
        evaluations["apps"] = summarize_apps(
            populations["benchmark"], populations["honest_controls"]
        )
    baselines = {}
    for name, spec in ctx.config.get("baselines", {}).items():
        if spec["kind"] == "id":
            path = ctx.input(spec["input"])
            rows = (
                json.loads(path.read_text())
                if path.suffix == ".json"
                else read_rows(path)
            )
            rows = attach_inputs(rows, read_rows(ctx.input("id")))
            baselines[name] = {
                "kind": "id",
                "qualification": spec["qualification"],
                "metrics": id_metrics(rows),
            }
        elif spec["kind"] == "apps":
            data = json.loads(ctx.input(spec["input"]).read_text())
            for key in spec.get("keys", []):
                data = data[key]
            baselines[name] = {
                "kind": "apps",
                "qualification": spec["qualification"],
                "metrics": data,
            }
        else:
            raise ValueError(f"unknown baseline kind: {spec['kind']}")
    return {
        "status": "complete",
        "evaluation_scope": "failed_parity_diagnostic"
        if ctx.config["evaluation"].get("failed_parity_diagnostic", False)
        else "parity_gated",
        "parity": json.loads((ctx.output / "optimized_parity.json").read_text()),
        "evaluations": evaluations,
        "baselines": baselines,
        "identity": {
            "config_sha256": file_hash(ctx.config_path),
            "manifest_sha256": file_hash(ctx.data / "manifest.json"),
            "adapter": json.loads((ctx.adapter / "complete.json").read_text()),
            "merged_artifact_sha256": file_hash(ctx.output / "merged_artifact.json"),
        },
    }


def require_evaluation_gate(gate: dict, *, diagnostic: bool) -> None:
    """Explicit diagnostics may exceed MAE, never finite/ranking/effect guards."""
    for value in (gate, gate["versus_merged_bf16"]):
        if value["passed"]:
            continue
        if (
            not diagnostic
            or not value["finite"]
            or value["correlation"] is None
            or value["correlation"] < value["limits"]["min_correlation"]
            or value["adapter_effect"] <= value["limits"]["min_adapter_effect"]
        ):
            raise ValueError("new-adapter optimized serving score agreement failed")
