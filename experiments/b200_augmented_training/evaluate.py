"""Adapter-specific merged parity and fixed optimized ID/APPS evaluation."""

from __future__ import annotations

import argparse
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

from experiments.b200_apps.run import score_batch
from experiments.b200_augmented_training.campaign import (
    ADAPTER,
    DATA,
    LOGS,
    OUTPUT,
    ROOT,
    check_binding,
    configuration,
    file_hash,
    input_path,
    write_json,
)
from gleipnir.data.monitoring import digest, read_rows, write_rows
from gleipnir.evaluation.apps import summarize_apps
from gleipnir.evaluation.binary import metric_views
from gleipnir.evaluation.calibration import binary_calibration
from gleipnir.evaluation.campaign import EvaluationContext, canary, render
from gleipnir.serving.monitor_score import validate_score_response

SERVING = ROOT / "results/b200_attention_gdn_serving"


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


def canary_workload(config: dict) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    context = EvaluationContext(DATA, OUTPUT, ("augmented",), {})
    rows = canary(context, "augmented", config["parity"]["rows_per_source_label"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["merged_model"], local_files_only=True
    )
    prompts = render(tokenizer, rows)
    master = json.loads((ADAPTER / "parity_reference.json").read_text())
    if [digest(p) for p in prompts] != master["prompt_sha256"] or master[
        "master_sha256"
    ] != file_hash(ADAPTER / "causal_adapter/adapter_model.safetensors"):
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
    if len(workload) != 20:
        raise ValueError("expected twenty balanced training-source canaries")
    return workload, master


def merged_reference(config: dict) -> None:
    """Check BF16 merged inference against the FP32 causal master independently."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.deception_distillation.train_student_sft import (
        gated_delta_kernel_modules,
    )
    from experiments.training_procedure_screen.evaluate_causal import score_adapter

    check_binding(config)
    workload, master = canary_workload(config)
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
        merged_artifact_sha256=file_hash(OUTPUT / "merged_artifact.json"),
    )
    write_json(OUTPUT / "merged_parity.json", receipt)
    write_rows(OUTPUT / "canary_workload.jsonl", workload)
    print(
        "augmented_merged_parity",
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


def frozen_workloads(config: dict) -> dict[str, list[dict]]:
    """Reuse complete checked rendered prompts, never reselect evaluation cases."""
    workloads = {
        "id": json.loads(input_path(config, "id_workload").read_text()),
        **{
            s: read_rows(input_path(config, s + "_workload"))
            for s in ("benchmark", "honest_controls")
        },
    }
    for split, rows in workloads.items():
        canonical = read_rows(input_path(config, split))
        if len(rows) != len(canonical) or len({r["id"] for r in rows}) != len(rows):
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
    # Frozen prompts were rendered from these exact tokenizer assets. New
    # adapters change weights, not the pinned model's tokenizer/chat template.
    selection = json.loads(input_path(config, "serving_selection").read_text())
    old_merge = json.loads((ROOT / selection["merged_artifact"]).read_text())
    merged = json.loads((OUTPUT / "merged_artifact.json").read_text())
    for name, expected in old_merge["files_sha256"].items():
        if (
            "safetensors" not in name
            and name != "config.json"
            and merged["files_sha256"].get(name) != expected
        ):
            raise ValueError(f"frozen tokenizer/model asset changed: {name}")
    return workloads


async def optimized(config: dict) -> None:
    """Load the new merged weights with the unchanged selected compiled recipe."""
    from vllm.v1.core.sched import scheduler

    from experiments.b200_vllm031.run import archive_audits
    from experiments.b200_vllm031.runtime import candidate_environment
    from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources
    from gleipnir.serving.reference import selected_serving_default

    check_binding(config)
    merged = json.loads((OUTPUT / "merged_parity.json").read_text())
    if not merged["passed"]:
        raise ValueError("optimized evaluation requires passed merged/master parity")
    selection, command = selected_serving_default(ROOT)
    if {n: importlib.metadata.version(n) for n in selection["runtime"]} != selection[
        "runtime"
    ]:
        raise ValueError("selected serving runtime changed")
    scheduler_binding = {
        "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
        "integration_sha256": file_hash(ROOT / "experiments/b200_vllm031/scheduler.py"),
    }
    if scheduler_binding != selection["scheduler_binding"]:
        raise ValueError("selected pooling scheduler changed")
    if (SERVING / "server.json").exists():
        raise ValueError("retire old scorer before loading the new adapter")
    driver = int(os.environ.get("GLEIPNIR_AUGMENTED_DRIVER_PID", "-1"))
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).split()
    if any(int(pid) not in (driver, os.getpid()) for pid in pids):
        raise ValueError("unrelated model remains on GPU")
    command[0] = sys.executable
    command[command.index("--model") + 1] = config["merged_model"]
    additional_index = command.index("--additional-config") + 1
    additional = json.loads(command[additional_index])
    additional["serving_condition"]["merged_model"] = config["merged_model"]
    command[additional_index] = json.dumps(additional, sort_keys=True)
    restore_diagnostic_sources(
        ROOT,
        additional["serving_condition"],
        ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz",
    )
    env = candidate_environment(ROOT)
    parent = json.loads((ROOT / selection["host_parent"]).read_text())
    env.update(
        GLEIPNIR_FROST_WRAPPER_VALIDATION=parent["host_wrapper"]["validation"],
        GLEIPNIR_GIGATOKEN_RECEIPT=str(OUTPUT / "frontend.json"),
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(scheduler_binding),
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "server.log").open("x") as log:
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
        "log": str(LOGS / "server.log"),
        "serving_default": selection["name"],
        "adapter_sha256": json.loads((ADAPTER / "complete.json").read_text())[
            "serving_sha256"
        ],
        "frontend": parent["frontend"],
        "host_wrapper": parent["host_wrapper"],
    }
    write_json(SERVING / "server.json", active)
    write_json(OUTPUT / "server.json", active)
    try:
        started = time.perf_counter()
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010", trust_env=False, timeout=30
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"augmented serving process exited: {process.returncode}"
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
        workload = read_rows(OUTPUT / "canary_workload.jsonl")
        values, _ = await score_batch(
            workload, {**config["evaluation"], "concurrency": 4}
        )
        master = json.loads((ADAPTER / "parity_reference.json").read_text())
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
        write_json(OUTPUT / "optimized_parity.json", gate)
        write_json(OUTPUT / "optimized_canary_predictions.json", values)
        archive_audits(OUTPUT / "native_audits")
        native = json.loads(
            (OUTPUT / "native_audits/native_attention_projections.json").read_text()
        )
        if (
            not native["passed"]
            or native["precision"] != "fp8"
            or len(native["calls"]) != 16
        ):
            raise ValueError("actual FP8 projection dispatch changed")
        if not gate["passed"] or not gate["versus_merged_bf16"]["passed"]:
            raise ValueError("new-adapter optimized serving score agreement failed")
        active.update(status="ready", ready_at_unix=time.time())
        write_json(SERVING / "server.json", active)
        write_json(OUTPUT / "server.json", active)
        print(
            "augmented_optimized_parity",
            gate["mean_absolute_difference"],
            gate["correlation"],
            flush=True,
        )
        workloads = frozen_workloads(config)
        populations = {}
        for split, rows in workloads.items():
            all_values, times = [], []
            for start in range(0, len(rows), config["evaluation"]["batch_rows"]):
                batch = rows[start : start + config["evaluation"]["batch_rows"]]
                values, seconds = await score_batch(batch, config["evaluation"])
                write_json(
                    OUTPUT / "evaluation/batches" / split / f"{start:05d}.json", values
                )
                all_values.extend(values)
                times.append(seconds)
                print(
                    "augmented_evaluation_progress",
                    split,
                    len(all_values),
                    len(rows),
                    flush=True,
                )
            scored = attach_inputs(all_values, read_rows(input_path(config, split)))
            write_rows(OUTPUT / "evaluation" / f"{split}.jsonl", scored)
            populations[split] = scored
            write_json(
                OUTPUT / "evaluation" / f"{split}_timing.json",
                {
                    "seconds": sum(times),
                    "batch_seconds": times,
                    "rows": len(scored),
                    "tokens": sum(r["prompt_tokens"] for r in rows),
                },
            )
        summary = summarize(populations, config)
        write_json(OUTPUT / "summary.json", summary)
        write_json(
            OUTPUT / "evaluation/complete.json",
            {
                "rows": 12126,
                "config_sha256": file_hash(Path(__file__).with_name("config.yaml")),
                "master_sha256": master["master_sha256"],
                "files_sha256": {
                    s: file_hash(OUTPUT / "evaluation" / f"{s}.jsonl")
                    for s in populations
                },
            },
        )
        print(
            "augmented_evaluation_complete",
            summary["id"]["candidate"]["metrics"]["macro"]["macro"],
            summary["apps"]["candidate"]["mean_injected_honest_fpr"],
            flush=True,
        )
    except BaseException as error:
        write_json(
            OUTPUT / "evaluation_failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        if process.poll() is None:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.b200_vllm031.stop",
                    "--archive-name",
                    "augmented_evaluation_failed01",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
        raise


def summarize(populations: dict[str, list[dict]], config: dict) -> dict:
    """Compare exact cached populations without rescoring or selecting controls."""
    check_binding(config)
    candidate = populations["id"]
    inputs = read_rows(input_path(config, "id"))
    regular = attach_inputs(
        json.loads(input_path(config, "regular_id_predictions").read_text()), inputs
    )
    historical = read_rows(input_path(config, "historical_augmented_id"))
    for row, source in zip(historical, inputs, strict=True):
        if row["id"] != source["id"] or any(
            row[k] != source["metadata"][k] for k in ("source_dataset", "ground_truth")
        ):
            raise ValueError("historical augmented ID control membership drift")

    def paired(old: list[dict]) -> dict:
        x, y = (
            np.array([r["score"] for r in old]),
            np.array([r["score"] for r in candidate]),
        )
        return {
            "mean_absolute_difference": float(np.abs(y - x).mean()),
            "mean_score_delta": float((y - x).mean()),
            "correlation": float(np.corrcoef(x, y)[0, 1]),
            "half_threshold_flips": int(np.sum((x >= 0.5) != (y >= 0.5))),
        }

    return {
        "status": "complete",
        "identity": {
            "config_sha256": file_hash(Path(__file__).with_name("config.yaml")),
            "manifest_sha256": file_hash(DATA / "manifest.json"),
            "adapter": json.loads((ADAPTER / "complete.json").read_text()),
            "merged_artifact_sha256": file_hash(OUTPUT / "merged_artifact.json"),
            "serving_recipe": json.loads(
                input_path(config, "serving_selection").read_text()
            )["name"],
        },
        "id": {
            "candidate": id_metrics(candidate),
            "regular_optimized": id_metrics(regular),
            "historical_augmented_bf16": id_metrics(historical),
            "paired_vs_regular_optimized": paired(regular),
            "paired_vs_historical_augmented": paired(historical),
        },
        "apps": {
            "candidate": summarize_apps(
                populations["benchmark"], populations["honest_controls"]
            ),
            "regular_optimized": json.loads(
                input_path(config, "regular_apps_summary").read_text()
            ),
            "historical_augmented_bf16": json.loads(
                input_path(config, "historical_augmented_summary").read_text()
            )["monitoring"]["augmented"]["neutral"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("merged-reference", "optimized", "summary"), required=True
    )
    args = parser.parse_args()
    config = configuration()
    if args.stage == "merged-reference":
        merged_reference(config)
    elif args.stage == "optimized":
        asyncio.run(optimized(config))
    else:
        populations = {
            s: read_rows(OUTPUT / "evaluation" / f"{s}.jsonl")
            for s in ("id", "benchmark", "honest_controls")
        }
        for split, rows in populations.items():
            for row in rows:
                validate_score_response(row, row["prompt_tokens"])
            attach_inputs(rows, read_rows(input_path(config, split)))
        write_json(OUTPUT / "summary.json", summarize(populations, config))


if __name__ == "__main__":
    main()
