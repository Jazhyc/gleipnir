"""Materialize the user-selected combined reference from completed measurements."""

import copy
import json
import shutil
from pathlib import Path

from experiments.b200_inference_benchmark.run import EXPERIMENT, ROOT, sha, write


def main() -> None:
    selection_path = EXPERIMENT / "baseline.json"
    parent = json.loads(selection_path.read_text())
    compare = ROOT / "results/b200_attention_gdn_serving/host_wrapper_compare01"
    startup = ROOT / "results/b200_attention_gdn_serving/host_wrapper_start01"
    source = json.loads((compare / "summary.json").read_text())
    initial = json.loads((startup / "summary.json").read_text())
    if (
        source["status"] != "complete"
        or source["selected"] != "direct"
        or source["baseline_sha256"] != sha(selection_path)
    ):
        raise ValueError("completed selected-host reference/parent mismatch")
    archive = EXPERIMENT / "baselines" / f"{parent['name']}.json"
    if archive.exists() or parent.get("frontend"):
        raise FileExistsError("reference already selected or parent archive exists")
    archive.write_bytes(selection_path.read_bytes())
    relative = Path(
        "results/b200_attention_gdn_serving/gigatoken_direct_host_reference01"
    )
    out = ROOT / relative
    out.mkdir(exist_ok=False)
    (out / "executed_selection.py").write_bytes(Path(__file__).read_bytes())
    files = {}
    trials = {1: [], 128: []}
    scores = {1: [], 128: []}
    for t in source["trials"]:
        if t["mode"] != "direct":
            continue
        c, i = t["concurrency"], t["pair"]
        measurement = {
            k: v for k, v in t.items() if k not in {"mode", "pair", "before", "after"}
        }
        measurement["repeat"] = i
        trials[c].append(measurement)
        name = f"c1_repeat{i}.json" if c == 1 else f"high_c128_repeat{i}.json"
        origin = compare / f"c{c}_direct_pair{i}.json"
        shutil.copy2(origin, out / name)
        files[str(origin.relative_to(ROOT))] = sha(origin)
        scores[c].append(json.loads(origin.read_text()))
    if len(trials[1]) != 3 or len(trials[128]) != 6:
        raise ValueError("incomplete combined reference repeats")
    ranges = [
        max(run[i]["score"] for run in scores[1])
        - min(run[i]["score"] for run in scores[1])
        for i in range(64)
    ]
    summary = {
        **initial,
        "status": "complete",
        "trials": trials[1],
        "startup_only": False,
        "timing_sweep_skipped": False,
        "measurements_reused": True,
        "score_variation": {
            "mean_range": sum(ranges) / 64,
            "max_range": max(ranges),
            "threshold_unstable_rows": sum(
                len({run[i]["score"] >= 0.5 for run in scores[1]}) > 1
                for i in range(64)
            ),
        },
    }
    high = {
        "status": "complete",
        "rows": 320,
        "trials": trials[128],
        "manifest_sha256": source["manifest_sha256"],
        "server": source["server"],
        "serving_config_overrides": initial["kernel_condition"][
            "serving_config_overrides"
        ],
        "ranking": source["comparison"]["c128"]["ranking"][
            "candidate_repeat_median_scores"
        ],
        "measurements_reused": True,
    }
    write(out / "summary.json", summary)
    write(out / "high_summary.json", high)
    for filename in ["condition.json", "http_parity.json"]:
        shutil.copy2(startup / filename, out / filename)
    write(out / "selected_server.json", source["server"])
    for path in [
        compare / "summary.json",
        startup / "summary.json",
        startup / "http_parity.json",
        ROOT / source["server"]["frontend"]["validation"],
        ROOT / source["server"]["host_wrapper"]["validation"],
    ]:
        files[str(path.relative_to(ROOT))] = sha(path)
    write(
        out / "provenance.json",
        {
            "source_files_sha256": files,
            "measurement_reuse": "direct-mode completed passes only",
            "user_selection_date": "2026-10-07",
            "final_id_used": False,
        },
    )
    acceptance = {
        "status": "user_accepted_finite",
        "scope": "combined B200 serving reference",
        "selected_at": "2026-10-07",
        "basis": (
            "User selected native Gigatoken with the retained direct FROST host wrapper"
        ),
        "parent_gpu_acceptance": parent["quality_acceptance"],
        "parent_gpu_acceptance_sha256": parent["quality_acceptance_sha256"],
        "strict_native_precision_passed": parent["strict_native_precision_passed"],
        "new_kernel_arithmetic": False,
        "native_host_bitwise_validation": True,
    }
    write(out / "quality_acceptance.json", acceptance)
    new = copy.deepcopy(parent)
    new.update(
        name=parent["name"] + "_gigatoken_direct_host",
        results=str(relative),
        previous_selection=str(archive.relative_to(ROOT)),
        parent_gpu_selection_sha256=sha(archive),
        summary_sha256=sha(out / "summary.json"),
        high_summary_sha256=sha(out / "high_summary.json"),
        server_metadata=str(relative / "selected_server.json"),
        quality_acceptance=str(relative / "quality_acceptance.json"),
        quality_acceptance_sha256=sha(out / "quality_acceptance.json"),
        confirmation_results=str(relative),
        confirmation_summary_sha256=sha(out / "high_summary.json"),
        latency_confirmation_results=str(relative),
        latency_confirmation_summary_sha256=sha(out / "summary.json"),
        comparison_concurrency=[1, 128],
        latency_concurrency=[1],
        throughput_concurrency=[128],
        frontend={
            k: source["server"]["frontend"][k]
            for k in [
                "backend",
                "package_path",
                "validation",
                "validation_sha256",
                "ab_control",
            ]
        },
        host_wrapper={
            k: source["server"]["host_wrapper"][k]
            for k in ["validation", "validation_sha256"]
        },
        host_wrapper_mode="direct",
        warm_reference_tokens_per_second=source["comparison"]["c128"]["tokens_s"][
            "direct"
        ],
        warm_reference_latency_seconds=source["comparison"]["c1"]["latency_p50"][
            "direct"
        ],
        warm_reference_p95_seconds=source["comparison"]["c1"]["latency_p95"]["direct"],
        warm_reference_auroc={
            "macro": high["ranking"]["macro"]["macro"]["auroc"],
            "pooled": high["ranking"]["pooled"]["auroc"],
        },
    )
    write(selection_path, new)
    write(
        out / "checksums.json",
        {
            "files_sha256": {
                str(p.relative_to(ROOT)): sha(p)
                for p in out.rglob("*")
                if p.is_file() and p.name != "checksums.json"
            }
        },
    )
    print(
        json.dumps(
            {
                "selection_sha256": sha(selection_path),
                "results": str(relative),
                "reference_input_tokens_s": new["warm_reference_tokens_per_second"],
            }
        )
    )


if __name__ == "__main__":
    main()
