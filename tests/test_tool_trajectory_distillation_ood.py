import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.compare_distilled_ood_parity import (
    compare_predictions,
)
from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    materialize_rows,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.run_distilled_ood_lambda import (
    parity_report_passed,
)
from experiments.tool_trajectory_monitoring.summarize_distilled_ood import (
    summary_rows,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_materialize_rows_preserves_exact_trajectory_and_uses_student_prompt() -> None:
    prompts = load_prompt_set()
    trajectory = "user: inspect this\nassistant: done"
    teacher_prompt = prompts.teacher.render(trajectory)
    metadata = {
        "source_dataset": "agentdojo",
        "ground_truth": 0,
        "rendered_prompt_sha256": hashlib.sha256(teacher_prompt.encode()).hexdigest(),
        "trajectory_sha256": hashlib.sha256(trajectory.encode()).hexdigest(),
    }
    source = {"id": "0", "metadata": metadata, "prompt": teacher_prompt}
    rows = materialize_rows([dict(source, id=str(index)) for index in range(6_395)])
    assert rows[0]["prompt"] == prompts.student.render(trajectory)
    assert rows[0]["metadata"]["prompt_role"] == "student"
    assert (
        rows[0]["metadata"]["teacher_rendered_prompt_sha256"]
        == metadata["rendered_prompt_sha256"]
    )


def frozen_config(tmp_path: Path) -> dict[str, object]:
    jobs = []
    expected = []
    for rows in (204, 504):
        name = f"soft-n{rows:05d}-seed0"
        expected.append(name)
        jobs.append(
            {
                "job_name": name,
                "target": "kimi_soft",
                "soft_loss_weight": 1.0,
                "direct_loss_weight": 0.0,
                "rank": 128,
                "seed": 0,
            }
        )
    jobs_path = tmp_path / "jobs.jsonl"
    write_jsonl(jobs_path, jobs)
    return {
        "status": "frozen_evaluation",
        "prompt": {
            "role": "student",
            "enable_thinking": False,
            "assistant_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "decision_prefix": "Prediction:",
        },
        "scoring": {"tokens": ["0", "1"]},
        "engine": {"max_lora_rank": 128},
        "model_groups": {
            "9b": {"jobs": str(jobs_path), "expected_jobs": expected},
            "4b": {"jobs": str(jobs_path), "expected_jobs": expected},
        },
    }


def test_distillation_ood_jobs_are_soft_only_and_exact(tmp_path: Path) -> None:
    config = frozen_config(tmp_path)
    validate_config(config)
    assert len(validate_jobs(config, "9b")) == 2
    selected = validate_jobs(config, "9b", "soft-n00204-seed0")
    assert [job["job_name"] for job in selected] == ["soft-n00204-seed0"]
    with pytest.raises(ValueError, match="unknown"):
        validate_jobs(config, "9b", "missing")
    jobs_path = Path(config["model_groups"]["9b"]["jobs"])
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines()]
    jobs[0]["direct_loss_weight"] = 1.0
    write_jsonl(jobs_path, jobs)
    with pytest.raises(ValueError, match="soft-only"):
        validate_jobs(config, "9b")


def test_summary_uses_raw_macro_pauroc_and_hosted_cost(tmp_path: Path) -> None:
    config = {
        "scope": {"rows": 6_395},
        "model_groups": {
            "9b": {
                "id": "Qwen/Qwen3.5-9B",
                "expected_jobs": ["soft-n00204-seed0"],
                "hosted_price_usd_per_million": {"input": 0.1, "output": 0.4},
            }
        },
    }
    result_path = tmp_path / "9b" / "adapters" / "soft-n00204-seed0" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "rows": 6_395,
                "prompt_tokens": {"mean": 4_000.0},
                "metrics": {
                    "macro": {"macro": {"pauroc_at_20": 0.75, "auroc": 0.9}},
                    "pooled": {"pauroc_at_20": 0.7, "auroc": 0.88},
                },
                "training_job": {"train_rows": 204, "seed": 0},
                "training_metadata_sha256": "a",
                "adapter_rebase_source_sha256": "b",
                "adapter_rebase_destination_sha256": "c",
            }
        )
    )
    row = summary_rows(config, tmp_path)[0]
    assert row["mean_ood_pauroc_at_20"] == 0.75
    assert row["uncached_hosted_cost_usd_per_1000"] == pytest.approx(0.4004)


def test_lambda_runner_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/tool_trajectory_monitoring/run_distilled_ood_lambda.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Parity-gate" in completed.stdout


def test_parity_comparison_normalizes_serving_identifiers(tmp_path: Path) -> None:
    eager = tmp_path / "eager.jsonl"
    served = tmp_path / "served.jsonl"
    write_jsonl(
        eager,
        [
            {"dataset": "source", "index": "a", "label": 0, "score": 0.1},
            {"dataset": "source", "index": "b", "label": 1, "score": 0.9},
        ],
    )
    write_jsonl(
        served,
        [
            {"source": "source", "id": "a", "label": 0, "score": 0.11},
            {"source": "source", "id": "b", "label": 1, "score": 0.89},
        ],
    )
    report = compare_predictions(eager, served)
    assert report["rows"] == 2
    assert report["mean_absolute_score_difference"] == pytest.approx(0.01)
    assert report["pearson_score_correlation"] == pytest.approx(1.0)


def test_only_exact_passed_parity_report_is_reused(tmp_path: Path) -> None:
    path = tmp_path / "parity.json"
    assert not parity_report_passed(path, "9b", "job")
    path.write_text(json.dumps({"passed": True, "model_size": "9b", "job_name": "job"}))
    assert parity_report_passed(path, "9b", "job")
    assert not parity_report_passed(path, "4b", "job")
