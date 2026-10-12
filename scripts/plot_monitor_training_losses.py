"""Audit and plot saved, matched 4B/9B regular online distillation losses."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleipnir-matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np

from gleipnir.analysis.plotting import set_plot_style
from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Verify the shared target population and compare recorded training curves."""
    data = ROOT / "data/student_injection_awareness"
    manifest = json.loads((data / "manifest.json").read_text())
    teacher = data / "soft_targets.jsonl"
    assert file_hash(teacher) == manifest["files_sha256"][teacher.name]
    targets = [json.loads(line)["soft_target"] for line in teacher.open()]
    assert len(targets) == manifest["training_rows"] == 8688
    assert all(0 <= q <= 1 for q in targets)
    entropy = float(
        np.mean([-sum(p * math.log(p) for p in (q, 1 - q) if p > 0) for q in targets])
    )
    report = {
        "population": "unfiltered regular tool trajectories",
        "training_rows": 8688,
        "teacher_entropy_nats": entropy,
        "files_sha256": {str(teacher.relative_to(ROOT)): file_hash(teacher)},
        "qualification": (
            "Losses are online, before updates, across changing model states. "
            "They are not final-checkpoint training-set scores. Epoch KL is "
            "approximate: Trainer averages logical updates, including the "
            "short last batch; teacher entropy is row averaged. Late-window "
            "KL is not inferred from whole-population entropy."
        ),
        "models": {},
    }
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    shared = None
    for size, color in [("4b", "#4C78A8"), ("9b", "#E45756")]:
        path = (
            ROOT
            / f"results/student_injection_awareness/{size}/regular"
            / "causal_adapter/training_metadata.json"
        )
        meta = json.loads(path.read_text())
        settings = {
            key: meta["optimization"][key]
            for key in (
                "learning_rate",
                "optimizer",
                "lr_scheduler_type",
                "warmup_steps",
                "weight_decay",
                "lora_dropout",
                "dataset_sampling",
                "dataset_loss_weighting",
                "soft_target_logit_scale",
            )
        }
        assert meta["losses"]["soft_type"] == "bce"
        assert meta["losses"]["soft_weight"] == 1
        assert all(
            meta["losses"][key] == 0
            for key in (
                "direct_weight",
                "completion_weight",
                "pairwise_weight",
                "mil_weight",
                "prefix_weight",
            )
        )
        assert (
            meta["losses"]["accumulation_policy"]
            == "sum_per_example_over_logical_batch_v1"
        )
        assert not meta["quantization"]["enabled"]
        assert meta["training_state"]["global_step"] == 272
        assert meta["training_batch"]["logical_batch_size"] == 32
        assert meta["seed"] == 0
        assert (
            sum(r["examples"] for r in meta["adaptive_microbatching"]["records"])
            == 8688
        )
        if shared is None:
            shared = settings
        else:
            assert settings == shared
        logs = [r for r in meta["training_state"]["log_history"] if "loss" in r]
        assert [r["step"] for r in logs] == list(range(5, 271, 5))
        losses = np.asarray([r["loss"] for r in logs])
        assert np.isfinite(losses).all()
        steps = np.asarray([r["step"] for r in logs])
        mean = meta["train_metrics"]["train_loss"]
        report["files_sha256"][str(path.relative_to(ROOT))] = file_hash(path)
        report["models"][size] = {
            "epoch_mean_online_ce": mean,
            "approximate_epoch_mean_online_kl": mean - entropy,
            "last_logged_five_update_ce": float(losses[-1]),
            "last_50_logged_updates_mean_ce": float(losses[-10:].mean()),
            "logged_windows": logs,
            "shared_optimization": settings,
            "checkpointing": meta["gradient_checkpointing"],
            "trainable_parameters": meta["trainable_parameters"],
        }
        ax.plot(steps, losses, color=color, alpha=0.22, linewidth=1)
        ax.plot(
            steps[9:],
            np.convolve(losses, np.ones(10) / 10, mode="valid"),
            color=color,
            linewidth=2.5,
            label=f"{size.upper()} · trailing 50 updates",
        )
    ax.axhline(
        entropy,
        color="#666666",
        linestyle="--",
        linewidth=1.5,
        label="Whole-population teacher entropy",
    )
    ax.set(
        xlabel="Optimizer update",
        ylabel="Online soft cross-entropy (nats)",
        title="Matched 4B / 9B training · 8,688 rows · one epoch",
        ylim=(0.13, 0.75),
    )
    ax.legend(fontsize=11)
    fig.subplots_adjust(bottom=0.27)
    fig.text(
        0.5,
        0.02,
        "Faint curves: five-update reports. "
        "Final two updates are not separately logged.\n"
        "Online losses are not final-checkpoint fit; entropy varies between batches.",
        ha="center",
        fontsize=10,
    )
    stem = ROOT / "figures/monitor_training_losses"
    exports = {}
    for suffix in ("png", "svg", "pdf"):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
        exports[str(path.relative_to(ROOT))] = file_hash(path)
    plt.close(fig)
    report["exports_sha256"] = exports
    report["files_sha256"][str(Path(__file__).relative_to(ROOT))] = file_hash(
        Path(__file__)
    )
    write_json(stem.with_suffix(".provenance.json"), report)
    write_json(ROOT / "results/training_loss_capacity_audit/report.json", report)
    print(
        json.dumps(
            {
                k: {x: v for x, v in d.items() if x.endswith("ce") or x.endswith("kl")}
                for k, d in report["models"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
