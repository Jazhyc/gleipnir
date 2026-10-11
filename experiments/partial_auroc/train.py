"""Gradient-cache a full logical-batch tail loss around the existing Trainer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from gleipnir.data.monitoring import write_json
from gleipnir.packed_sequences import packed_partition
from gleipnir.training.adaptive_microbatching import AdaptiveMicrobatchTrainerMixin
from gleipnir.training.partial_auc import partial_auc_loss, replay_row_losses


def gradient_canary() -> dict:
    """Compare exact logical parameter gradients with singleton gradient replay."""
    x = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]], dtype=torch.float64
    )
    labels = torch.tensor([0, 0, 1, 1])
    sources = torch.zeros(4, dtype=torch.long)
    weight = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
    direct = partial_auc_loss(x @ weight, labels, sources)
    expected = torch.autograd.grad(direct, weight)[0]
    leaf = (x @ weight).detach().requires_grad_(True)
    objective = partial_auc_loss(leaf, labels, sources)
    coefficients = torch.autograd.grad(objective, leaf)[0]
    actual = torch.zeros_like(weight)
    for i in range(4):
        replay = (
            replay_row_losses(
                (x[i : i + 1] @ weight), coefficients[i : i + 1], 4, objective
            ).mean()
            / 4
        )
        actual += torch.autograd.grad(replay, weight)[0]
    if not torch.allclose(expected, actual, atol=1e-12, rtol=1e-12):
        raise ValueError("logical ranking gradient replay canary failed")
    return {
        "passed": True,
        "maximum_parameter_gradient_error": float((expected - actual).abs().max()),
    }


def install(module, spec: dict, receipt_path: Path) -> None:
    if spec["weight"] != 0.1 or spec["beta"] != 0.2 or spec["temperature"] != 1:
        raise ValueError("single predeclared partial-AUC setting changed")
    state = {"mode": None, "records": [], "canary": gradient_canary()}
    write_json(
        receipt_path, {"spec": spec, "gradient_canary": state["canary"], "records": []}
    )
    original_loss = module.soft_binary_distillation_losses
    original_step = AdaptiveMicrobatchTrainerMixin.training_step

    def losses(logits, targets, **kwargs):
        ordinary = original_loss(logits, targets, **kwargs)
        margins = logits[:, 1].float() - logits[:, 0].float()
        if state["mode"] == "scout":
            state["scored"] = margins.detach()
        elif state["mode"] == "replay":
            indices = state["indices"]
            reference = state["margins"][indices]
            error = float((margins.detach() - reference).abs().max())
            state["maximum_replay_error"] = max(state["maximum_replay_error"], error)
            if error > 1e-5:
                raise ValueError(f"score/replay margins differ: {error}")
            ordinary = ordinary + spec["weight"] * replay_row_losses(
                margins,
                state["gradients"][indices],
                state["logical_size"],
                state["loss"],
            )
        return ordinary

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self.sequence_packing or self.args.gradient_accumulation_steps != 1:
            raise ValueError(
                "partial-AUC replay requires validated single-update packing"
            )
        model.train()
        features = inputs["logical_batch"].features
        partition = packed_partition(
            [len(f["direct_input_ids"]) for f in features],
            self.microbatch_policy.max_padded_tokens,
        )
        original_collator = self.physical_collator
        started = time.perf_counter()
        scores = None
        try:
            state["mode"] = "scout"
            for indices in partition:
                with torch.no_grad(), self.compute_loss_context_manager():
                    batch = self._prepare_inputs(
                        original_collator([features[i] for i in indices])
                    )
                    self.compute_loss(model, batch)
                observed = state.pop("scored")
                if scores is None:
                    scores = torch.empty(
                        len(features), device=observed.device, dtype=observed.dtype
                    )
                scores[indices] = observed
            scout_seconds = time.perf_counter() - started
            leaf = scores.detach().requires_grad_(True)
            labels = torch.tensor(
                [f["binary_label"] for f in features], device=leaf.device
            )
            sources = torch.tensor(
                [f["dataset_id"] for f in features], device=leaf.device
            )
            objective = partial_auc_loss(leaf, labels, sources, spec["beta"])
            coefficients = torch.autograd.grad(objective, leaf)[0]
            state.update(
                mode="replay",
                margins=scores,
                gradients=coefficients,
                logical_size=len(features),
                loss=objective.detach(),
                maximum_replay_error=0.0,
            )
            sequence = iter(partition)

            def collate(selected):
                state["indices"] = next(sequence)
                return original_collator(selected)

            self.physical_collator = collate
            value = original_step(self, model, inputs, num_items_in_batch)
            if next(sequence, None) is not None:
                raise ValueError("score/replay physical partition changed")
            active = sum(
                bool(((sources == s) & (labels == 0)).any())
                and bool(((sources == s) & (labels == 1)).any())
                for s in torch.unique(sources)
            )
            record = {
                "update": len(self.logical_batch_sizes),
                "rank_loss": float(objective.detach()),
                "active_sources": active,
                "logical_rows": len(features),
                "scout_tokens": sum(len(f["direct_input_ids"]) for f in features),
                "scout_seconds": scout_seconds,
                "maximum_replay_margin_error": state["maximum_replay_error"],
            }
            state["records"].append(record)
            write_json(
                receipt_path,
                {
                    "spec": spec,
                    "gradient_canary": state["canary"],
                    "records": state["records"],
                    "gradient_normalization": "full_logical_batch_gradient_replay",
                },
            )
            print("partial_auc_update", json.dumps(record), flush=True)
            return value
        finally:
            self.physical_collator = original_collator
            state["mode"] = None

    module.soft_binary_distillation_losses = losses
    AdaptiveMicrobatchTrainerMixin.training_step = training_step


if __name__ == "__main__":
    import yaml

    import experiments.deception_distillation.train_student_sft as module

    config = yaml.safe_load(Path(os.environ["GLEIPNIR_PAUC_CONFIG"]).read_text())
    install(module, config["partial_auc"], Path(os.environ["GLEIPNIR_PAUC_RECEIPT"]))
    module.main()
