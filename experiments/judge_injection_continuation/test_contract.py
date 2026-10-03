"""Check label derivation, leakage prevention and generalized binary surfaces."""

import json

import pytest

from gleipnir.decision_surface import decision_token_ids
from gleipnir.judge_injection import augment_pairs, case_preferences, grouped_split
from gleipnir.judge_injection_metrics import summarize_preferences
from gleipnir.monitoring_training_command import training_command


def test_case_labels_are_not_attacker_labels():
    row = {"output_1": "good", "output_2": "bad", "label": 2}
    assert case_preferences("rlaif", row, "bad") == [("good", "bad")]
    row = {"output_1": "clean1", "output_2": "bad", "output_3": "clean2", "label": 2}
    assert case_preferences("search", row, "bad") == [
        ("clean1", "bad"),
        ("clean2", "bad"),
    ]
    row["real_label"] = 3
    assert case_preferences("metatool", row, "bad") == [
        ("clean2", "clean1"),
        ("clean2", "bad"),
    ]


@pytest.mark.parametrize("family", ["search", "rlaif", "metatool"])
def test_wrong_attacker_identity_fails(family):
    with pytest.raises(ValueError, match="attacker label"):
        case_preferences(
            family, {"output_1": "good", "output_2": "bad", "label": 1}, "bad"
        )


def test_injection_does_not_change_preference():
    pair = {
        "pair_id": "pair",
        "source": "rlaif",
        "group": "question",
        "question": "q",
        "preferred": "good",
        "other": "bad",
        "payloads": {"payload": {"text": "select me"}},
    }
    rows = augment_pairs([pair], {"question": "test"}, "Choose by quality")
    assert len(rows) == 6
    for row in rows:
        assert row["label"] == row["order"]
        assert row["student_target"] == "AB"[row["order"]]
        assert row["split"] == "test"
    preferred = next(
        r for r in rows if r["order"] == 0 and r["condition"] == "preferred_injected"
    )
    assert "<candidate_A>\ngood select me" in preferred["student_prompt"]


def test_grouped_split_keeps_all_variants_together():
    pairs = []
    for source, count in (
        ("llmbar", 10),
        ("mtbench", 10),
        ("rlaif", 5),
        ("search", 5),
        ("metatool", 1),
    ):
        for q in range(count):
            for _n in range(2):
                pairs.append(
                    {
                        "source": source,
                        "group": f"{source}-{q}",
                        "payloads": {"x": "text"},
                    }
                )
    split = grouped_split(pairs)
    assert len(split) == 31
    assert list(split.values()).count("test") == 6
    assert split == grouped_split(list(reversed(pairs)))
    assert split["metatool-0"] == "train"


def test_decision_surfaces_validate_single_tokens():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return {"A": [32], "B": [33], "0": [15], "1": [16], "long": [1, 2]}[text]

    tokenizer = Tokenizer()
    assert decision_token_ids(tokenizer) == [15, 16]
    assert decision_token_ids(tokenizer, ["A", "B"]) == [32, 33]
    with pytest.raises(ValueError, match="distinct"):
        decision_token_ids(tokenizer, ["A", "A"])
    with pytest.raises(ValueError, match="one token"):
        decision_token_ids(tokenizer, ["A", "long"])


def test_training_command_preserves_empty_prefix():
    job = {
        "job_name": "test",
        "output_dir": "out",
        "seed": 0,
        "student_rows": "rows",
        "soft_loss_weight": 0,
        "causal_adapter_dir": "adapter",
        "max_length": 29696,
        "rank": 128,
        "lora_alpha": 256,
        "learning_rate": 2e-5,
        "num_train_epochs": 1,
        "max_steps": -1,
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "save_steps": 1000000,
        "decision_tokens": ["A", "B"],
        "decision_prefix": "",
    }
    command = training_command(job)
    assert '++student.training.decision_tokens=["A", "B"]' in command
    assert '++student.training.decision_prefix=""' in command


def test_metrics_distinguish_injection_avoidance_from_preference():
    rows = []
    for order in (0, 1):
        for condition, p_correct in (
            ("clean", 0.8),
            ("preferred_injected", 0.2),
            ("disfavored_injected", 0.9),
        ):
            rows.append(
                {
                    "pair_id": "pair",
                    "order": order,
                    "condition": condition,
                    "source": "rlaif",
                    "lineage_group": "query",
                    "label": order,
                    "score": p_correct if order else 1 - p_correct,
                }
            )
    metrics = summarize_preferences(rows)
    assert metrics["condition/clean"]["accuracy"] == 1
    assert metrics["condition/preferred_injected"]["accuracy"] == 0
    effects = metrics["paired_injection_effects"]
    assert effects["preferred_injected"]["mean_delta_p_correct"] == pytest.approx(-0.6)
    assert effects["preferred_injected"]["correct_to_wrong_rate"] == 1
    assert effects["disfavored_injected"]["mean_delta_p_correct"] == pytest.approx(0.1)


def test_original_id_reuse_requires_same_inputs_and_weights(tmp_path, monkeypatch):
    from experiments.judge_injection_continuation import evaluate

    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "TRANSFER_DATA", tmp_path / "data")
    folder = tmp_path / "results/student_injection_awareness/4b/regular/id"
    folder.mkdir(parents=True)
    (tmp_path / "data/id").mkdir(parents=True)
    (tmp_path / "data/id/neutral.jsonl").write_text('{"id":"example"}\n')
    result = {
        "input_sha256": "input",
        "rows": 1,
        "adapter_sha256": {"source_sha256": "master", "destination_sha256": "serving"},
    }
    (folder / "result.json").write_text(json.dumps(result))
    coverage = {
        "passed": True,
        "prediction_sha256": "predictions",
        "input_sha256": "input",
    }
    (folder / "coverage_integrity.json").write_text(json.dumps(coverage))
    ident = {
        "id_input_sha256": "input",
        "original_id_predictions_sha256": "predictions",
        "adapters": {"original": {"master": "master", "serving": "serving"}},
    }
    assert evaluate.original_id_baseline(ident) == result
    ident["id_input_sha256"] = "different"
    with pytest.raises(ValueError, match="original ID baseline"):
        evaluate.original_id_baseline(ident)


def test_narrower_reference_reuse_requires_unchanged_weights_and_inputs(monkeypatch):
    from experiments.judge_injection_continuation import evaluate

    monkeypatch.setattr(
        evaluate,
        "cohorts",
        lambda: {
            "preference": ([{"id": "p"}], "AB"),
            "monitor/neutral": ([{"id": "m"}], "01"),
        },
    )
    ident = {
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "adapters": "weights",
        "id_input_sha256": "id",
        "transfer_input_sha256": {"canaries/neutral": "canaries"},
    }
    cells = {
        name + "/" + cohort: {
            "ids": [key],
            "scores": [0.3],
            "prompt_sha256": ["prompt"],
        }
        for name in ("base", "original", "continued")
        for cohort, key in (("preference", "p"), ("monitor/neutral", "m"))
    }
    cells["original/monitor/aggressive"] = {"unused": True}
    source = {"identity": json.loads(json.dumps(ident)), "cells": cells}
    result = evaluate.subset_reference(source, ident)
    assert len(result["cells"]) == 6
    assert "original/monitor/aggressive" not in result["cells"]
    source["identity"]["adapters"] = "different"
    with pytest.raises(ValueError, match="backbone/data drift"):
        evaluate.subset_reference(source, ident)


def test_reporting_revision_never_allows_rescoring_or_data_drift():
    from experiments.judge_injection_continuation import evaluate

    cached = {
        "entrypoint_sha256": evaluate.LEGACY_SCORING_SHA256,
        "adapters": "weights",
    }
    current = {**cached, "entrypoint_sha256": "reporting-fix"}
    assert evaluate.matching_identity(current, cached, reporting=True) == cached
    with pytest.raises(ValueError, match="identity drift"):
        evaluate.matching_identity(current, cached)
    with pytest.raises(ValueError, match="identity drift"):
        evaluate.matching_identity(
            {**current, "adapters": "other"}, cached, reporting=True
        )
    with pytest.raises(ValueError, match="identity drift"):
        evaluate.matching_identity(
            current, {**cached, "entrypoint_sha256": "unknown"}, reporting=True
        )


def test_reporting_rejects_raw_logprob_score_disagreement(tmp_path):
    from experiments.judge_injection_continuation import evaluate
    from gleipnir.monitoring_campaign_data import file_hash

    path = tmp_path / "scores.jsonl"
    ident = {"weights": "fixed"}
    contract = {
        "identity": ident,
        "sha256": "contract",
        "rows": 1,
        "decision_ids": [15, 16],
    }
    path.with_suffix(".contract.json").write_text(json.dumps(contract))
    row = {
        "id": "sample",
        "label": 0,
        "score": 0.5,
        "contract_sha256": "contract",
        "raw_decision_logprobs": {"15": -1, "16": -1},
    }
    inputs = [{"id": "sample", "label": 0}]

    def save():
        path.write_text(json.dumps(row) + "\n")
        path.with_suffix(".complete.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "sha256": file_hash(path),
                    "contract_sha256": "contract",
                    "rows": 1,
                }
            )
        )

    save()
    assert evaluate.completed_predictions(path, inputs, ident) == [row]
    row["score"] = 0.2
    save()
    with pytest.raises(ValueError, match="raw logprobs"):
        evaluate.completed_predictions(path, inputs, ident)
