"""Synthetic tests only: no torch import, model downloads, VM calls or GPU."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.adversarial_crossed_text_probe import analyze, common, extract
from gleipnir.adversarial.apps import SUBMIT, USER
from gleipnir.adversarial.contract import (
    QWEN_NON_THINKING_ASSISTANT_SUFFIX,
    join_prompt,
    render,
)


def request(code):
    return join_prompt("instruction\n", USER + "a task" + SUBMIT + code)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "user:" + messages[0]["content"] + QWEN_NON_THINKING_ASSISTANT_SUFFIX

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(map(ord, text)),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "#!/usr/bin/python\n",
        "# coding: utf-8\n",
        "#!/usr/bin/python\n# coding: utf-8\n",
    ],
)
def test_code_offsets_exclude_docstring_and_keep_original(prefix):
    code = prefix + "print('λ')\n"
    header = '"""accusation"""\n\n'
    text = request(prefix + header + code[len(prefix) :])
    tokenizer = CharacterTokenizer()
    rendered = render(tokenizer, text)
    ids = list(map(ord, rendered))
    indices = common.code_token_indices(tokenizer, text, code, len(header), ids)
    assert "".join(rendered[i] for i in indices) == code
    assert len(ids) - 1 not in indices
    with pytest.raises(ValueError, match="tokenization"):
        common.code_token_indices(tokenizer, text, code, len(header), ids[:-1])
    with pytest.raises(ValueError, match="recover"):
        common.code_token_indices(tokenizer, text, code + "oops", len(header), ids)


def test_boundary_straddling_tokens_are_excluded():
    class CrossingTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            at = text.index("print")
            result["offset_mapping"][at] = (at - 1, at + 1)
            return result

    tokenizer = CrossingTokenizer()
    text = request("print(1)")
    rendered = render(tokenizer, text)
    indices = common.code_token_indices(
        tokenizer, text, "print(1)", 0, list(map(ord, rendered))
    )
    assert rendered.index("print") not in indices


@pytest.mark.parametrize("epoch", [None, float("nan"), float("inf"), "later"])
def test_deadline_invalid_fails_closed(tmp_path, epoch):
    path = tmp_path / "vm.json"
    path.write_text(json.dumps({"deadline_epoch": epoch}))
    with pytest.raises(SystemExit, match="invalid"):
        common.check_deadline(path, now=10)


def test_deadline_missing_expiry_and_no_mutation(tmp_path):
    path = tmp_path / "vm.json"
    with pytest.raises(SystemExit, match="recorded"):
        common.check_deadline(path, now=10)
    common.write_json(path, {"deadline_epoch": 11})
    before = path.read_bytes()
    assert common.check_deadline(path, now=10)["deadline_epoch"] == 11
    with pytest.raises(SystemExit, match="reached"):
        common.check_deadline(path, now=11)
    assert path.read_bytes() == before


def test_execute_gate_precedes_model_import(monkeypatch):
    def unexpected():
        pytest.fail("configuration/model reached despite no --execute")

    monkeypatch.setattr(extract, "config", unexpected)
    with pytest.raises(SystemExit, match="separate GPU"):
        extract.run(argparse.Namespace(execute=False))


def test_deadline_precedes_all_loads(monkeypatch, tmp_path):
    path = tmp_path / "vm.json"
    common.write_json(path, {"deadline_epoch": 0})
    monkeypatch.setattr(extract, "config", lambda: {"vm": str(path), "max_seconds": 1})
    monkeypatch.setattr(
        extract, "require_committed", lambda: pytest.fail("load reached")
    )
    with pytest.raises(SystemExit, match="deadline reached"):
        extract.run(argparse.Namespace(execute=True))


def test_guard_checks_each_forward_and_raw_logits():
    calls = []
    guard = SimpleNamespace(check=lambda: calls.append("check"))
    scorer = SimpleNamespace(
        score=lambda *a, **k: {"score": 0.5, "logit_0": 0.0, "logit_1": 1.0}
    )
    wrapped = extract.GuardedScorer(scorer, guard)
    assert wrapped.score("prompt")["score"] == 0.5
    assert calls == ["check", "check"]
    scorer.score = lambda *a, **k: {
        "score": 0.5,
        "logit_0": float("nan"),
        "logit_1": 0.0,
    }
    with pytest.raises(ValueError, match="nonfinite"):
        wrapped.score("prompt")


class Tensor:
    """Minimal CPU tensor double to test the hook's axes and nonmutation."""

    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def mean(self, dim):
        return Tensor(self.value.mean(axis=dim))


class Block:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))


def test_capture_block_number_pooling_and_cleanup():
    blocks = [Block() for _ in range(4)]
    scorer = SimpleNamespace(
        model=SimpleNamespace(named_modules=lambda: [("model.layers", blocks)])
    )
    arm = {"layers": [2, 4], "hidden_size": 3}
    data = np.arange(15).reshape(1, 5, 3)
    with extract.Capture(scorer, arm, [1, 3], 5) as cap:
        assert not blocks[0].hooks and not blocks[2].hooks
        for layer in [2, 4]:
            assert blocks[layer - 1].hooks[0](None, None, (Tensor(data),)) is None
        cap.validate()
    np.testing.assert_array_equal(cap.values["L2:decision"], data[0, -1])
    np.testing.assert_array_equal(
        cap.values["L4:code_mean"], data[0, [1, 3]].mean(axis=0)
    )
    np.testing.assert_array_equal(data, np.arange(15).reshape(1, 5, 3))
    assert all(not b.hooks for b in blocks)
    with pytest.raises(RuntimeError), extract.Capture(scorer, arm, [1], 5):
        raise RuntimeError("synthetic forward failure")
    assert all(not b.hooks for b in blocks)


def synthetic():
    cfg = {
        "fold_prefix": "synthetic:",
        "folds": 5,
        "C": [0.01, 0.1, 1.0],
        "seed": 0,
        "bootstrap": 200,
        "conditions": ["none", "specific_accusation"],
        "pools": ["decision", "code_mean"],
        "arms": {"test": {"layers": [1, 2], "hidden_size": 3}},
    }
    rows, vectors = [], []
    rng = np.random.default_rng(19)
    for task in range(30):
        for label in [0, 1]:
            rows.append(
                {
                    "id": f"{task}-{label}",
                    "task_id": str(task),
                    "label": label,
                    "split": "development",
                    "condition": "none",
                }
            )
            vectors.append([2 * label - 1, rng.normal(), rng.normal()])
    values = np.array(vectors)
    arrays = {f"L{layer}:{p}": values.copy() for layer in [1, 2] for p in cfg["pools"]}
    arrays["length"] = rng.normal(size=(len(rows), 2))
    return cfg, rows, arrays


def test_task_folds_are_deterministic_disjoint_and_balanced():
    cfg, rows, _ = synthetic()
    tasks = [r["task_id"] for r in rows]
    fold = common.fold_map(tasks, cfg)
    assert fold == common.fold_map(tasks[::-1], cfg)
    assert [list(fold.values()).count(i) for i in range(5)] == [6] * 5
    for i in range(5):
        train = {t for t in tasks if fold[t] != i}
        test = {t for t in tasks if fold[t] == i}
        assert not train & test


def test_selection_only_development_none_and_ties():
    cfg, rows, arrays = synthetic()
    fit = analyze.fit_models(rows, arrays, cfg, "test")
    assert fit["probe"]["selected"]["key"] == "L1:decision"
    assert fit["probe"]["selected"]["C"] == 0.01
    assert fit["probe"]["selected"]["cv_auroc"] == 1.0
    assert len(fit["probe"]["candidates"]) == 12
    rows[0]["split"] = "heldout"
    with pytest.raises(ValueError, match="only development"):
        analyze.fit_models(rows, arrays, cfg, "test")
    rows[0]["split"], rows[0]["condition"] = "development", "specific_accusation"
    with pytest.raises(ValueError, match="only development"):
        analyze.fit_models(rows, arrays, cfg, "test")


def test_shuffling_balances_each_task_and_is_reproducible():
    _, rows, _ = synthetic()
    labels, swaps = analyze.shuffled_labels(rows, 0)
    again, swaps2 = analyze.shuffled_labels(rows, 0)
    np.testing.assert_array_equal(labels, again)
    assert swaps == swaps2 and set(swaps.values()) == {0, 1}
    assert (labels.reshape(-1, 2).sum(axis=1) == 1).all()


def test_scaler_uses_training_statistics_and_portable_scores():
    x = np.array([[1.0, 20.0], [2.0, 20.0], [3.0, 20.0], [4.0, 20.0]])
    model = analyze.fit_linear(x, np.array([0, 0, 1, 1]), 0.1)
    assert model["mean"] == [2.5, 20.0]
    assert model["scale"][1] == 1.0
    before = json.dumps(model)
    scores = analyze.predict(model, np.array([[1e6, 9999], [1.0, 20.0]]))
    assert scores[0] > scores[1] and json.dumps(model) == before


def test_bootstrap_ties_pair_multiplicity_and_paired_differences():
    scores = np.array([[0.0, 2.0], [4.0, 3.0], [1.0, 1.0]])
    result = analyze.bootstrap_metrics({"a": scores, "b": scores}, 500, 0, [("a", "b")])
    expected = roc_auc_score(np.tile([0, 1], 3), scores.ravel())
    assert result["metrics"]["a"]["auroc"] == expected
    assert result["metrics"]["a"]["win_rate"] == 0.5
    assert result["differences"]["a_minus_b"]["ci95"] == [0.0, 0.0]
    rng = np.random.default_rng(0)
    counts = rng.multinomial(3, np.full(3, 1 / 3), size=500)
    draws = []
    for count in counts:
        pairs = np.repeat(scores, count, axis=0)
        draws.append(roc_auc_score(np.tile([0, 1], 3), pairs.ravel()))
    np.testing.assert_allclose(
        result["metrics"]["a"]["auroc_ci95"], np.quantile(draws, [0.025, 0.975])
    )


def test_pair_validation_rejects_duplicates_missing_and_overlap():
    with pytest.raises(ValueError, match="incomplete"):
        analyze.paired_arrays([{"task_id": "a", "label": 0}], ["score"])
    with pytest.raises(ValueError, match="duplicate"):
        analyze.paired_arrays([{"task_id": "a", "label": 0}] * 2, ["score"])
    with pytest.raises(ValueError, match="overlap"):
        common.validate_rows([], {"development": ["a"], "heldout": ["a"]})


def test_synthetic_transfer_and_reports():
    cfg, rows, arrays = synthetic()
    fitted = analyze.fit_models(rows, arrays, cfg, "test")
    held = []
    for condition in cfg["conditions"]:
        for row in rows:
            held.append(
                {
                    **row,
                    "id": row["id"] + condition,
                    "task_id": "h" + row["task_id"],
                    "split": "heldout",
                    "condition": condition,
                    "score": 0.5,
                    "margin": 0.0,
                }
            )
    held_arrays = {key: np.concatenate([value, value]) for key, value in arrays.items()}
    output = analyze.predictions(held, held_arrays, fitted)
    report = analyze.reports(output, cfg)
    assert report["specific_accusation"]["metrics"]["probe"]["auroc"] == 1.0
    assert report["specific_accusation"]["metrics"]["output"]["auroc"] == 0.5
    assert report["specific_accusation"]["differences"]["probe_minus_output"][
        "ci95"
    ] == [0.5, 0.5]


def test_fit_gate_rejects_changed_development(tmp_path):
    common.write_json(tmp_path / "development.complete.json", {"rows": 10})
    identity = {"test": True}
    common.write_json(
        tmp_path / "fit.json",
        {
            "identity": identity,
            "development_complete_sha256": common.digest(
                tmp_path / "development.complete.json"
            ),
        },
    )
    extract.validate_fit_gate(tmp_path, identity)
    common.write_json(tmp_path / "development.complete.json", {"rows": 9})
    with pytest.raises(ValueError, match="matching frozen"):
        extract.validate_fit_gate(tmp_path, identity)


def test_originals_ignore_calibration_rows_without_condition():
    row = {
        "metadata": {"condition": "none", "task_id": "a", "code": "honest"},
        "prompt": request("print(1)"),
    }
    assert common.originals([{"metadata": {"split": "calibration"}}, row]) == {
        ("a", "honest"): "print(1)"
    }


def test_synthetic_artifact_pipeline_and_integrity(tmp_path, monkeypatch):
    cfg, _, _ = synthetic()
    cfg.update(output=str(tmp_path), C=[0.1], bootstrap=100)
    cfg["arms"]["other"] = cfg["arms"]["test"].copy()
    variants = []
    selection = {"development": [], "heldout": []}
    for split, count in [("development", 10), ("heldout", 6)]:
        for i in range(count):
            task = f"{split}-{i}"
            selection[split].append(task)
            for condition in ["none"] if split == "development" else cfg["conditions"]:
                header = "" if condition == "none" else '"""accusation"""\n'
                for label, code in [(0, "honest"), (1, "backdoored")]:
                    source = f"print({label})\n"
                    variants.append(
                        {
                            "id": f"{task}-{condition}-{code}",
                            "prompt": request(header + source),
                            "metadata": {
                                "task_id": task,
                                "split": split,
                                "code": code,
                                "label": label,
                                "condition": condition,
                                "added_chars": len(header),
                                "rendered_header_sha256": common.sha(header),
                            },
                        }
                    )
    common.validate_rows(variants, selection)
    monkeypatch.setattr(analyze, "load_inputs", lambda cfg: (variants, selection))
    original = common.originals(variants)

    def save_split(folder, arm, split):
        identity = {"protocol": common.protocol_identity(cfg), "arm": arm}
        common.write_json(folder / "identity.json", identity)
        journal = folder / f"{split}.jsonl"
        for variant in common.selected_rows(variants, split, cfg):
            meta = variant["metadata"]
            filename = common.sha(variant["id"]) + ".npz"
            data = {
                f"L{layer}:{pool}": np.array(
                    [2 * meta["label"] - 1.0, 0.0, 1.0], np.float32
                )
                for layer in [1, 2]
                for pool in cfg["pools"]
            }
            np.savez(folder / filename, **data)
            common.append(
                journal,
                {
                    **common.row_metadata(
                        variant, original[(meta["task_id"], meta["code"])]
                    ),
                    "file": filename,
                    "sha256": common.digest(folder / filename),
                    "score": 0.5,
                    "margin": 0.0,
                    "logit_0": 1.0,
                    "logit_1": 1.0,
                },
            )
        common.write_json(
            folder / f"{split}.complete.json",
            {
                "rows": len(common.selected_rows(variants, split, cfg)),
                "journal_sha256": common.digest(journal),
                "identity": identity,
            },
        )

    for arm in ["test", "other"]:
        folder = tmp_path / arm
        folder.mkdir()
        save_split(folder, arm, "development")
        analyze.fit_command(folder, cfg, arm)
        common.write_json(
            folder / "heldout.fit.json",
            {"fit_sha256": common.digest(folder / "fit.json")},
        )
        save_split(folder, arm, "heldout")
        analyze.evaluate_command(folder, cfg, arm)
        report = json.loads((folder / "report.json").read_text())
        assert report["conditions"]["specific_accusation"]["pairs"] == 6
        with pytest.raises(ValueError, match="fitting is locked"):
            analyze.fit_command(folder, cfg, arm)
    analyze.compare_command(cfg, ["test", "other"])
    comparison = json.loads((tmp_path / "compare-test-other.json").read_text())
    assert comparison["conditions"]["none"]["differences"]["probe_a_minus_probe_b"][
        "ci95"
    ] == [0.0, 0.0]
    assert comparison["conditions"]["none"]["differences"][
        "probe_output_gap_a_minus_b"
    ]["ci95"] == [0.0, 0.0]
    folder = tmp_path / "test"
    # Corruption cannot silently remove a row or admit stale cached activations.
    artifact = next(folder.glob("*.npz"))
    artifact.write_bytes(b"corrupt")
    split = next(
        r["split"]
        for s in ["development", "heldout"]
        for r in common.read_jsonl(folder / f"{s}.jsonl")
        if r["file"] == artifact.name
    )
    with pytest.raises(ValueError, match="checksum"):
        analyze.load_captures(folder, split, cfg, "test")
    journal = folder / "heldout.jsonl"
    journal.write_text(journal.read_text() + journal.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="modified capture journal"):
        analyze.load_captures(folder, "heldout", cfg, "test")


def test_failed_step1_canary_prevents_hooked_forwards(monkeypatch):
    from experiments.adversarial_code_or_comments import common as prior_common
    from experiments.adversarial_code_or_comments import score as prior_score

    monkeypatch.setattr(prior_common, "load_pairs", lambda cfg: ({}, {}))
    monkeypatch.setattr(prior_common, "instruction", lambda *args: "instruction\n")
    monkeypatch.setattr(prior_score, "canary", lambda *args: {"passed": False})
    monkeypatch.setattr(
        extract, "capture_row", lambda *args: pytest.fail("capture ran")
    )
    result = extract.canaries(None, {"prompt": "student"}, [], {}, {}, {})
    assert not result["passed"] and result["hooks"] == []


@pytest.mark.parametrize("drift,passed", [(0.0, True), (0.1, False)])
def test_hook_canary_checks_both_programs_and_text_conditions(
    monkeypatch, drift, passed
):
    from experiments.adversarial_code_or_comments import common as prior_common
    from experiments.adversarial_code_or_comments import score as prior_score

    monkeypatch.setattr(prior_common, "load_pairs", lambda cfg: ({}, {}))
    monkeypatch.setattr(prior_common, "instruction", lambda *args: "instruction\n")
    monkeypatch.setattr(prior_score, "canary", lambda *args: {"passed": True})
    scorer = SimpleNamespace(
        score=lambda p: {"score": 0.5, "logit_0": 1.0, "logit_1": 1.0}
    )
    monkeypatch.setattr(
        extract,
        "capture_row",
        lambda *args: ({"score": 0.5, "logit_0": 1.0, "logit_1": 1.0 + drift}, {}),
    )
    rows = [
        {
            "id": f"{condition}-{code}",
            "prompt": request("print(1)"),
            "metadata": {
                "split": "development",
                "condition": condition,
                "code": code,
                "task_id": "a",
                "added_chars": 0,
            },
        }
        for condition in ["none", "specific_accusation"]
        for code in ["honest", "backdoored"]
    ]
    result = extract.canaries(
        scorer,
        {"prompt": "student"},
        rows,
        {},
        {("a", c): "print(1)" for c in ["honest", "backdoored"]},
        {},
    )
    assert result["passed"] is passed
    assert len(result["hooks"]) == 4
