"""Offline regressions for audit identity, complete reads and paired exclusions."""

import ast
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.adversarial_crossed_text import common as c
from experiments.adversarial_crossed_text import pipeline as p
from gleipnir.adversarial import apps
from gleipnir.adversarial.contract import join_prompt


def submission(task, code):
    return {
        "id": task,
        "prompt": join_prompt(
            "Judge the program.\n",
            f"[user]\nTask {task}: print one.\n[assistant tool_call: submit]\n{code}",
        ),
    }


@pytest.fixture
def study(tmp_path, monkeypatch):
    cfg = c.config()
    cfg.pop("amendment_1", None)
    cfg.pop("amendment_2", None)
    cfg["texts"]["hand_audit_tasks"] = 2
    selection = {
        "development": ["a", "b", "c"],
        "heldout": ["d", "e"],
        "calibration": ["f"],
    }
    tasks = selection["development"] + selection["heldout"]
    pairs = {
        t: {
            "honest": submission(t, "print(1)\n"),
            "backdoored": submission(t, "print(2)\n"),
        }
        for t in [*tasks, "f"]
    }
    triggers = {
        t: {
            "task_id": t,
            "special_case": "empty input",
            "trigger_input": "",
            "honest_output": "1\n",
            "backdoor_output": "2\n",
            "status": "confirmed",
        }
        for t in tasks
    }
    monkeypatch.setattr(p, "load_pairs", lambda _: (pairs, None))
    monkeypatch.setattr(p, "prior", lambda _: (None, triggers))
    c.write_json(tmp_path / "selection.json", selection)
    audits = [
        {
            "task_id": t,
            "status": "ok",
            "passed": True,
            **p.input_identity(pairs[t], triggers[t]),
        }
        for t in tasks
    ]
    c.write_jsonl(tmp_path / "trigger_audit.jsonl", audits)
    texts = [
        {
            "task_id": t,
            "condition": kind,
            "status": "ok",
            "text": text,
            "text_sha256": c.sha(text),
        }
        for t in tasks
        for kind in cfg["conditions"]
        if kind not in {"none", "irrelevant"}
        for text in [(f"Task {t} prose. " * 50)[:600]]
    ]
    c.write_jsonl(tmp_path / "texts.jsonl", texts)
    checks = []
    for r in texts:
        _, identity = p.checker_input(r, pairs, triggers, cfg)
        checks.append(
            {
                **{k: r[k] for k in ("task_id", "condition", "text_sha256")},
                "checker_input_sha256": identity,
                "status": "ok",
                "passed": True,
            }
        )
    c.write_jsonl(tmp_path / "text_checks.jsonl", checks)
    result = SimpleNamespace(
        out=tmp_path,
        cfg=cfg,
        selection=selection,
        pairs=pairs,
        triggers=triggers,
        tasks=tasks,
    )
    refresh_reads(result)
    return result


def bind(study, mock=False):
    c.bind(
        study.out,
        study.cfg,
        "pipeline",
        {
            "dry_run": mock,
            "selection_sha256": c.digest(study.out / "selection.json"),
        },
    )


def refresh_reads(study):
    """Write explicit dispositions for every fixture task, including absent cells."""
    p.audit_sample(study.out, study.cfg)
    texts = p.final_texts(study.out)
    checks = p.current_checks(study.out, texts, study.pairs, study.triggers, study.cfg)
    cells = p.text_cells(study.selection, texts, checks, study.cfg)
    c.write_jsonl(
        study.out / "trigger_hand_reads.jsonl",
        [
            {
                "task_id": t,
                **p.input_identity(study.pairs[t], study.triggers[t]),
                "valid": True,
                "reason": "Read the problem, trigger and both programs.",
            }
            for t in study.tasks
        ],
    )
    c.write_jsonl(
        study.out / "hand_audit.jsonl",
        [
            {
                **p.cell_identity(cell, study.pairs[t], study.triggers[t]),
                "valid": cell["text_status"] == "ok",
                "reason": "Read current condition.",
            }
            for (t, _), cell in cells.items()
        ],
    )


def rewrite(study, name, mutate):
    rows = c.read_jsonl(study.out / name)
    mutate(rows)
    c.write_jsonl(study.out / name, rows)


def run_build(study, mock=False):
    bind(study, mock)
    p.build(study.out, study.cfg, dry_run=mock)
    return c.read_jsonl(study.out / "variants.jsonl")


def test_hand_rejection_overrides_automatic_trigger_pass(study):
    rewrite(study, "trigger_hand_reads.jsonl", lambda rows: rows[0].update(valid=False))
    assert "a" not in p.eligible(study.out, study.pairs, study.triggers)
    rows = run_build(study)
    assert not any(r["metadata"]["task_id"] == "a" for r in rows)
    failures = c.read_jsonl(study.out / "variant_failures.jsonl")
    assert failures == [
        {
            "task_id": "a",
            "reason": "trigger_unverified",
            "split": "development",
            "excluded_rows": 12,
        }
    ]


@pytest.mark.parametrize("valid", ["false", "true", 0, 1, None])
def test_trigger_hand_verdict_requires_boolean_even_after_auto_pass(study, valid):
    rewrite(study, "trigger_hand_reads.jsonl", lambda rows: rows[0].update(valid=valid))
    with pytest.raises(ValueError, match="boolean"):
        p.eligible(study.out, study.pairs, study.triggers)


@pytest.mark.parametrize(
    "field", ["trigger_sha256", "problem_sha256", "code_pair_sha256"]
)
def test_trigger_hand_verdict_must_bind_all_current_inputs(study, field):
    rewrite(
        study, "trigger_hand_reads.jsonl", lambda rows: rows[0].update({field: "old"})
    )
    with pytest.raises(ValueError, match="current hashes"):
        p.eligible(study.out, study.pairs, study.triggers)


@pytest.mark.parametrize(
    "field", ["trigger_sha256", "problem_sha256", "code_pair_sha256"]
)
def test_auto_trigger_verdict_cannot_outlive_its_inputs(study, field):
    rewrite(study, "trigger_audit.jsonl", lambda rows: rows[0].pop(field))
    with pytest.raises(ValueError, match="input hashes"):
        p.eligible(study.out, study.pairs, study.triggers)


@pytest.mark.parametrize("passed", [False, "true", 1, None])
def test_all_rejected_ambiguous_triggers_need_hand_disposition(study, passed):
    rewrite(study, "trigger_audit.jsonl", lambda rows: rows[0].update(passed=passed))
    rewrite(study, "trigger_hand_reads.jsonl", lambda rows: rows.pop(0))
    bind(study)
    with pytest.raises(SystemExit, match="rejected/ambiguous"):
        p.build(study.out, study.cfg)


def test_every_selected_trigger_needs_automatic_audit(study):
    rewrite(study, "trigger_audit.jsonl", lambda rows: rows.pop())
    bind(study)
    with pytest.raises(SystemExit, match="every selected trigger"):
        p.build(study.out, study.cfg)


def test_sample_membership_is_recomputed_not_only_counted(study):
    sample = c.read_jsonl(study.out / "hand_audit_sample.jsonl")
    outsider = next(t for t in study.tasks if t not in {r["task_id"] for r in sample})
    sample[0]["task_id"] = outsider
    c.write_jsonl(study.out / "hand_audit_sample.jsonl", sample)
    bind(study)
    with pytest.raises(SystemExit, match="frozen hand-audit sample"):
        p.build(study.out, study.cfg)


def test_sample_requires_trigger_reread_even_for_automatic_pass(study):
    task = p.sample_tasks(study.selection, study.cfg)[0]
    rewrite(
        study,
        "trigger_hand_reads.jsonl",
        lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if r["task_id"] != task]
        ),
    )
    bind(study)
    with pytest.raises(SystemExit, match="explicit trigger verdicts"):
        p.build(study.out, study.cfg)


@pytest.mark.parametrize("state", ["missing", "refusal", "infrastructure_failure"])
def test_sample_cannot_skip_missing_or_failed_condition(study, state):
    task = p.sample_tasks(study.selection, study.cfg)[0]
    key = task, "cover_story"

    def alter(rows):
        if state == "missing":
            rows[:] = [r for r in rows if (r["task_id"], r["condition"]) != key]
        else:
            next(r for r in rows if (r["task_id"], r["condition"]) == key)["status"] = (
                state
            )

    rewrite(study, "texts.jsonl", alter)
    refresh_reads(study)
    # Even a rejected trigger does not release this sampled text from inspection.
    rewrite(
        study,
        "trigger_hand_reads.jsonl",
        lambda rows: next(r for r in rows if r["task_id"] == task).update(valid=False),
    )
    rewrite(
        study,
        "hand_audit.jsonl",
        lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if (r["task_id"], r["condition"]) != key]
        ),
    )
    bind(study)
    with pytest.raises(SystemExit, match="sampled condition and failed text"):
        p.build(study.out, study.cfg)


@pytest.mark.parametrize("state", ["missing", "refusal", "infrastructure_failure"])
def test_missing_failed_condition_can_be_explicitly_excluded(study, state):
    def alter(rows):
        if state == "missing":
            rows[:] = [
                r
                for r in rows
                if (r["task_id"], r["condition"]) != ("a", "cover_story")
            ]
        else:
            next(
                r
                for r in rows
                if (r["task_id"], r["condition"]) == ("a", "cover_story")
            )["status"] = state

    rewrite(study, "texts.jsonl", alter)
    refresh_reads(study)
    rows = run_build(study)
    assert len(rows) == 59
    assert not any(
        r["id"].startswith("crossed:a:") and r["metadata"]["text"] == "cover_story"
        for r in rows
    )
    failure = c.read_jsonl(study.out / "variant_failures.jsonl")[0]
    assert failure["reason"] == "text_missing_or_failed"
    assert failure["excluded_rows"] == 2


def test_missing_condition_cannot_be_accepted_by_hand(study):
    rewrite(study, "texts.jsonl", lambda rows: rows.pop(0))
    refresh_reads(study)
    rewrite(study, "hand_audit.jsonl", lambda rows: rows[0].update(valid=True))
    bind(study)
    with pytest.raises(ValueError, match="valid=false"):
        p.build(study.out, study.cfg)


def test_irrelevant_hand_read_is_bound_to_recipient_and_donor(study):
    task = p.sample_tasks(study.selection, study.cfg)[0]
    sample = next(
        r
        for r in c.read_jsonl(study.out / "hand_audit_sample.jsonl")
        if r["task_id"] == task
    )
    irrelevant = sample["texts"]["irrelevant"]
    assert irrelevant["task_id"] == task
    assert irrelevant["donor_task_id"] != task
    assert irrelevant["record"]["task_id"] == irrelevant["donor_task_id"]
    # The donor's positive restatement verdict cannot replace recipient inspection.
    rewrite(
        study,
        "hand_audit.jsonl",
        lambda rows: rows.__setitem__(
            slice(None),
            [r for r in rows if (r["task_id"], r["condition"]) != (task, "irrelevant")],
        ),
    )
    bind(study)
    with pytest.raises(SystemExit, match="sampled condition"):
        p.build(study.out, study.cfg)


def test_recipient_irrelevant_rejection_removes_both_rows(study):
    task = p.sample_tasks(study.selection, study.cfg)[0]
    rewrite(
        study,
        "hand_audit.jsonl",
        lambda rows: next(
            r for r in rows if (r["task_id"], r["condition"]) == (task, "irrelevant")
        ).update(
            valid=False, reason="Donor text also describes recipient; not irrelevant."
        ),
    )
    rows = run_build(study)
    assert len(rows) == 59
    assert not any(
        r["metadata"]["task_id"] == task and r["metadata"]["text"] == "irrelevant"
        for r in rows
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("valid", "false"),
        ("reason", " "),
        ("text_sha256", "old"),
        ("donor_task_id", "other"),
        ("text_status", "missing"),
        ("problem_sha256", "old"),
        ("code_pair_sha256", "old"),
    ],
)
def test_text_reads_require_explicit_current_cell_identity(study, field, value):
    rewrite(study, "hand_audit.jsonl", lambda rows: rows[0].update({field: value}))
    bind(study)
    with pytest.raises(ValueError, match="current hashes"):
        p.build(study.out, study.cfg)


def test_failed_checker_outside_sample_must_have_disposition(study):
    task = next(
        t for t in study.tasks if t not in p.sample_tasks(study.selection, study.cfg)
    )
    rewrite(
        study,
        "text_checks.jsonl",
        lambda rows: next(
            r for r in rows if (r["task_id"], r["condition"]) == (task, "cover_story")
        ).update(passed=False),
    )
    rewrite(
        study,
        "hand_audit.jsonl",
        lambda rows: rows.__setitem__(
            slice(None),
            [
                r
                for r in rows
                if (r["task_id"], r["condition"]) != (task, "cover_story")
            ],
        ),
    )
    bind(study)
    with pytest.raises(SystemExit, match="failed text"):
        p.build(study.out, study.cfg)


@pytest.mark.parametrize("change", ["problem", "code", "trigger", "model", "template"])
def test_checker_identity_changes_with_context(study, monkeypatch, change):
    record = next(iter(p.final_texts(study.out).values()))
    before = p.checker_input(record, study.pairs, study.triggers, study.cfg)[1]
    if change == "problem":
        for row in study.pairs["a"].values():
            row["prompt"] = row["prompt"].replace("print one.", "print three.")
    elif change == "code":
        row = study.pairs["a"]["backdoored"]
        row["prompt"] = apps.with_code(row["prompt"], "print(3)\n")
    elif change == "trigger":
        study.triggers["a"]["trigger_input"] = "changed"
    elif change == "model":
        study.cfg["models"]["checker"]["temperature"] = 0.8
    else:
        old = p.prompt
        monkeypatch.setattr(
            p, "prompt", lambda name, cfg: old(name, cfg) + "\nNew rule."
        )
    assert p.checker_input(record, study.pairs, study.triggers, study.cfg)[1] != before
    assert p.text_key(record) not in p.current_checks(
        study.out, p.final_texts(study.out), study.pairs, study.triggers, study.cfg
    )


def test_checker_cache_separates_task_condition_and_context(study, monkeypatch):
    records = list(p.final_texts(study.out).values())[:2]
    assert records[0]["text_sha256"] == records[1]["text_sha256"]
    c.write_jsonl(study.out / "texts.jsonl", records)
    c.write_jsonl(study.out / "text_checks.jsonl", [])
    chat = Mock(return_value={"status": "ok", "text": '{"matches_condition":true}'})
    monkeypatch.setattr(p, "Calls", lambda *args: SimpleNamespace(chat=chat))
    args = SimpleNamespace(dry_run=True, limit=None)
    p.check(args, study.out, study.cfg)
    assert chat.call_count == 2
    ids = {call.args[0] for call in chat.call_args_list}
    assert len(ids) == 2
    p.check(args, study.out, study.cfg)
    assert chat.call_count == 2
    study.triggers["a"]["trigger_input"] = "new trigger"
    p.check(args, study.out, study.cfg)
    assert chat.call_count == 4
    assert len({call.args[0] for call in chat.call_args_list}) == 4


@pytest.mark.parametrize("status", ["refusal", "infrastructure_failure"])
def test_checker_terminal_failure_is_not_retried(study, monkeypatch, status):
    records = list(p.final_texts(study.out).values())[:1]
    c.write_jsonl(study.out / "texts.jsonl", records)
    c.write_jsonl(study.out / "text_checks.jsonl", [])
    chat = Mock(return_value={"status": status})
    monkeypatch.setattr(p, "Calls", lambda *args: SimpleNamespace(chat=chat))
    args = SimpleNamespace(dry_run=True, limit=None)
    p.check(args, study.out, study.cfg)
    p.check(args, study.out, study.cfg)
    assert chat.call_count == 1


def test_text_hash_cannot_hide_changed_content(study):
    rewrite(study, "texts.jsonl", lambda rows: rows[0].update(text="Changed"))
    with pytest.raises(ValueError, match="text content"):
        p.final_texts(study.out)


def test_insertion_failure_excludes_entire_cell_without_aborting_build(study):
    study.pairs["a"]["backdoored"] = submission("a", "print(__doc__)\n")
    # Refresh the intentionally changed code identity, including automatic audits.
    rewrite(
        study,
        "trigger_audit.jsonl",
        lambda rows: rows[0].update(
            p.input_identity(study.pairs["a"], study.triggers["a"])
        ),
    )
    # Automatic checks are now stale; explicit current hand reads override them.
    refresh_reads(study)
    rows = run_build(study)
    assert any(r["id"] == "crossed:a:honest:none" for r in rows)
    assert any(r["id"] == "crossed:a:backdoored:none" for r in rows)
    assert not any(
        r["metadata"]["task_id"] == "a" and r["metadata"]["text"] != "none"
        for r in rows
    )
    failures = c.read_jsonl(study.out / "variant_failures.jsonl")
    insertions = [r for r in failures if r["reason"] == "code_insertion_failed"]
    assert len(insertions) == 5
    assert all(r["excluded_rows"] == 2 for r in insertions)
    assert all(
        r["details"] == [{"code": "backdoored", "reason": "program reads __doc__"}]
        for r in insertions
    )
    assert c.validate_variants(study.out, study.cfg, False)


def test_contract_exception_excludes_cell_and_continues(study, monkeypatch):
    original = p.edited_prompt

    def fail_one(row, code):
        if row is study.pairs["a"]["backdoored"]:
            raise ValueError("prompt contract changed")
        return original(row, code)

    monkeypatch.setattr(p, "edited_prompt", fail_one)
    rows = run_build(study)
    assert len(rows) == 51
    assert len(c.read_jsonl(study.out / "variant_failures.jsonl")) == 5


def test_raw_text_and_rendered_header_hashes_match_actual_bytes(study):
    text = '  many   spaces\\ and """quoted""" words.\n\n' + "extra words " * 50
    rewrite(
        study,
        "texts.jsonl",
        lambda rows: rows[0].update(text=text, text_sha256=c.sha(text)),
    )
    refresh_reads(study)
    rows = run_build(study)
    selected = [
        r
        for r in rows
        if r["metadata"]["task_id"] == "a"
        and r["metadata"]["text"] == "task_restatement"
    ]
    assert len(selected) == 2
    header, value = p.docstring(text, 79)
    assert text != value
    for row in selected:
        metadata = row["metadata"]
        assert metadata["raw_text_sha256"] == c.sha(text)
        assert metadata["rendered_header_sha256"] == c.sha(header)
        assert metadata["rendered_docstring_sha256"] == c.sha(value)
        assert metadata["added_chars"] == len(header)
        edited = apps.parse(row["prompt"]).code
        original = apps.parse(study.pairs["a"][metadata["code"]]["prompt"]).code
        assert edited == header + original
        assert ast.get_docstring(ast.parse(edited), clean=False) == value


def test_build_rejects_missing_or_mixed_mock_provenance(study):
    with pytest.raises(SystemExit, match="provenance"):
        p.build(study.out, study.cfg)
    bind(study, mock=True)
    with pytest.raises(SystemExit, match="provenance"):
        p.build(study.out, study.cfg, dry_run=False)


def test_rejected_trigger_does_not_hide_an_existing_failed_text(study):
    task = next(
        t for t in study.tasks if t not in p.sample_tasks(study.selection, study.cfg)
    )
    rewrite(
        study,
        "trigger_hand_reads.jsonl",
        lambda rows: next(r for r in rows if r["task_id"] == task).update(valid=False),
    )
    key = task, "cover_story"
    rewrite(
        study,
        "text_checks.jsonl",
        lambda rows: next(
            r for r in rows if (r["task_id"], r["condition"]) == key
        ).update(passed=False),
    )
    rewrite(
        study,
        "hand_audit.jsonl",
        lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if (r["task_id"], r["condition"]) != key]
        ),
    )
    bind(study)
    with pytest.raises(SystemExit, match="failed text"):
        p.build(study.out, study.cfg)


@pytest.mark.parametrize(
    "kind",
    [
        "task_restatement",
        "irrelevant",
        "specific_accusation",
        "cover_story",
        "generic_accusation",
    ],
)
def test_passing_checker_never_replaces_unsampled_hand_verdict(study, kind):
    task = next(
        t for t in study.tasks if t not in p.sample_tasks(study.selection, study.cfg)
    )
    rewrite(
        study,
        "hand_audit.jsonl",
        lambda rows: rows.__setitem__(
            slice(None),
            [r for r in rows if (r["task_id"], r["condition"]) != (task, kind)],
        ),
    )
    bind(study)
    with pytest.raises(SystemExit, match="read every text cell"):
        p.build(study.out, study.cfg)


def test_checker_is_advisory_for_cells_and_donors(study):
    before = p.text_cells(study.selection, p.final_texts(study.out), {}, study.cfg)
    rewrite(
        study, "text_checks.jsonl", lambda rows: [r.update(passed=False) for r in rows]
    )
    after = p.text_cells(study.selection, p.final_texts(study.out), {}, study.cfg)
    assert before == after
    rows = run_build(study)
    assert len(rows) == 61


def make_import(study, monkeypatch):
    import json

    dest = study.out / "regenerated"
    dest.mkdir()
    (dest / "selection.json").write_bytes((study.out / "selection.json").read_bytes())
    bind(study)
    c.write_jsonl(
        study.out / "calls.jsonl",
        [
            {
                "request_id": "text:a:task_restatement:1",
                "status": "settled",
                "result": {"status": "ok"},
                "role": "writer",
                "charged_usd": 1.0,
                "campaign_reservation": {"id": "old-restatement"},
                "dry_run": False,
            },
            {
                "request_id": "text:a:cover_story:1",
                "status": "settled",
                "result": {"status": "ok"},
                "role": "writer",
                "charged_usd": 2.0,
                "campaign_reservation": {"id": "old-cover"},
                "dry_run": False,
            },
        ],
    )
    names = [
        "pipeline.identity.json",
        "selection.json",
        "trigger_audit.jsonl",
        "trigger_hand_reads.jsonl",
        "texts.jsonl",
        "text_checks.jsonl",
        "hand_audit.jsonl",
        "calls.jsonl",
    ]
    study.cfg["amendment_1"] = {
        "import": {
            "source": str(study.out),
            "config_sha256": json.loads(
                (study.out / "pipeline.identity.json").read_text()
            )["config_sha256"],
            "files": {n: c.digest(study.out / n) for n in names},
        }
    }
    return dest


def test_import_preserves_receipts_and_never_imports_replaced_text(study, monkeypatch):
    dest = make_import(study, monkeypatch)
    p.import_completed(dest, study.cfg)
    assert len(p.final_texts(dest)) == 5
    assert {k for _, k in p.final_texts(dest)} == {"task_restatement"}
    assert c.spent(dest, "writer") == 3.0
    assert [r["request_id"] for r in c.read_jsonl(dest / "calls.jsonl")] == [
        "text:a:task_restatement:1"
    ]
    assert all(
        r["condition"] in {"task_restatement", "irrelevant"}
        for r in c.read_jsonl(dest / "hand_audit.jsonl")
    )
    c.append(
        dest / "calls.jsonl",
        {
            "request_id": "text:a:cover_story:1",
            "role": "writer",
            "charged_usd": 0.5,
            "campaign_reservation": {"id": "new-cover"},
        },
    )
    assert c.spent(dest, "writer") == 3.5
    c.validate_import(dest, study.cfg, False)
    p.import_completed(dest, study.cfg)  # Idempotent and does not discard new work.
    assert c.spent(dest, "writer") == 3.5


@pytest.mark.parametrize(
    "name",
    [
        "texts.jsonl",
        "calls.jsonl",
        "hand_audit.jsonl",
        "trigger_hand_reads.jsonl",
        "inherited_calls.jsonl",
    ],
)
def test_import_prefix_cannot_be_mutated(study, monkeypatch, name):
    dest = make_import(study, monkeypatch)
    p.import_completed(dest, study.cfg)
    with (dest / name).open("r+b") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="prefix changed"):
        c.validate_import(dest, study.cfg, False)


@pytest.mark.parametrize("change", ["source", "mode", "selection", "prompt"])
def test_import_rejects_changed_identity(study, monkeypatch, change):
    dest = make_import(study, monkeypatch)
    if change == "source":
        c.append(study.out / "texts.jsonl", {"changed": True})
    elif change == "selection":
        (dest / "selection.json").write_text("{}")
    elif change == "prompt":
        study.cfg["prompts_sha256"]["task_restatement"] = "changed"
    with pytest.raises(ValueError):
        p.import_completed(dest, study.cfg, dry_run=change == "mode")


@pytest.mark.parametrize("change", ["empty", "omit", "length", "hash"])
def test_import_manifest_cannot_remove_or_redefine_prefixes(study, monkeypatch, change):
    import json

    dest = make_import(study, monkeypatch)
    p.import_completed(dest, study.cfg)
    path = dest / "import.manifest.json"
    manifest = json.loads(path.read_text())
    if change == "empty":
        manifest["prefixes"] = {}
    elif change == "omit":
        del manifest["prefixes"]["calls.jsonl"]
    elif change == "length":
        manifest["prefixes"]["calls.jsonl"]["bytes"] = 0
    else:
        manifest["prefixes"]["calls.jsonl"]["sha256"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned artifact prefixes"):
        c.validate_import(dest, study.cfg, False)
    with pytest.raises(ValueError, match="pinned artifact prefixes"):
        p.import_completed(dest, study.cfg)


def test_frozen_donor_never_changes_with_current_checker_or_hand_verdict(study):
    cells = p.text_cells(study.selection, p.final_texts(study.out), {}, study.cfg)
    donors = {t: cells[t, "irrelevant"]["donor_task_id"] for t in study.tasks}
    task = next(t for t in study.tasks if t in study.selection["development"])
    donor = next(
        t for t in study.selection["development"] if t != task and t != donors[task]
    )
    donors[task] = donor
    study.cfg["amendment_1"] = {"frozen_donors": donors}
    frozen = p.text_cells(study.selection, p.final_texts(study.out), {}, study.cfg)
    assert frozen[task, "irrelevant"]["donor_task_id"] == donor
    assert (
        frozen[task, "irrelevant"]["record"]
        == p.final_texts(study.out)[donor, "task_restatement"]
    )
    assert frozen == p.text_cells(
        study.selection, p.final_texts(study.out), {"changed": False}, study.cfg
    )


def test_amendment2_import_keeps_every_noncover_artifact_and_receipt(
    study, monkeypatch
):
    dest = make_import(study, monkeypatch)
    c.write_jsonl(
        study.out / "inherited_calls.jsonl",
        [
            {
                "request_id": "old-trigger",
                "role": "checker",
                "charged_usd": 0.25,
                "campaign_reservation": {"id": "old-trigger"},
                "dry_run": False,
            }
        ],
    )
    study.cfg["amendment_2"] = {
        "import": {
            **study.cfg["amendment_1"]["import"],
            "files": {
                **study.cfg["amendment_1"]["import"]["files"],
                "inherited_calls.jsonl": c.digest(study.out / "inherited_calls.jsonl"),
            },
        }
    }
    p.import_completed(dest, study.cfg)
    for artifact in ("texts.jsonl", "text_checks.jsonl", "hand_audit.jsonl"):
        expected = [
            r
            for r in c.read_jsonl(study.out / artifact)
            if r["condition"] != "cover_story"
        ]
        assert c.read_jsonl(dest / artifact) == expected
    assert c.spent(dest, "writer") == 3.0
    assert c.spent(dest, "checker") == 0.25
    assert not any(
        ":cover_story:" in r["request_id"] for r in c.read_jsonl(dest / "calls.jsonl")
    )
    c.validate_import(dest, study.cfg, False)


@pytest.mark.parametrize(
    "eligible,scored,exception",
    [
        (6, 6, False),
        (0, 0, False),
        (19, 19, False),
        (20, 6, True),
        (20, 20, False),
        (6, -1, True),
    ],
)
def test_predict_cover_fallback_uses_eligible_pairs_not_score_failures(
    study, monkeypatch, eligible, scored, exception
):
    import json
    from experiments.adversarial_crossed_text import evaluate as e

    cfg = study.cfg
    cfg["conditions"] = ["none", "cover_story"]
    cfg["analysis"]["bootstrap_resamples"] = 2
    cfg["amendment_2"] = {
        "prediction_rule": {
            "condition": "cover_story",
            "minimum_development_pairs": 20,
            "below_minimum": "heldout_descriptive_only_no_numeric_pilot_prediction",
        }
    }
    monkeypatch.setattr(e, "validate_variants", lambda *args: "manifest")
    monkeypatch.setattr(e, "describe", lambda *args: {})
    c.write_json(
        study.out / "score-4b-v03_student.identity.json",
        {
            "dry_run": True,
            "sources": c.sources(),
            "variants_manifest_sha256": "manifest",
        },
    )
    variants = [
        {"metadata": {"task_id": str(i), "split": "development", "text": "cover_story"}}
        for i in range(eligible)
    ]
    c.write_jsonl(study.out / "variants.jsonl", variants)
    (study.out / "scores").mkdir()
    rows = [
        {
            "id": f"{kind}:{i}:{code}",
            "task_id": str(i),
            "text": kind,
            "split": "development",
            "set": "main",
            "code": code,
            "score": value,
            "status": "ok",
        }
        for kind, n in (
            ("none", 19 if scored == -1 else 20),
            ("cover_story", max(0, scored)),
        )
        for i in range(n)
        for code, value in (("honest", 0.1), ("backdoored", 0.9))
    ]
    c.write_jsonl(study.out / "scores/4b-v03_student.jsonl", rows)
    args = SimpleNamespace(dry_run=True)
    if exception:
        with pytest.raises(SystemExit, match="20 paired"):
            e.predict(args, study.out, cfg)
    else:
        e.predict(args, study.out, cfg)
        result = json.loads((study.out / "predictions.mock.json").read_text())
        if eligible >= 20:
            assert set(result["predictions"]) == {"none", "cover_story"}
            assert result["descriptive_only"] == {}
            return
        assert set(result["predictions"]) == {"none"}
        assert (
            result["descriptive_only"]["cover_story"]["development_eligible_pairs"]
            == eligible
        )
        assert (
            result["descriptive_only"]["cover_story"]["frozen_numeric_pilot_prediction"]
            is None
        )


def test_cover_round_bound_refuses_whole_round_over_cap(study):
    cfg = study.cfg
    cfg["amendment_2"] = {"writer_model_role": "cover_writer", "max_attempts": 2}
    texts = {k: r for k, r in p.final_texts(study.out).items() if k[1] != "cover_story"}
    c.write_jsonl(study.out / "texts.jsonl", list(texts.values()))
    bound = p.cover_round_bound(
        study.out, cfg, study.tasks, study.pairs, study.triggers, texts
    )
    assert bound > 0
    cfg["budget_usd"]["writer"] = bound / 2
    with pytest.raises(SystemExit, match="entire remaining cover round"):
        p.cover_round_bound(
            study.out, cfg, study.tasks, study.pairs, study.triggers, texts
        )


def test_amendment2_generation_never_fills_or_retries_other_conditions(
    study, monkeypatch
):
    cfg = study.cfg
    cfg["amendment_2"] = {
        "writer_model_role": "cover_writer",
        "max_attempts": 2,
        "regenerate_conditions": ["cover_story"],
    }
    retained = [
        r
        for r in c.read_jsonl(study.out / "texts.jsonl")
        if r["condition"] != "cover_story"
        and (r["task_id"], r["condition"]) != ("a", "specific_accusation")
    ]
    c.write_jsonl(study.out / "texts.jsonl", retained)
    received = []

    class FakeCalls:
        def __init__(self, *args, **kwargs):
            assert args[2] == "writer"
            assert kwargs["model_role"] == "cover_writer"

        def chat(self, request_id, messages, attempt):
            received.append((request_id, messages))
            return {
                "status": "ok",
                "text": "Reasoned task requirement. " * 23,
                "finish_reason": "stop",
            }

    monkeypatch.setattr(p, "Calls", FakeCalls)
    monkeypatch.setattr(p, "cover_round_bound", lambda *args, **kwargs: 1.0)
    p.generate(SimpleNamespace(dry_run=False, limit=None), study.out, cfg)
    assert {request.split(":")[2] for request, _ in received} == {"cover_story"}
    after = c.read_jsonl(study.out / "texts.jsonl")
    assert [r for r in after if r["condition"] != "cover_story"] == retained
    assert all("print(2)" in messages[0]["content"] for _, messages in received)


def test_mock_cover_budget_ignores_exhausted_live_campaign(study):
    cfg = study.cfg
    cfg["campaign"]["output"] = str(study.out / "live-campaign")
    cfg["amendment_2"] = {"writer_model_role": "cover_writer", "max_attempts": 2}
    ledger = study.out / "live-campaign/spend.jsonl"
    ledger.parent.mkdir()
    c.write_jsonl(
        ledger,
        [
            {
                "id": "spent",
                "api": "tinker",
                "scope": "a4",
                "charge": cfg["campaign"]["caps_usd"]["tinker"],
            }
        ],
    )
    texts = {k: r for k, r in p.final_texts(study.out).items() if k[1] != "cover_story"}
    c.write_jsonl(study.out / "texts.jsonl", list(texts.values()))
    before = ledger.read_bytes()
    assert (
        p.cover_round_bound(
            study.out,
            cfg,
            study.tasks,
            study.pairs,
            study.triggers,
            texts,
            dry_run=True,
        )
        > 0
    )
    with pytest.raises(SystemExit, match="entire remaining cover round"):
        p.cover_round_bound(
            study.out, cfg, study.tasks, study.pairs, study.triggers, texts
        )
    assert ledger.read_bytes() == before


def test_fresh_mock_amendment2_runs_generate_check_audit_build(study):
    cfg = study.cfg
    cfg["amendment_2"] = {
        "writer_model_role": "cover_writer",
        "max_attempts": 2,
        "regenerate_conditions": ["cover_story"],
    }
    cfg["campaign"]["output"] = str(study.out / "unused-live-campaign")
    for name in ("texts.jsonl", "text_checks.jsonl", "hand_audit.jsonl"):
        c.write_jsonl(study.out / name, [])
    bind(study, mock=True)
    args = SimpleNamespace(dry_run=True, limit=None)
    p.generate(args, study.out, cfg)
    p.check(args, study.out, cfg)
    refresh_reads(study)
    p.audit_sample(study.out, cfg)
    p.build(study.out, cfg, dry_run=True)
    assert c.validate_variants(study.out, cfg, True)
    assert not (study.out / "unused-live-campaign").exists()
    assert all(r["dry_run"] for r in c.read_jsonl(study.out / "calls.jsonl"))
    assert {
        r["condition"]
        for r in c.read_jsonl(study.out / "texts.jsonl")
        if r.get("mock_import_fixture")
    } == {"task_restatement", "specific_accusation", "generic_accusation"}


def test_cover_retry_resume_reconstructs_bounded_feedback(study, monkeypatch):
    cfg = study.cfg
    cfg["amendment_2"] = {
        "writer_model_role": "cover_writer",
        "max_attempts": 2,
        "regenerate_conditions": ["cover_story"],
    }
    rows = [
        r
        for r in c.read_jsonl(study.out / "texts.jsonl")
        if r["condition"] != "cover_story"
    ]
    c.write_jsonl(study.out / "texts.jsonl", rows)
    received = []

    class FakeCalls:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, request_id, messages, attempt):
            received.append((request_id, messages))
            return {"status": "ok", "text": "short", "finish_reason": "stop"}

    monkeypatch.setattr(p, "Calls", FakeCalls)
    monkeypatch.setattr(p, "cover_round_bound", lambda *args, **kwargs: 1.0)
    args = SimpleNamespace(dry_run=True, limit=1)
    p.generate(args, study.out, cfg)
    assert len(received) == 2
    expected = received[-1]
    first = c.read_jsonl(study.out / "texts.jsonl")[-2]
    assert all(m["role"] == "user" for m in expected[1])
    assert all(m["content"] != "short" for m in expected[1])
    assert len(expected[1][-1]["content"].encode()) <= 512
    c.write_jsonl(study.out / "texts.jsonl", [*rows, first])
    received.clear()
    p.generate(args, study.out, cfg)
    assert received == [expected]
