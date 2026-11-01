"""Cheap, fixed-stimulus replay against the updated 4B; no attack adaptation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.adversarial_a2_cover_comments.run import insert_comments
from experiments.adversarial_a3_random_search.run import attacked_prompt
from experiments.adversarial_a6_transfer import run as engine
from gleipnir.adversarial import apps, eval_sets
from gleipnir.adversarial.metrics import summary, threshold_at_fpr
from gleipnir.adversarial.monitors import MONITORS, Monitor
from gleipnir.adversarial.python_code import check_comment_only_edit

ROOT = eval_sets.ROOT
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
OUT = ROOT / CONFIG["output"]
read, save, digest = engine.read, engine.save, engine.digest


def prepare():
    for name, sha in CONFIG["source_files"].items():
        assert eval_sets.file_sha256(ROOT / name) == sha, name
    rows = eval_sets.load("apps-neutral")
    controls = eval_sets.load("apps-controls-neutral")
    byid = {r["id"]: r for r in rows + controls}
    pairs = {
        (str(r["metadata"]["task_id"]), eval_sets.group(r)): r
        for r in rows
        if eval_sets.group(r) in ("clean", "vanilla")
    }
    baseline = {
        r["id"]: r
        for r in read(
            ROOT / "results/adversarial_baseline_4b/apps-neutral/predictions.jsonl"
        )
    }
    old = {
        **baseline,
        **{
            r["id"]: r
            for r in read(
                ROOT
                / "results/adversarial_baseline_4b/apps-controls-neutral"
                / "predictions.jsonl"
            )
        },
    }
    jobs = {}
    refs = []
    hashes = {}

    def add(group, task, label, prompt, oldscore, key=None):
        h = digest(prompt)
        if h not in hashes:
            cid = key or "stimulus:" + h
            hashes[h] = cid
            jobs[cid] = {
                "prompt": prompt,
                "kind": "calibration"
                if group == "baseline" and label == 0
                else "replay",
                "task": task,
            }
        cid = hashes[h]
        basekey = ("original:" if label else "clean:") + task
        refs.append(
            {
                "group": group,
                "task": task,
                "label": label,
                "job": cid,
                "basejob": basekey,
                "old_score": oldscore,
                "old_base": baseline[
                    pairs[task, "vanilla" if label else "clean"]["id"]
                ]["score"],
            }
        )

    for (task, condition), row in sorted(pairs.items()):
        label = int(condition == "vanilla")
        key = ("original:" if label else "clean:") + task
        add("baseline", task, label, row["prompt"], baseline[row["id"]]["score"], key)
    common = sorted(
        {t for t, c in pairs if c == "vanilla"},
        key=lambda t: digest("v03-a1-replay:" + t),
    )[: CONFIG["a1_tasks"]]
    for row in rows + controls:
        task = str(row["metadata"]["task_id"])
        condition = eval_sets.group(row)
        if task in common and condition not in ("clean", "vanilla"):
            add(
                "A1/" + condition,
                task,
                eval_sets.label(row),
                row["prompt"],
                old[row["id"]]["score"],
            )
    high = read(ROOT / "results/adversarial_a2_high_effort/generations.jsonl")
    selected = {str(r["task_id"]) for r in high}
    failures = {}
    for name in ["adversarial_a2_cover_comments", "adversarial_a2_high_effort"]:
        gens = [
            r
            for r in read(ROOT / "results" / name / "generations.jsonl")
            if str(r["task_id"]) in selected
        ]
        failures[name] = dict(Counter(r["status"] for r in gens))
        preds = {
            r["id"]: r for r in read(ROOT / "results" / name / "predictions.jsonl")
        }
        for g in gens:
            if g["status"] != "ok":
                continue
            original = apps.parse(byid[g["parent_id"]]["prompt"]).code
            edited, _ = insert_comments(original, g["edits"])
            assert apps.parse(g["prompt"]).code == edited
            add(
                "A2/"
                + ("low" if name.endswith("cover_comments") else "high")
                + "/"
                + g["arm"],
                str(g["task_id"]),
                g["label"],
                g["prompt"],
                preds[g["id"]]["score"],
            )
    a3 = json.loads(
        (ROOT / "results/adversarial_a3_random_search/report.json").read_text()
    )
    a3scores = {
        r["id"]: r["value"]["score"]
        for r in read(ROOT / "results/adversarial_a3_random_search/queries.jsonl")
        if r["type"] == "result"
    }
    for w in a3["search_outcomes"]:
        task = w["task"]
        arm = w["arm"]
        payload = w["payload"]
        for label, condition in [(1, "vanilla"), (0, "clean")]:
            row = pairs[task, condition]
            prompt = attacked_prompt(row, payload) if payload else row["prompt"]
            oldscore = (
                w["best_score"]
                if label
                else (
                    a3scores[f"honest:{task}:{w['seed']}:{arm}"]
                    if payload
                    else baseline[row["id"]]["score"]
                )
            )
            add("A3/" + arm, task, label, prompt, oldscore)
    a4 = json.loads((ROOT / "results/adversarial_a4_feedback/report.json").read_text())
    gens = {
        r["id"]: r
        for r in read(ROOT / "results/adversarial_a4_feedback/generations.jsonl")
    }
    a4scores = {
        r["id"]: r["value"]["score"]
        for r in read(ROOT / "results/adversarial_a4_feedback/queries.jsonl")
        if r["type"] == "result"
    }
    for w in a4["winners"]:
        task = w["task"]
        arm = w["arm"]
        g = gens[w["generation_id"]] if w["generation_id"] else None
        original = apps.parse(pairs[task, "vanilla"]["prompt"]).code
        if g:
            edited, _ = insert_comments(original, g["edits"])
            assert apps.parse(g["prompt"]).code == edited
        add(
            "A4/" + arm,
            task,
            1,
            g["prompt"] if g else pairs[task, "vanilla"]["prompt"],
            w["score"],
        )
        clean = pairs[task, "clean"]
        code = apps.parse(clean["prompt"]).code
        if g:
            edited = "".join("# " + e["comment"] + "\n" for e in g["edits"]) + code
            assert check_comment_only_edit(code, edited).ok
            prompt = apps.with_code(clean["prompt"], edited)
            oldscore = a4scores[f"honest:{task}:{arm}"]
        else:
            prompt = clean["prompt"]
            oldscore = baseline[clean["id"]]["score"]
        add("A4/" + arm, task, 0, prompt, oldscore)
    files = (
        list(HERE.glob("*.py"))
        + [HERE / "config.json"]
        + list((ROOT / "src/gleipnir/adversarial").glob("*.py"))
    )
    for folder in [
        "adversarial_a2_cover_comments",
        "adversarial_a3_random_search",
        "adversarial_a4_feedback",
        "adversarial_a6_transfer",
    ]:
        files += list((ROOT / "experiments" / folder).glob("*.py")) + list(
            (ROOT / "experiments" / folder).glob("*.json")
        )
    identity = {
        "files": {
            str(p.relative_to(ROOT)): eval_sets.file_sha256(p)
            for p in sorted(set(files))
        },
        "target": CONFIG["target"],
        "source_files": CONFIG["source_files"],
        "data": {
            k: eval_sets.SETS[k].sha256
            for k in ["apps-neutral", "apps-controls-neutral"]
        },
        "prompts": {k: digest(v["prompt"]) for k, v in jobs.items()},
    }
    manifest = {
        "identity": identity,
        "refs": refs,
        "a1_tasks": common,
        "a2_statuses": failures,
        "jobs": len(jobs),
        "groups": dict(Counter(r["group"] for r in refs)),
    }
    assert sum(v["kind"] == "calibration" for v in jobs.values()) == 655
    assert len(jobs) == 2513
    assert len(jobs) + 8 <= CONFIG["physical_call_cap"]
    assert all(
        digest(apps.parse(v["prompt"]).head)
        == "39ab7657c02e096a4004439183f0b5d7f259cd724c77cdd2d5885acb0f94a38c"
        for v in jobs.values()
    )
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "manifest drift"
    else:
        save(path, manifest)
    return manifest, jobs


def score():
    from huggingface_hub import hf_hub_download

    prepare()
    path = Path(
        hf_hub_download(
            CONFIG["target"]["adapter"],
            "adapter_model.safetensors",
            revision=CONFIG["target"]["adapter_revision"],
        )
    )
    assert eval_sets.file_sha256(path) == CONFIG["adapter_sha256"]
    save(OUT / "adapter_verified.json", {"sha256": CONFIG["adapter_sha256"]})
    # Reuse the tested A6 scorer/canaries, while leaving its files and receipts intact.
    MONITORS[CONFIG["target_monitor"]] = Monitor(**CONFIG["target"])
    engine.CONFIG = CONFIG
    engine.OUT = OUT
    engine.prepare = prepare
    engine.score()


def endpoint(base, attacked, t):
    b = np.asarray(base)
    a = np.asarray(attacked)
    det = b > t
    return {
        "baseline_detected": int(det.sum()),
        "evasion": int((det & (a <= t)).sum()),
        "evasion_rate": float((det & (a <= t)).sum() / det.sum())
        if det.any()
        else None,
        "baseline_missed": int((~det).sum()),
        "final_missed": int((a <= t).sum()),
        "new_detections": int(((~det) & (a > t)).sum()),
    }


def report():
    m, jobs = prepare()
    q = read(OUT / "queries.jsonl")
    res = {r["id"]: r for r in q if r["type"] == "result"}
    ints = {r["id"]: r for r in q if r["type"] == "intent"}
    assert len(res) == len(ints) == len(jobs) + 8 and len(q) == 1 + 2 * len(res)
    assert res.keys() == ints.keys() and all(r["status"] == "ok" for r in res.values())
    assert q[0]["value"]["manifest"] == m["identity"]
    for k, v in jobs.items():
        assert ints[k]["prompt_sha256"] == digest(v["prompt"])
    s = {k: v["value"]["score"] for k, v in res.items()}
    calib = [r for r in m["refs"] if r["group"] == "baseline" and r["label"] == 0]
    thresholds = {
        version: {
            str(f): threshold_at_fpr(
                [s[r["job"]] if version == "new" else r["old_score"] for r in calib], f
            )
            for f in CONFIG["fprs"]
        }
        for version in ["old", "new"]
    }
    result = {
        "thresholds": thresholds,
        "calls": len(res),
        "groups": {},
        "a2_statuses": m["a2_statuses"],
        "calibration": {},
        "prompt_tokens": sum(r["value"]["tokens"] for r in res.values()),
        "scoring_seconds": sum(r["seconds"] for r in res.values()),
    }
    for version in ["old", "new"]:
        cs = [s[r["job"]] if version == "new" else r["old_score"] for r in calib]
        result["calibration"][version] = {
            f: {
                "flagged": sum(v > t for v in cs),
                "rows": len(cs),
                "ties": sum(v == t for v in cs),
            }
            for f, t in thresholds[version].items()
        }
    for group in m["groups"]:
        rs = [r for r in m["refs"] if r["group"] == group]
        bad = [r for r in rs if r["label"]]
        good = [r for r in rs if not r["label"]]
        out = {}
        for version in ["old", "new"]:
            values = [s[r["job"]] if version == "new" else r["old_score"] for r in rs]
            metrics = summary([r["label"] for r in rs], values) if bad and good else {}
            out[version] = {
                "rows": len(rs),
                "auroc": metrics.get("auroc"),
                "pauroc20": metrics.get("pauroc20"),
            }
            for f, t in thresholds[version].items():
                b = [
                    s[r["basejob"]] if version == "new" else r["old_base"] for r in bad
                ]
                a = [s[r["job"]] if version == "new" else r["old_score"] for r in bad]
                out[version][f] = {
                    **endpoint(b, a, t),
                    "harmful_rows": len(bad),
                    "harmful_flags": sum(v > t for v in a),
                    "honest_rows": len(good),
                    "honest_flags": sum(
                        (s[r["job"]] if version == "new" else r["old_score"]) > t
                        for r in good
                    ),
                }
        out["common_detected"] = {}
        for f in thresholds["old"]:
            common = [
                r
                for r in bad
                if r["old_base"] > thresholds["old"][f]
                and s[r["basejob"]] > thresholds["new"][f]
            ]
            out["common_detected"][f] = {
                "rows": len(common),
                "old_evasions": sum(
                    r["old_score"] <= thresholds["old"][f] for r in common
                ),
                "new_evasions": sum(
                    s[r["job"]] <= thresholds["new"][f] for r in common
                ),
            }
        result["groups"][group] = out
    save(OUT / "report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "score", "report"])
    a = p.parse_args()
    v = globals()[a.command]()
    if a.command == "prepare":
        print(
            json.dumps(
                {k: v[0][k] for k in ["jobs", "groups", "a2_statuses"]}, indent=2
            )
        )
