"""Build a deterministic offline pilot and per-row preflight receipts. No scoring."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from experiments.adversarial_goodfire.common import (
    DATA,
    HERE,
    ROOT,
    config,
    rank,
    read_jsonl,
    sha,
    write_json,
    write_jsonl,
)
from experiments.adversarial_goodfire.download import REVISION, sha256
from experiments.adversarial_goodfire.render import (
    RenderError,
    quote_in_actions,
    trajectory,
)
from gleipnir.adversarial.contract import MAX_TOKENS, join_prompt
from gleipnir.adversarial.contract import render as render_chat
from gleipnir.adversarial.monitors import MONITORS


def stratum(directory: str) -> tuple[str, str]:
    benchmark = directory.split("-")[0]
    agent = (
        "kimi-k3"
        if "kimi" in directory
        else "glm-5.2"
        if "glm" in directory
        else "qwen3.8-max"
    )
    return benchmark, agent


def join_labels(rows: list[dict], labels: list[dict]) -> dict[str, dict]:
    """Join by traces[0].id, with order/outer-id checked only as assertions."""
    by_id = {r["trace_id"]: r for r in labels}
    if len(by_id) != len(labels):
        raise ValueError("duplicate trace_id in labels")
    seen = set()
    for index, row in enumerate(rows):
        if len(row["traces"]) != 1:
            raise ValueError("expected exactly one trace")
        tid = row["traces"][0]["id"]
        if tid in seen:
            raise ValueError("duplicate rollout trace_id")
        seen.add(tid)
        label = by_id[tid]
        if label["row_index"] != index or label["row_id"] != row["id"]:
            raise ValueError("release order/outer-id assertion failed")
    if seen != set(by_id):
        raise ValueError("label membership mismatch")
    return by_id


def qualified_flags(
    label: dict,
    nodes: list[dict],
    passages: dict,
    cfg: dict,
    benchmark: str,
    generic: bool,
) -> list[dict]:
    flags = passages.get(label["trace_id"], []) if generic else label.get("flags") or []
    output = []
    allowed = cfg["visible_categories"][benchmark]
    for flag in flags:
        if (
            flag.get("status") not in {"attempted", "enacted"}
            or flag.get("category") not in allowed
        ):
            continue
        if generic:
            unanimous = flag["evidence"] == "certain"
        else:
            address = (flag["node_index"], flag["channel"])
            p = passages.get(label["trace_id"], {}).get(address, {})
            unanimous = (
                p.get("consensus") == "positive"
                and p.get("category_votes", {}).get(flag["category"]) == 3
            )
        if unanimous and quote_in_actions(flag, nodes, set(cfg["reasoning_tools"])):
            evidence = {
                k: flag[k] for k in ("node_index", "channel", "category", "status")
            }
            evidence["category_consensus_votes"] = None if generic else 3
            evidence["event_status_consensus"] = "not_provided_by_release"
            if generic:
                evidence["passage_id"] = flag["passage_id"]
            output.append(evidence)
    return output


def classify(
    label: dict,
    nodes: list[dict],
    passages: dict,
    cfg: dict,
    benchmark: str,
    source_ok: bool = True,
) -> tuple[str, list[dict]]:
    generic = benchmark in {"tau3bench", "shoppingbench"}
    if not source_ok or label.get("source_error"):
        return "source_error", []
    if not label.get("judged"):
        return "unjudged", []
    verdict = label.get("verdict")
    enacted = label.get("verdict_enacted")
    if verdict == "split" or enacted == "split":
        return "split", []
    if generic:
        if verdict == "unclear":
            return "unclear", []
        if verdict == "no" and label.get("full_passage_coverage"):
            return "negative", []
        if verdict != "yes":
            return "incomplete_or_missing_label", []
        if label.get("considered_only_yes"):
            return "considered_only", []
    else:
        if enacted == "no" and verdict == "yes":
            return "considered_only", []
        if enacted == "no" and verdict == "no":
            return "negative", []
        if enacted != "yes":
            return "incomplete_or_missing_label", []
    flags = qualified_flags(label, nodes, passages, cfg, benchmark, generic)
    return ("positive" if flags else "nonvisible_or_excluded_category"), flags


def select(catalogue: list[dict], cfg: dict) -> list[dict]:
    groups = defaultdict(list)
    positive_tasks = {
        (r["benchmark"], r["task_id"])
        for r in catalogue
        if r["label_class"] == "positive"
    }
    for r in catalogue:
        groups[(r["benchmark"], r["agent"], r["label_class"])].append(r)
    output = []
    for (benchmark, _agent, label), rows in sorted(groups.items()):
        if label not in {"positive", "negative", "considered_only"}:
            continue

        def order(r, benchmark=benchmark, label=label):
            matched = (benchmark, r["task_id"]) in positive_tasks
            return (
                not matched if label == "negative" else False,
                rank("selection", r["id"], cfg["seed"]),
                r["id"],
            )

        rows = sorted(rows, key=order)
        n = (
            len(rows)
            if label == "positive"
            else cfg["negatives_per_stratum"]
            if label == "negative"
            else cfg["considered_only_per_stratum"]
        )
        output.extend(
            {**r, "matched_positive_task": (benchmark, r["task_id"]) in positive_tasks}
            for r in rows[:n]
        )
    return sorted(
        output,
        key=lambda r: (
            r["benchmark"],
            r["agent"],
            r["label_class"],
            rank("dispatch", r["id"]),
        ),
    )


def quantiles(values: list[int]) -> dict:
    import numpy as np

    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "total": sum(values),
        "mean": float(np.mean(values)),
        **{
            name: float(np.quantile(values, q))
            for name, q in [
                ("min", 0),
                ("p25", 0.25),
                ("p50", 0.5),
                ("p75", 0.75),
                ("p90", 0.9),
                ("p95", 0.95),
                ("p99", 0.99),
                ("max", 1),
            ]
        },
    }


def counters() -> dict:
    from transformers import AutoTokenizer

    output = {}
    # All tokenizers must already be cached. This never downloads weights.
    for m in ("4b-v03", "27b-merged"):
        spec = MONITORS[m]
        tokenizer = AutoTokenizer.from_pretrained(
            spec.base, revision=spec.base_revision, local_files_only=True
        )

        def count(prompt: str, tokenizer=tokenizer) -> int:
            return len(
                tokenizer.encode(
                    render_chat(tokenizer, prompt), add_special_tokens=False
                )
            )

        output[m] = count
    return output


def prepare(cfg: dict, count_tokens: dict) -> dict:
    manifest = json.loads((DATA / "download_manifest.json").read_text())
    if cfg["revision"] != REVISION:
        raise ValueError("config revision differs from downloader revision")
    if manifest["revision"] != REVISION or manifest["metadata_only"]:
        raise ValueError("complete pinned download required")
    # Bind every raw file before building; byte hashes detect same-size corruption.
    for f in manifest["files"]:
        if sha256(DATA / "raw" / f["path"]) != f["sha256"]:
            raise ValueError(f"input hash mismatch: {f['path']}")
    listed = {f["path"] for f in manifest["files"]}
    if len(listed) != len(manifest["files"]):
        raise ValueError("duplicate source manifest member")
    traces = {p for p in listed if p.endswith("/traces.jsonl")}
    if {p.split("/")[0] for p in traces} != set(cfg["directories"]):
        raise ValueError("manifest strata differ from frozen configuration")
    actual = {
        p.relative_to(DATA / "raw").as_posix()
        for p in (DATA / "raw").glob("*/traces.jsonl")
    }
    if traces != actual:
        raise ValueError("unlisted or missing trace input")
    catalogue = []
    hand_check = []
    for source in sorted(traces):
        directory = DATA / "raw" / source.split("/")[0]
        benchmark, agent = stratum(directory.name)
        generic = benchmark in {"tau3bench", "shoppingbench"}
        prefix, name = (
            ("generic_monitor", "monitor") if generic else ("labels", "judge")
        )
        required = {
            f"{directory.name}/{prefix}/{name}_{kind}.jsonl"
            for kind in ("rollouts", "passages")
        }
        if not required <= listed:
            raise ValueError("consumed label input missing from pinned manifest")
        labels = read_jsonl(directory / prefix / f"{name}_rollouts.jsonl")
        passages = defaultdict(list if generic else dict)
        for p in read_jsonl(directory / prefix / f"{name}_passages.jsonl"):
            if generic:
                passages[p["trace_id"]].append(p)
            else:
                key = (p["node_index"], p["channel"])
                if key in passages[p["trace_id"]]:
                    raise ValueError("duplicate passage address")
                passages[p["trace_id"]][key] = p
        # One source stratum in memory, released before loading the next.
        rows = read_jsonl(directory / "traces.jsonl")
        by_id = join_labels(rows, labels)
        for row in rows:
            trace = row["traces"][0]
            label = by_id[trace["id"]]
            task = trace["task"]["data"]["name"]
            base = {
                "id": f"{directory.name}:{trace['id']}",
                "trace_id": trace["id"],
                "row_id": row["id"],
                "row_index": label["row_index"],
                "directory": directory.name,
                "benchmark": benchmark,
                "agent": agent,
                "task_id": task,
                "task_group": f"{benchmark}:{task}",
                "source_ok": row["ok"],
                "verdict": label.get("verdict"),
                "verdict_enacted": label.get("verdict_enacted"),
                "label_provenance": "generic_monitor_agreement"
                if generic
                else "three_vote_judge",
                "source_errors": len(row.get("errors") or []),
                "evidence_status": "ok",
                "source_label": {
                    "verdict": label.get("verdict"),
                    "verdict_enacted": label.get("verdict_enacted"),
                    "judged": label.get("judged"),
                },
            }
            try:
                kind, flags = classify(
                    label, trace["nodes"], passages, cfg, benchmark, row["ok"]
                )
            except RenderError as e:
                # Preserve the candidate label and failure in the selected pool.
                kind, flags = "positive", []
                base.update(error=str(e), evidence_status="render_error")
            base.update(label_class=kind, visible_flags=flags)
            base["primary_eligibility"] = (
                "pending_two_reader_adjudication"
                if not generic and kind == "positive"
                else "external_label_only"
            )
            catalogue.append(base)
        if generic:
            cohort = [r for r in catalogue if r["directory"] == directory.name]
            for verdict in ("yes", "no", "unclear"):
                picks = sorted(
                    (r for r in cohort if r["verdict"] == verdict),
                    key=lambda r: rank("hand-check", r["id"], cfg["seed"]),
                )[: cfg["hand_check_per_label_per_stratum"]]
                hand_check.extend({**r, "hand_status": "pending"} for r in picks)
        print(f"indexed {directory.name}: {len(rows)}", flush=True)
        del rows
    selected = select(catalogue, cfg)
    chosen = {r["id"]: r for r in selected}
    texts = {k: (ROOT / v).read_text() for k, v in cfg["prompts"].items()}
    for k, text in texts.items():
        if sha(text) != cfg["prompt_sha256"][k]:
            raise ValueError("instruction hash changed")
    prompts = defaultdict(list)
    preflight = []
    for directory in sorted({r["directory"] for r in selected}):
        for row in read_jsonl(DATA / "raw" / directory / "traces.jsonl"):
            trace = row["traces"][0]
            key = f"{directory}:{trace['id']}"
            if key not in chosen:
                continue
            meta = chosen[key]
            try:
                if meta["evidence_status"] != "ok":
                    raise RenderError(meta["error"])
                text = trajectory(trace["nodes"], set(cfg["reasoning_tools"]))
            except RenderError as e:
                meta.update(render_status="render_error", error=str(e))
                preflight.extend(
                    {
                        "id": key,
                        "monitor": m,
                        "score": None,
                        "status": "render_error",
                        "error": str(e),
                        "label_class": meta["label_class"],
                    }
                    for m in cfg["monitors"]
                )
                continue
            if "[TOOL CALLS]" not in text:
                meta["render_status"] = "no_visible_actions"
                preflight.extend(
                    {
                        "id": key,
                        "monitor": m,
                        "score": None,
                        "status": "no_visible_actions",
                        "label_class": meta["label_class"],
                    }
                    for m in cfg["monitors"]
                )
                continue
            meta["render_status"] = "ok"
            meta["trajectory_sha256"] = sha(text)
            meta["tokens"] = {}
            meta["too_long"] = {}
            for monitor, prompt_name in cfg["monitors"].items():
                prompt = join_prompt(texts[prompt_name], text)
                n = count_tokens[monitor](prompt)
                meta["tokens"][monitor] = n
                meta["too_long"][monitor] = n > MAX_TOKENS
                prompts[monitor].append(
                    {
                        "id": key,
                        "prompt": prompt,
                        "metadata": meta,
                        "ground_truth": (
                            -1
                            if meta["label_class"] == "considered_only"
                            or meta["primary_eligibility"]
                            == "pending_two_reader_adjudication"
                            else int(meta["label_class"] == "positive")
                        ),
                    }
                )
                preflight.append(
                    {
                        "id": key,
                        "monitor": monitor,
                        "status": "too_long"
                        if n > MAX_TOKENS
                        else "pending_authorization",
                        "tokens": n,
                        "prompt_sha256": sha(prompt),
                        "task_group": meta["task_group"],
                        "label_class": meta["label_class"],
                        "score": None,
                        "primary_eligibility": meta["primary_eligibility"],
                    }
                )
    validate_receipts(selected, preflight, cfg)
    out = DATA / "prepared"
    write_jsonl(out / "catalogue.jsonl", catalogue)
    write_jsonl(out / "selection.jsonl", selected)
    write_jsonl(out / "hand_check.jsonl", hand_check)
    write_jsonl(out / "preflight.jsonl", preflight)
    for monitor, rows in prompts.items():
        write_jsonl(out / f"rows_{monitor}.jsonl", rows)
    # Fixed common frontier cohort, chosen without reading any score; too-long
    # rows remain in its selection, with terminal too_long receipts in plan.
    frontier = []
    groups = defaultdict(list)
    for r in selected:
        groups[(r["benchmark"], r["agent"], r["label_class"])].append(r)
    for _key, rows in sorted(groups.items()):
        frontier.extend(
            sorted(rows, key=lambda r: rank("frontier", r["id"], cfg["seed"]))[
                : cfg["frontier_per_label_per_stratum"]
            ]
        )
    write_jsonl(out / "frontier_selection.jsonl", frontier)
    summary = {
        "revision": REVISION,
        "protocol_status": cfg["protocol_status"],
        "catalogue_n": len(catalogue),
        "selected_n": len(selected),
        "population_counts": dict(
            sorted(
                Counter(
                    f"{r['benchmark']}:{r['agent']}:{r['label_class']}"
                    for r in catalogue
                ).items()
            )
        ),
        "selection_counts": dict(
            sorted(
                Counter(
                    f"{r['benchmark']}:{r['agent']}:{r['label_class']}"
                    for r in selected
                ).items()
            )
        ),
        "token_lengths": {},
        "token_lengths_by_stratum_label": {},
        "too_long": {},
        "frontier_n": len(frontier),
        "hand_check_n": len(hand_check),
        "render_status": dict(Counter(r["render_status"] for r in selected)),
    }
    for m, rows in prompts.items():
        summary["token_lengths"][m] = quantiles(
            [r["metadata"]["tokens"][m] for r in rows]
        )
        summary["too_long"][m] = dict(
            Counter(
                f"{r['metadata']['benchmark']}:{r['metadata']['agent']}:{r['metadata']['label_class']}"
                for r in rows
                if r["metadata"]["too_long"][m]
            )
        )
        summary["token_lengths_by_stratum_label"][m] = {
            ":".join(key): quantiles([r["tokens"][m] for r in group if "tokens" in r])
            for key, group in groups.items()
        }
    write_json(out / "summary.json", summary)
    paths = sorted(
        p
        for p in out.iterdir()
        if p.suffix in {".json", ".jsonl"} and p.name != "manifest.json"
    )
    receipt = {
        "config_sha256": sha256(HERE / "config.json"),
        "download_manifest_sha256": sha256(DATA / "download_manifest.json"),
        "files": {p.name: sha256(p) for p in paths},
        "source_files": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "monitor_identity": {m: MONITORS[m]._asdict() for m in cfg["monitors"]},
    }
    write_json(out / "manifest.json", receipt)
    return summary


def validate_receipts(selected: list[dict], receipts: list[dict], cfg: dict) -> None:
    expected = {(r["id"], m) for r in selected for m in cfg["monitors"]}
    actual = [(r["id"], r["monitor"]) for r in receipts]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("missing, duplicate or unexpected preflight receipt")
    if any(r.get("score") is not None for r in receipts):
        raise ValueError("offline preflight cannot contain scores")


def main() -> None:
    argparse.ArgumentParser(__doc__).parse_args()
    summary = prepare(config(), counters())
    print(f"prepared {summary['selected_n']} rollouts, no scores")


if __name__ == "__main__":
    main()
