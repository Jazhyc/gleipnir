"""Fixed-winner frontier transfer, with per-request budget reservations.

No search, GPU imports, generation, retries or calibration calls. The historical
generation manifest remains unchanged; this amendment owns a separate identity.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

from experiments.adversarial_code_or_comments.common import load_key
from experiments.adversarial_crossed_text import frontier as crossed
from experiments.adversarial_crossed_text.campaign import Campaign
from gleipnir.adversarial.contract import join_prompt, split_prompt
from gleipnir.adversarial.eval_sets import file_sha256
from gleipnir.adversarial.frontier import (
    anthropic_batch_params,
    anthropic_result_record,
    score_prompt,
)

from .common import HERE, ROOT, append, config, digest, fixed, lock, read, sha, unique
from .dependency import arm_config, instructed_pairs, instruction
from .pipeline import (
    honest_prompt,
    job,
    require_dual_seal,
    sealed,
    validate,
    validate_scores,
)
from .selection import benchmark_pairs

ARMS = ("4b-v03", "27b-merged")
TARGETS = {"kimi-k3", "gpt-6.1-sol", "claude-sonnet-5.5"}


def settings() -> tuple[dict, dict]:
    spec = json.loads((HERE / "frontier_config.json").read_text())
    source = ROOT / spec["source_config"]
    if file_sha256(source) != spec["source_config_sha256"]:
        raise ValueError("pinned crossed-text configuration changed")
    cfg = json.loads(source.read_text())
    if (
        set(spec["monitors"]) != TARGETS
        or set(spec["calibration_artifacts"]) != TARGETS
    ):
        raise ValueError("need exactly all three registered frontier targets")
    if spec["endpoints"] != {
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "anthropic-batch": "https://api.anthropic.com",
    } or spec["caps_usd"] != {"openrouter": 3.0, "anthropic-batch": 1.5}:
        raise ValueError("A4 endpoints or caps changed")
    for name, m in spec["monitors"].items():
        if m != cfg["frontier"]["monitors"][name]:
            raise ValueError("pinned frontier settings changed")
        if (
            spec["calibration_artifacts"][name]
            != cfg["frontier"]["calibration_artifacts"][name]
        ):
            raise ValueError("pinned frontier calibration changed")
    return spec, cfg


def reservation_monitor(m):
    """Reserve at permitted routing ceilings, retaining pinned request settings."""
    return replace(
        m,
        usd_per_m_prompt=max(m.usd_per_m_prompt, m.extra.get("max_prompt_price", 0)),
        usd_per_m_completion=max(
            m.usd_per_m_completion, m.extra.get("max_completion_price", 0)
        ),
    )


def source_hashes() -> dict:
    paths = list(HERE.glob("*.py")) + [HERE / "frontier_config.json"]
    paths += [
        ROOT / p
        for p in (
            "experiments/adversarial_crossed_text/frontier.py",
            "experiments/adversarial_crossed_text/common.py",
            "experiments/adversarial_crossed_text/campaign.py",
            "experiments/adversarial_code_or_comments/common.py",
            "src/gleipnir/adversarial/frontier.py",
            "src/gleipnir/teachers/openrouter.py",
        )
    ]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}


def calibration(name: str, spec: dict, crossed_cfg: dict, tasks: set) -> list[dict]:
    rows = crossed.calibration_records(name, crossed_cfg, ROOT / crossed_cfg["output"])
    if len(rows) != 200 or len({r["task_id"] for r in rows}) != 200:
        raise ValueError("need the pinned full 200-row honest calibration")
    if tasks & {r["task_id"] for r in rows}:
        raise ValueError("calibration overlaps A4 evaluation")
    return rows


def arm_rows(out: Path, cfg: dict, pairs: dict) -> tuple[dict, list[dict]]:
    """Reconstruct frozen winners without rewriting their source manifest."""
    manifest = json.loads((out / "manifest.json").read_text())
    require_dual_seal(out, manifest)
    for t, codes in manifest["pair_hashes"].items():
        if any(sha(pairs[t][c]["prompt"]) != value for c, value in codes.items()):
            raise ValueError("original APPS prompt changed")
    # The amendment changes only these two existing files. All other frozen
    # generation dependencies still have to match byte for byte.
    amended = {str((HERE / p).relative_to(ROOT)) for p in ("README.md", "pipeline.py")}
    for path, value in manifest["sources"].items():
        if path in amended:
            data = subprocess.run(
                ["git", "show", f"da71a19:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            import hashlib

            observed = hashlib.sha256(data).hexdigest()
        else:
            observed = file_sha256(ROOT / path)
        if observed != value:
            raise ValueError(f"frozen generation source changed: {path}")
    records = sealed(out, manifest)
    ranked = json.loads((out / "winners.json").read_text())
    receipt = json.loads((out / "search_scores.json").read_text())
    search = json.loads((out / "search_jobs.json").read_text())
    validate_scores(search, receipt)
    if ranked["manifest"] != digest(manifest) or ranked["receipt_sha256"] != digest(
        receipt
    ):
        raise ValueError("fixed winner identity changed")
    head = instruction(cfg, "teacher")
    rows = []
    for task, winner in ranked["winners"].items():
        cid = winner["winner"]
        record = records[cid] if cid else None
        if record:
            checked = validate(record["response"], pairs[task]["backdoored"])
            if checked["status"] != "ok" or any(
                record.get(k) != v for k, v in checked.items()
            ):
                raise ValueError("winner preservation changed")
        for code in ("honest", "backdoored"):
            original = pairs[task][code]["prompt"]
            edited = (
                original
                if not record
                else (
                    record["prompt"]
                    if code == "backdoored"
                    else honest_prompt(original, record["edits"])
                )
            )
            for kind, prompt in (("original", original), ("winner", edited)):
                rendered = join_prompt(head, split_prompt(prompt).trajectory)
                rows.append(
                    job(
                        f"{task}:{code}:{kind}",
                        task,
                        kind,
                        rendered,
                        code=code,
                        winner=cid if kind == "winner" else None,
                    )
                )
    return {"manifest": digest(manifest), "winners_sha256": digest(ranked)}, rows


def build(study: Path, spec: dict, crossed_cfg: dict) -> dict:
    cfg = config()
    raw = benchmark_pairs(cfg)
    arms = {}
    tasks = set()
    for arm in ARMS:
        acfg = arm_config(cfg, arm)
        binding, rows = arm_rows(study / arm, acfg, instructed_pairs(raw, acfg))
        arms[arm] = (binding, rows)
        tasks.update(r["task_id"] for r in rows)
    monitors = {}
    for name in spec["monitors"]:
        m = crossed.monitor(name, crossed_cfg)
        cal = calibration(name, spec, crossed_cfg, tasks)
        identity = {
            "monitor": name,
            "backend": "frontier_api",
            "instruction": "teacher",
            "instruction_sha256": sha(instruction(cfg, "teacher")),
            "endpoint": spec["endpoints"][m.api],
            "settings": spec["monitors"][name],
            "calibration_artifact": spec["calibration_artifacts"][name],
        }
        identity["runtime_receipt_sha256"] = digest(identity)
        batches, requests = {}, {}
        for arm, (binding, rows) in arms.items():
            jobs = [
                job(
                    f"calibration:{r['task_id']}",
                    r["task_id"],
                    "calibration",
                    join_prompt(
                        instruction(cfg, "teacher"),
                        split_prompt(raw[r["task_id"]]["honest"]["prompt"]).trajectory,
                    ),
                )
                for r in cal
            ] + rows
            batches[arm] = {**binding, "monitor_identity": identity, "jobs": jobs}
            for row in rows:
                # The same exact prompt may serve multiple arms/control views.
                requests.setdefault(
                    row["prompt_sha256"],
                    {
                        "id": row["prompt_sha256"],
                        "prompt": row["prompt"],
                        "metadata": {},
                    },
                )
        worst = sum(
            reservation_monitor(m).worst_case_usd(
                len(
                    (
                        score_prompt(r["prompt"]) if m.mode == "score" else r["prompt"]
                    ).encode()
                )
                + 2048
            )
            for r in requests.values()
        )
        monitors[name] = {
            "identity": identity,
            "batches": batches,
            "requests": list(requests.values()),
            "worst_case_usd": worst,
        }
    return {
        "version": 1,
        "study": str(study.resolve()),
        "settings": spec,
        "sources": source_hashes(),
        "monitors": monitors,
    }


def request_bound(m, row: dict) -> float:
    prompt = score_prompt(row["prompt"]) if m.mode == "score" else row["prompt"]
    return reservation_monitor(m).worst_case_usd(len(prompt.encode()) + 2048)


def budget_reasons(plan: dict, campaign: Campaign, api: str, worst: float) -> list[str]:
    latest = list(campaign.latest().values())
    own = sum(r["charge"] for r in latest if r.get("group") == "a4-frontier-" + api)
    total = sum(r["charge"] for r in latest if r["api"] == api)
    reasons = []
    if own + worst > plan["settings"]["caps_usd"][api]:
        reasons.append(f"{api} A4 cap exceeded")
    if total + worst > campaign.cfg["campaign"]["caps_usd"][api]:
        reasons.append(f"{api} shared campaign cap exceeded")
    return reasons


def preflight(plan: dict, campaign: Campaign) -> dict:
    """Full bounds are diagnostic; dispatch gates each next request/batch."""
    return {
        "plan_sha256": digest(plan),
        "status": "per_request_reservations",
        "caps_usd": plan["settings"]["caps_usd"],
        "monitor_bounds": {
            n: {"requests": len(r["requests"]), "worst_case_usd": r["worst_case_usd"]}
            for n, r in plan["monitors"].items()
        },
        "campaign_latest_sha256": digest(list(campaign.latest().values())),
    }


def committed(plan: dict) -> None:
    if source_hashes() != plan["sources"]:
        raise ValueError("frontier implementation changed after preparation")
    paths = list(plan["sources"]) + [str((HERE / "README.md").relative_to(ROOT))]
    for path in paths:
        data = subprocess.run(
            ["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        if data != (ROOT / path).read_bytes():
            raise ValueError("commit the frontier amendment before calls")


def check_identity(record: dict, identity: dict) -> None:
    m = identity["settings"]
    if record.get("model") != m["model"] or (
        m.get("provider_only")
        and str(record.get("provider", "")).lower() != m["provider_only"].lower()
    ):
        raise ValueError("returned frontier model/provider drift; stop dispatch")


def check_returned_identity(record: dict, identity: dict) -> None:
    # Parsing exceptions can be labelled infrastructure_failure even though a
    # provider response, including its model/provider identity, was received.
    if (
        record.get("model") is not None
        or record.get("provider") is not None
        or record.get("result_type") == "succeeded"
        or (
            identity["settings"]["api"] == "openrouter"
            and record["status"] != "infrastructure_failure"
        )
    ):
        check_identity(record, identity)


def request_record(m, row: dict, key: str, post, identity: dict) -> dict:
    """Capture identity before the inherited parser touches choices/logprobs.

    The shared adapter catches parser exceptions. Keep the observed identity
    outside that adapter so even an empty choices array cannot erase drift.
    """
    observed = {}

    def checked_post(*args, **kwargs):
        response = post(*args, **kwargs)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            observed.update(model=data.get("model"), provider=data.get("provider"))
            check_returned_identity(
                {**observed, "status": "infrastructure_failure"}, identity
            )
        return response

    record = crossed.request_record(m, row, key, checked_post)
    return {**record, **observed}


def known_cost(rows: list[dict]) -> float | None:
    if all(
        not isinstance(r.get("cost"), bool)
        and isinstance(r.get("cost"), (int, float))
        and math.isfinite(r["cost"])
        and r["cost"] >= 0
        for r in rows
    ):
        return sum(r["cost"] for r in rows)
    return None


def stop_budget(plan, out, name, rows, worst, reasons):
    result = {
        "plan_sha256": digest(plan),
        "monitor": name,
        "status": "blocked_budget",
        "completed": len(rows),
        "planned": len(plan["monitors"][name]["requests"]),
        "next_reservation_usd": worst,
        "reasons": reasons,
    }
    fixed(out / f"{name}.budget_stop.json", result)
    print(json.dumps(result), flush=True)


def run(plan: dict, out: Path, cfg: dict, crossed_cfg: dict, monitor=None) -> None:
    """Settle one request or small batch before reserving more. Never retry."""
    import requests

    committed(plan)
    campaign = Campaign(cfg, out)
    for name in (
        (monitor,) if monitor else ("kimi-k3", "gpt-6.1-sol", "claude-sonnet-5.5")
    ):
        item = plan["monitors"][name]
        m = crossed.monitor(name, crossed_cfg)
        units = out / "units" / name
        done = unique(read(out / f"{name}.jsonl"))
        for record in done.values():
            check_returned_identity(record, item["identity"])
            if record.get("error", {}).get("http_status") in (401, 403):
                raise SystemExit("API authentication failure; stop dispatch")
        # A receipt saved just before a crash can be settled without a new call.
        for path in sorted(units.glob("*.intent.json")):
            state = json.loads(path.read_text())
            if state["plan_sha256"] != digest(plan):
                raise ValueError("dispatch intent plan changed")
            marker = path.with_suffix(".settled")
            if marker.exists():
                continue
            if m.api != "openrouter":
                raise SystemExit("unsettled Anthropic batch; collect first")
            if not set(state["ids"].values()) <= done.keys():
                raise SystemExit("ambiguous request; reconcile, never redispatch")
            campaign.settle(
                state["reservation"],
                known_cost([done[r] for r in state["ids"].values()]),
            )
            fixed(marker, {"settled": True})
        remaining = [r for r in item["requests"] if r["id"] not in done]
        if not remaining:
            print(f"{name} complete {len(done)}/{len(item['requests'])}", flush=True)
            continue
        if (out / f"{name}.budget_stop.json").exists():
            print(f"{name} previously stopped at cap", flush=True)
            continue
        key = None
        while remaining:
            rows, worst = [], 0.0
            reasons = []
            # Shrink the next batch to fit, stopping at the first row that cannot.
            for row in remaining[: 20 if m.api == "anthropic-batch" else 1]:
                bound = request_bound(m, row)
                reasons = budget_reasons(plan, campaign, m.api, worst + bound)
                if reasons:
                    break
                rows.append(row)
                worst += bound
            if not rows:
                stop_budget(
                    plan, out, name, done, request_bound(m, remaining[0]), reasons
                )
                break
            if key is None:
                key = load_key(
                    "OPENROUTER_API_KEY"
                    if m.api == "openrouter"
                    else "ANTHROPIC_API_KEY",
                    cfg,
                )
            ids = {f"r{i}": row["id"] for i, row in enumerate(rows)}
            unit_id = digest(ids)
            path = units / f"{unit_id}.intent.json"
            if path.exists():
                raise SystemExit("dispatch intent exists; never redispatch")
            reservation = campaign.reserve(
                campaign.request_id(out, digest(plan) + ":" + name + ":" + unit_id),
                m.api,
                "a4-frontier-" + name,
                worst,
                plan["settings"]["caps_usd"][m.api],
                group="a4-frontier-" + m.api,
                group_cap=plan["settings"]["caps_usd"][m.api],
            )
            state = {
                "plan_sha256": digest(plan),
                "ids": ids,
                "reservation": reservation,
            }
            fixed(path, state)
            if m.api == "anthropic-batch":
                import anthropic

                client = anthropic.Anthropic(
                    api_key=key, base_url=item["identity"]["endpoint"], max_retries=0
                )
                batch = client.messages.batches.create(
                    requests=[
                        {
                            "custom_id": cid,
                            "params": anthropic_batch_params(m, row["prompt"]),
                        }
                        for cid, row in zip(ids, rows, strict=True)
                    ]
                )
                fixed(path.with_suffix(".pending"), {**state, "batch_id": batch.id})
                print(
                    f"{name} submitted {len(rows)} rows; collect before next batch",
                    flush=True,
                )
                return
            record = request_record(m, rows[0], key, requests.post, item["identity"])
            append(out / f"{name}.jsonl", record)
            campaign.settle(reservation, known_cost([record]))
            fixed(path.with_suffix(".settled"), {"settled": True})
            done[record["id"]] = record
            check_returned_identity(record, item["identity"])
            if record.get("error", {}).get("http_status") in (401, 403):
                raise SystemExit("API authentication failure; stop dispatch")
            remaining = remaining[len(rows) :]
            if len(done) % 20 == 0 or not remaining:
                print(f"{name} {len(done)}/{len(item['requests'])}", flush=True)


def collect(plan: dict, out: Path, cfg: dict, crossed_cfg: dict) -> None:
    import anthropic

    committed(plan)
    name = "claude-sonnet-5.5"
    item = plan["monitors"][name]
    m = crossed.monitor(name, crossed_cfg)
    intents = [
        p
        for p in (out / "units" / name).glob("*.intent.json")
        if not p.with_suffix(".settled").exists()
    ]
    if len(intents) != 1:
        raise SystemExit("need exactly one unsettled batch")
    path = intents[0]
    pending = path.with_suffix(".pending")
    if not pending.exists():
        raise SystemExit("ambiguous submission; reconcile, never redispatch")
    state = json.loads(pending.read_text())
    if state["plan_sha256"] != digest(plan):
        raise ValueError("batch plan changed")
    client = anthropic.Anthropic(
        api_key=load_key("ANTHROPIC_API_KEY", cfg),
        base_url=item["identity"]["endpoint"],
        max_retries=0,
    )
    batch = client.messages.batches.retrieve(state["batch_id"])
    if batch.processing_status != "ended":
        print(batch.processing_status, flush=True)
        return
    done = unique(read(out / f"{name}.jsonl"))
    for result in client.messages.batches.results(state["batch_id"]):
        rid = state["ids"][result.custom_id]
        if rid in done:
            continue
        record = anthropic_result_record(m, result)
        if result.result.type == "succeeded":
            record["model"] = result.result.message.model
        record.update(id=rid, batch_id=state["batch_id"])
        append(out / f"{name}.jsonl", record)
        done[rid] = record
    ids = set(state["ids"].values())
    if not ids <= done.keys():
        raise ValueError("incomplete batch results; retain full reservation")
    batch_rows = [done[r] for r in ids]
    Campaign(cfg, out).settle(state["reservation"], known_cost(batch_rows))
    # Check every stored row, including partial results from a previous collect.
    for record in batch_rows:
        check_returned_identity(record, item["identity"])
        if record.get("error", {}).get("http_status") in (401, 403):
            raise SystemExit("API authentication failure; stop dispatch")
    fixed(path.with_suffix(".settled"), {"settled": True})
    print(
        f"{name} collected {len(batch_rows)}; "
        f"total {len(done)}/{len(item['requests'])}",
        flush=True,
    )


def validate_contract_receipt(batch: dict, receipt: dict) -> None:
    """API request identity/calibration validation is distinct from a GPU canary."""
    identity = batch["monitor_identity"]
    binding = {k: v for k, v in identity.items() if k != "runtime_receipt_sha256"}
    if digest(binding) != identity["runtime_receipt_sha256"]:
        raise ValueError("frontier request contract mismatch")
    if receipt.get("request_contract_sha256") != digest(identity):
        raise ValueError("missing frontier request contract validation")
    if receipt.get("calibration_artifact") != identity["calibration_artifact"]:
        raise ValueError("frontier calibration binding mismatch")


def import_scores(plan: dict, out: Path, crossed_cfg: dict) -> None:
    for name, item in plan["monitors"].items():
        rows = unique(read(out / f"{name}.jsonl"))
        if rows.keys() != {r["id"] for r in item["requests"]}:
            raise ValueError("every frontier request needs a terminal receipt")
        for r in rows.values():
            check_returned_identity(r, item["identity"])
        cal = {
            r["task_id"]: r
            for r in calibration(name, plan["settings"], crossed_cfg, set())
        }
        for arm, batch in item["batches"].items():
            imported = []
            for j in batch["jobs"]:
                r = (
                    cal[j["task_id"]]
                    if j["kind"] == "calibration"
                    else rows[j["prompt_sha256"]]
                )
                status = r["status"]
                if status not in (
                    "ok",
                    "refusal",
                    "truncation",
                    "infrastructure_failure",
                ):
                    status = (
                        "invalid_response"
                        if status in ("invalid_score", "nonfinite")
                        else "infrastructure_failure"
                    )
                imported.append(
                    {
                        **r,
                        "id": j["id"],
                        "prompt_sha256": j["prompt_sha256"],
                        "status": status,
                        "raw_status": r["status"],
                        "reused_calibration": j["kind"] == "calibration",
                    }
                )
            receipt = {
                "batch_sha256": digest(batch),
                "monitor_identity": item["identity"],
                "request_contract_sha256": digest(item["identity"]),
                "calibration_artifact": item["identity"]["calibration_artifact"],
                "rows": imported,
            }
            validate_scores(batch, receipt)
            target = Path(plan["study"]) / arm
            with lock(target):
                fixed(target / f"transfer_{name}_jobs.json", batch)
                fixed(target / f"transfer_{name}_scores.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "collect", "import"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--monitor", choices=sorted(TARGETS))
    args = parser.parse_args()
    spec, crossed_cfg = settings()
    cfg = config()
    out = args.out or ROOT / spec["output"]
    with lock(out):
        if args.stage == "prepare":
            plan = build(ROOT / cfg["output"], spec, crossed_cfg)
            fixed(out / "plan.json", plan)
            gate = preflight(plan, Campaign(cfg, out))
            fixed(out / "preflight.json", gate)
            print(json.dumps(gate, indent=2))
            return
        plan = json.loads((out / "plan.json").read_text())
        if plan != build(ROOT / cfg["output"], spec, crossed_cfg):
            raise ValueError("frontier plan inputs changed")
        if args.stage == "run":
            run(plan, out, cfg, crossed_cfg, args.monitor)
        elif args.stage == "collect":
            collect(plan, out, cfg, crossed_cfg)
        else:
            import_scores(plan, out, crossed_cfg)


if __name__ == "__main__":
    main()
