"""Single-dispatch API monitors and durable, capped Anthropic batches."""

from __future__ import annotations

import argparse
import json
import math
import time

from experiments.adversarial_code_or_comments.common import load_key
from experiments.adversarial_code_or_comments.score import with_instruction
from experiments.adversarial_crossed_text.common import (
    ROOT,
    append,
    bind,
    config,
    digest,
    exclusive,
    frozen,
    instruction,
    output,
    prior,
    read_jsonl,
    sha,
    validate_variants,
    write_json,
)
from gleipnir.adversarial.attacker import error_details
from gleipnir.adversarial.frontier import (
    FrontierMonitor,
    anthropic_batch_params,
    anthropic_result_record,
    margin_fields,
    openrouter_score_payload,
    parse_score,
    score_prompt,
)


def monitor(name, cfg):
    allowed = set(FrontierMonitor.__dataclass_fields__)
    return FrontierMonitor(
        name=name,
        **{k: v for k, v in cfg["frontier"]["monitors"][name].items() if k in allowed},
    )


def selected(out, cfg):
    head = instruction("teacher", cfg)
    return [
        {**r, "prompt": with_instruction(r["prompt"], head)}
        for r in read_jsonl(out / "variants.jsonl")
        if r["metadata"]["split"] == "heldout"
    ]


def charges(out):
    last = {}
    for r in read_jsonl(out / "frontier" / "ledger.jsonl"):
        last[r["request_id"]] = r
    return list(last.values())


def reserve(out, cfg, m, request_id, worst, identity):
    from experiments.adversarial_crossed_text.campaign import Campaign

    campaign = Campaign(cfg, out, cfg["protocol_status"] != "frozen")
    ledger = out / "frontier" / "ledger.jsonl"
    ledger.parent.mkdir(exist_ok=True)
    records = charges(out)
    if any(r["request_id"] == request_id for r in records):
        raise SystemExit(
            "request already reserved; reconcile its outcome instead of redispatching"
        )
    total = sum(r["charged_usd"] for r in records if r["api"] == m.api)
    own = sum(r["charged_usd"] for r in records if r["monitor"] == m.name)
    if (
        total + worst > cfg["frontier"]["caps_usd"][m.api]
        or own + worst > cfg["frontier"]["monitors"][m.name]["cap_usd"]
    ):
        raise SystemExit(f"{m.name} spend cap reached; nothing sent")
    r = {
        "request_id": request_id,
        "monitor": m.name,
        "api": m.api,
        "status": "reserved",
        "charged_usd": worst,
        "identity": identity,
        "time": time.time(),
    }
    campaign_record = campaign.reserve(
        campaign.request_id(out, request_id),
        m.api,
        "crossed-frontier-" + m.name,
        worst,
        cfg["frontier"]["monitors"][m.name]["cap_usd"],
        group="crossed-" + m.api,
        group_cap=cfg["frontier"]["caps_usd"][m.api],
    )
    r["campaign_reservation"] = campaign_record
    r["campaign_path"] = str(campaign.path)
    append(ledger, r)
    return r


def settle(out, reservation, cost):
    charge = (
        cost
        if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0
        else reservation["charged_usd"]
    )
    append(
        out / "frontier" / "ledger.jsonl",
        {**reservation, "status": "settled", "charged_usd": charge},
    )
    if reservation.get("campaign_reservation"):
        from pathlib import Path

        from experiments.adversarial_crossed_text.campaign import Campaign

        campaign = Campaign(config(), out, config()["protocol_status"] != "frozen")
        campaign.path = Path(reservation["campaign_path"])
        campaign.ledger = campaign.path / "spend.jsonl"
        campaign.settle(reservation["campaign_reservation"], cost)


def request_record(m, row, key, post):
    """Exactly one HTTP dispatch. Refusals/transport failures are terminal."""
    from gleipnir.teachers.openrouter import (
        OpenRouterConfig,
        PromptRecord,
        binary_score_from_top_logprobs,
        extract_terminal_binary_top_logprobs,
        request_payload,
    )

    record = {"id": row["id"], **row["metadata"], "monitor": m.name}
    if m.mode == "probability":
        spec = OpenRouterConfig(
            model=m.model,
            max_tokens=m.max_tokens,
            top_logprobs=5,
            provider_sort=None,
            provider_only=m.provider_only,
            provider_max_prompt_price=m.extra.get("max_prompt_price"),
            provider_max_completion_price=m.extra.get("max_completion_price"),
            allow_fallbacks=False,
        )
        body = request_payload(PromptRecord(row["id"], row["prompt"], {}), spec)
    else:
        body = openrouter_score_payload(m, score_prompt(row["prompt"]))
    try:
        response = post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        record.update(
            cost=usage.get("cost"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            model=data.get("model"),
            provider=data.get("provider"),
        )
        reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        if reason in {"refusal", "content_filter"} or message.get("refusal"):
            return {**record, "status": "refusal"}
        if reason == "length":
            return {**record, "status": "truncation"}
        if m.mode == "probability":
            top, text, _ = extract_terminal_binary_top_logprobs(data)
            bounded = not {"0", "1"} <= set(top)
            if bounded:
                floor = min(top.values())
                top = {"0": floor, "1": floor, **top}
            value, logprobs = binary_score_from_top_logprobs(top)
            record.update(logit_0=logprobs["0"], logit_1=logprobs["1"], bounded=bounded)
        else:
            text = message.get("content") or ""
            try:
                value = parse_score(text)
            except ValueError:
                return {
                    **record,
                    "status": "invalid_score",
                    "response_text": text[:2000],
                }
            record.update(margin_fields(value))
        if not math.isfinite(value):
            return {**record, "status": "nonfinite"}
        return {**record, "status": "ok", "score": value, "response_text": text[:2000]}
    except Exception as error:
        details = error_details(error, key)
        response = getattr(error, "response", None)
        if response is not None:
            details["http_status"] = response.status_code
        return {**record, "status": "infrastructure_failure", "error": details}


def calibration_records(name, cfg, out):
    artifact = cfg["frontier"]["calibration_artifacts"][name]
    path = ROOT / artifact["path"]
    if not path.exists() or digest(path) != artifact["sha256"]:
        raise SystemExit("frontier calibration file missing or SHA mismatch")
    old, _ = prior(cfg)
    expected = set(old["heldout"])
    selection = json.loads((out / "selection.json").read_text())
    study = set(selection["heldout"] + selection["development"])
    if study & expected:
        raise ValueError("frontier calibration overlaps selected study tasks")
    old_cfg_path = (
        ROOT
        / cfg["prior"]["directory"].replace("results/", "experiments/")
        / "config.json"
    )
    old_spec = json.loads(old_cfg_path.read_text())["amendment_7"]["frontier"][
        "monitors"
    ][name]
    for key in (
        "model",
        "mode",
        "max_tokens",
        "reasoning_effort",
        "provider_only",
        "extra",
    ):
        if old_spec.get(key) != cfg["frontier"]["monitors"][name].get(key):
            raise ValueError("frontier calibration request settings changed")
    instruction("teacher", cfg)
    found = {}
    for r in read_jsonl(path):
        if (
            r.get("code") == "honest"
            and r.get("text") == "none"
            and r.get("set") == "main"
        ):
            if r["task_id"] not in expected or r.get("split") != "heldout":
                raise ValueError("unexpected frontier calibration membership")
            if r.get("status") == "ok":
                found[r["id"]] = r
    if {r["task_id"] for r in found.values()} != expected:
        raise SystemExit(
            "frontier calibration failures/missing tasks; thresholds undefined"
        )
    return [{**r, "set": "calibration", "split": "calibration"} for r in found.values()]


def run(args, out, cfg):
    import requests

    m = monitor(args.monitor, cfg)
    if m.api != "openrouter":
        raise SystemExit("use submit/collect for Anthropic")
    key = load_key("OPENROUTER_API_KEY", cfg)
    path = out / "frontier" / f"{m.name}.jsonl"
    path.parent.mkdir(exist_ok=True)
    done = {r["id"] for r in read_jsonl(path)}
    reserved = {r["request_id"] for r in charges(out)}
    rows = [
        r
        for r in selected(out, cfg)
        if r["id"] not in done and f"{m.name}:{r['id']}" not in reserved
    ]
    for n, row in enumerate(rows[: args.limit or len(rows)], 1):
        worst = m.worst_case_usd(len(row["prompt"].encode()) + 2048)
        reservation = reserve(
            out, cfg, m, f"{m.name}:{row['id']}", worst, sha(row["prompt"])
        )
        record = request_record(m, row, key, requests.post)
        append(path, record)
        settle(out, reservation, record.get("cost"))
        if record["status"] == "infrastructure_failure":
            details = record.get("error", {})
            if (
                details.get("http_status") in (401, 403)
                or details.get("exception_type") == "HTTPError"
            ):
                raise SystemExit("API HTTP/authentication failure; stop dispatch")
        if n % 20 == 0:
            print(f"{m.name} {n}/{len(rows)}", flush=True)


def submit(args, out, cfg):
    import anthropic

    m = monitor(args.monitor, cfg)
    if m.api != "anthropic-batch":
        raise SystemExit("submit requires an Anthropic monitor")
    state = out / "frontier" / f"{m.name}.pending.json"
    if state.exists():
        raise SystemExit("pending batch state exists; collect first")
    path = out / "frontier" / f"{m.name}.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    # All previously reserved rows remain terminal even after a transport crash.
    reserved_ids = {
        i
        for r in charges(out)
        if r["monitor"] == m.name and isinstance(r.get("identity"), dict)
        for i in r["identity"].get("row_ids", [])
    }
    rows = [r for r in selected(out, cfg) if r["id"] not in done | reserved_ids]
    rows = rows[: args.limit or 100]
    if not rows:
        print("no unreserved rows remain")
        return
    worst = sum(
        m.worst_case_usd(len(score_prompt(r["prompt"]).encode()) + 2048) for r in rows
    )
    ids = {f"r{i}": r["id"] for i, r in enumerate(rows)}
    rid = f"{m.name}:batch:{sha(json.dumps(ids, sort_keys=True))}"
    reservation = reserve(
        out,
        cfg,
        m,
        rid,
        worst,
        {
            "row_ids": list(ids.values()),
            "variants_sha256": digest(out / "variants.jsonl"),
        },
    )
    # Persist intent before SDK dispatch; SDK retries are explicitly disabled.
    write_json(state, {"batch_id": None, "ids": ids, "reservation": reservation})
    client = anthropic.Anthropic(
        api_key=load_key("ANTHROPIC_API_KEY", cfg), max_retries=0
    )
    batch = client.messages.batches.create(
        requests=[
            {"custom_id": cid, "params": anthropic_batch_params(m, row["prompt"])}
            for (cid, _), row in zip(ids.items(), rows, strict=True)
        ]
    )
    write_json(state, {"batch_id": batch.id, "ids": ids, "reservation": reservation})
    print(f"submitted {len(rows)} rows, batch {batch.id}, reserved ${worst:.2f}")


def collect(args, out, cfg):
    import anthropic

    m = monitor(args.monitor, cfg)
    state_path = out / "frontier" / f"{m.name}.pending.json"
    state = json.loads(state_path.read_text())
    if not state["batch_id"]:
        raise SystemExit(
            "ambiguous submission; reconcile with provider, never redispatch"
        )
    client = anthropic.Anthropic(
        api_key=load_key("ANTHROPIC_API_KEY", cfg), max_retries=0
    )
    batch = client.messages.batches.retrieve(state["batch_id"])
    if batch.processing_status != "ended":
        print(batch.processing_status, batch.request_counts)
        return
    rows = {r["id"]: r for r in selected(out, cfg)}
    path = out / "frontier" / f"{m.name}.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    for result in client.messages.batches.results(state["batch_id"]):
        row = rows[state["ids"][result.custom_id]]
        if row["id"] in done:
            continue
        record = anthropic_result_record(m, result)
        if "text" in record:
            record["response_text"] = record.pop("text")
        record.update(id=row["id"], **row["metadata"], batch_id=state["batch_id"])
        append(path, record)
        done.add(row["id"])
    all_rows = [r for r in read_jsonl(path) if r.get("batch_id") == state["batch_id"]]
    if len({r["id"] for r in all_rows}) != len(state["ids"]):
        raise SystemExit(
            "batch results incomplete; preserve reservation and collect again"
        )
    cost = sum(r.get("cost") or 0 for r in all_rows)
    # Unknown per-row usage keeps a conservative full-batch reservation.
    if any(r.get("cost") is None for r in all_rows):
        cost = None
    settle(out, state["reservation"], cost)
    state_path.rename(
        state_path.with_name(f"{m.name}.{state['batch_id']}.completed.json")
    )
    print(f"collected {len(all_rows)} rows")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["run", "submit", "collect", "report"])
    p.add_argument("--out")
    p.add_argument(
        "--monitor", choices=list(config()["frontier"]["monitors"]), required=True
    )
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    cfg, out = config(), output(args.out)
    if args.limit is not None and args.limit < 1:
        p.error("limit must be positive")
    if args.stage == "report":
        from experiments.adversarial_crossed_text.evaluate import describe

        rows = read_jsonl(
            out / "frontier" / f"{args.monitor}.jsonl"
        ) + calibration_records(args.monitor, cfg, out)
        write_json(
            out / f"report_{args.monitor}_heldout.json",
            describe(rows, read_jsonl(out / "variants.jsonl"), cfg, "heldout"),
        )
        return
    frozen(cfg, heldout=True, out=out)
    validate_variants(out, cfg, False)
    calibration_records(args.monitor, cfg, out)
    bind(
        out,
        cfg,
        f"frontier-{args.monitor}",
        {"variants_sha256": digest(out / "variants.jsonl")},
    )
    with exclusive(out, "frontier"):
        {"run": run, "submit": submit, "collect": collect}[args.stage](args, out, cfg)


if __name__ == "__main__":
    main()
