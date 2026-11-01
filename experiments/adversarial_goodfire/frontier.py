"""Prepare crossed-text-style request identities offline; never dispatch HTTP."""

from __future__ import annotations

import argparse
from dataclasses import fields

from experiments.adversarial_goodfire.common import (
    DATA,
    HERE,
    config,
    read_jsonl,
    sha,
    write_json,
    write_jsonl,
)
from experiments.adversarial_goodfire.download import sha256
from gleipnir.adversarial.frontier import (
    FrontierMonitor,
    openrouter_score_payload,
    score_prompt,
)
from gleipnir.teachers.openrouter import OpenRouterConfig, PromptRecord, request_payload


def monitor(name: str, cfg: dict) -> FrontierMonitor:
    allowed = {f.name for f in fields(FrontierMonitor)}
    return FrontierMonitor(
        name=name,
        **{k: v for k, v in cfg["frontier"]["monitors"][name].items() if k in allowed},
    )


def payload(m: FrontierMonitor, row: dict) -> dict:
    if m.mode == "score":
        body = openrouter_score_payload(m, score_prompt(row["prompt"]))
        body["provider"]["allow_fallbacks"] = False
        return body
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
    return request_payload(PromptRecord(row["id"], row["prompt"], {}), spec)


def plans(cfg: dict, selection: list[dict], rows: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in rows}
    output = []
    for name in cfg["frontier"]["monitors"]:
        m = monitor(name, cfg)
        for selected in selection:
            row = by_id.get(selected["id"])
            base = {
                "id": selected["id"],
                "monitor": name,
                "request_id": f"goodfire:{name}:{selected['id']}",
                "label_class": selected["label_class"],
                "score": None,
            }
            if selected.get("render_status") != "ok":
                output.append(
                    {**base, "status": selected.get("render_status", "render_error")}
                )
                continue
            if row is None:
                raise ValueError("rendered row missing from teacher prompts")
            if any(selected["too_long"].values()):
                output.append({**base, "status": "too_long"})
                continue
            body = payload(m, row)
            # A byte bound avoids presenting a Qwen count as a provider count.
            # It is intentionally loose, as in crossed-text's reservation path.
            prompt = body["messages"][0]["content"]
            token_bound = len(prompt.encode()) + 2048
            output.append(
                {
                    **base,
                    "status": "pending_authorization",
                    "prompt_sha256": sha(prompt),
                    "request_sha256": sha(
                        __import__("json").dumps(body, sort_keys=True)
                    ),
                    "provider_prompt_tokens_bound": token_bound,
                    "qwen_prompt_tokens_proxy": selected["tokens"]["27b-merged"],
                    "worst_case_usd": m.worst_case_usd(token_bound),
                    "payload": body,
                }
            )
    return output


def main() -> None:
    argparse.ArgumentParser(__doc__).parse_args()
    cfg = config()
    prepared = DATA / "prepared"
    manifest = __import__("json").loads((prepared / "manifest.json").read_text())
    if manifest["config_sha256"] != sha256(HERE / "config.json"):
        raise ValueError("config changed since build")
    for name, digest in manifest["files"].items():
        if sha256(prepared / name) != digest:
            raise ValueError(f"prepared artifact changed: {name}")
    result = plans(
        cfg,
        read_jsonl(prepared / "frontier_selection.jsonl"),
        read_jsonl(prepared / "rows_27b-merged.jsonl"),
    )
    out = prepared / "frontier"
    write_jsonl(out / "plans.jsonl", result)
    write_json(
        out / "manifest.json",
        {
            "mode": "offline_no_dispatch",
            "config_sha256": sha256(HERE / "config.json"),
            "build_manifest_sha256": sha256(prepared / "manifest.json"),
            "plans_sha256": sha256(out / "plans.jsonl"),
            "plans_n": len(result),
        },
    )
    print(f"{len(result)} offline request dispositions, no API calls")


if __name__ == "__main__":
    main()
