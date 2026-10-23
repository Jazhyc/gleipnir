"""Pre-registered five-call development-only gates; never score their proposals."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .common import ROOT, config, digest, fixed, read, sha, source_identity
from .pipeline import generate
from .selection import benchmark_pairs


def development_tasks(cfg: dict) -> list[str]:
    selection = json.loads((ROOT / cfg["step1"] / "selection.json").read_text())
    tasks = sorted(
        selection["development"], key=lambda t: sha(f"smoke:{cfg['smoke']['seed']}:{t}")
    )[:5]
    if len(tasks) != 5 or set(tasks) & (
        set(selection["heldout"]) | set(selection["calibration"])
    ):
        raise ValueError("smoke must use five distinct Step 1 development tasks")
    return tasks


def run(cfg: dict, out, pairs: dict) -> dict:
    tasks = development_tasks(cfg)
    fixed(
        out / "preregistration.json",
        {
            "smoke": cfg["smoke"],
            "tasks": tasks,
            "sources": source_identity(),
            "prompt": sha(
                (__import__("pathlib").Path(__file__).parent / "prompt.txt").read_text()
            ),
        },
    )
    summaries = []
    for index, spec in enumerate(cfg["smoke"]["candidates"]):
        stage_cfg = json.loads(json.dumps(cfg))
        stage_cfg.update(writer=spec, attempts=1, curve_attempts=[1])
        manifest = {
            "stage": "smoke",
            "config": stage_cfg,
            "sources": source_identity(),
            "selection": {"task_ids": tasks},
            "pair_hashes": {
                t: {"backdoored": sha(pairs[t]["backdoored"]["prompt"])} for t in tasks
            },
        }
        dest = out / str(index)
        fixed(dest / "manifest.json", manifest)
        generate(dest, manifest, pairs)
        records = read(dest / "generations.jsonl")
        valid = sum(r["status"] == "ok" for r in records)
        summary = {
            "candidate": index,
            "writer": spec,
            "valid": valid,
            "calls": len(records),
            "passed": len(records) == 5 and valid >= 4,
            "statuses": dict(Counter(r["status"] for r in records)),
            "charged_usd": sum(r["charged_usd"] for r in records),
            "receipts_sha256": digest(records),
            "mean_prompt_tokens": sum(
                (r.get("response", {}).get("usage", {}).get("prompt_tokens") or 0)
                for r in records
            )
            / 5,
            "mean_completion_tokens": sum(
                (r.get("response", {}).get("usage", {}).get("completion_tokens") or 0)
                for r in records
            )
            / 5,
            "usage_complete": all(
                all(
                    r.get("response", {}).get("usage", {}).get(k) is not None
                    for k in ("prompt_tokens", "completion_tokens")
                )
                for r in records
            ),
        }
        fixed(dest / "summary.json", summary)
        summaries.append(summary)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "writer"}), flush=True
        )
        if summary["passed"]:
            result = {
                "status": "passed",
                "selected_writer": spec,
                "summaries": summaries,
                "charged_usd": sum(s["charged_usd"] for s in summaries),
            }
            fixed(out / "gate.json", result)
            return result
    result = {
        "status": "failed",
        "summaries": summaries,
        "charged_usd": sum(s["charged_usd"] for s in summaries),
    }
    fixed(out / "gate.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    cfg = config()
    result = run(cfg, ROOT / cfg["smoke"]["output"], benchmark_pairs(cfg))
    print(json.dumps({"gate": result["status"], "charged_usd": result["charged_usd"]}))


def verify_evidence(cfg: dict, gate: dict, pairs: dict | None = None) -> dict:
    """Recompute the gate from sealed live calls and its committed registration."""
    import ast
    import subprocess

    from gleipnir.adversarial.eval_sets import file_sha256

    from .common import HERE
    from .pipeline import messages, sealed, validate

    out = ROOT / cfg["smoke"]["output"]
    prereg = json.loads((out / "preregistration.json").read_text())
    if prereg["smoke"] != cfg["smoke"] or prereg["tasks"] != development_tasks(cfg):
        raise ValueError("smoke preregistration changed")
    # Only orchestration may change between smoke and full freeze. Compare the
    # exact registered prompt, message builder and validator to their Git blob.
    path = "experiments/adversarial_a4_scale/pipeline.py"
    baseline = subprocess.check_output(
        ["git", "show", f"{cfg['smoke_registration_commit']}:{path}"],
        cwd=ROOT,
        text=True,
    )
    if sha(baseline) != prereg["sources"][path]:
        raise ValueError("smoke registration commit does not match observed source")

    def contract(source):
        names = {"validate", "messages", "DIRECTIVE", "REFUSAL"}
        nodes = []
        for node in ast.parse(source).body:
            targets = (
                {t.id for t in node.targets if isinstance(t, ast.Name)}
                if isinstance(node, ast.Assign)
                else {getattr(node, "name", None)}
            )
            if names & targets:
                nodes.append(ast.dump(node, include_attributes=False))
        return digest(nodes)

    if contract(baseline) != contract((ROOT / path).read_text()) or prereg[
        "prompt"
    ] != sha((HERE / "prompt.txt").read_text()):
        raise ValueError("prompt or validator changed after smoke")
    dependencies = [
        "experiments/adversarial_a2_cover_comments/run.py",
        "experiments/adversarial_a2_cover_comments/config.json",
        "experiments/adversarial_code_or_comments/llm.py",
        "src/gleipnir/adversarial/apps.py",
        "src/gleipnir/adversarial/python_code.py",
        "src/gleipnir/adversarial/contract.py",
    ]
    if any(file_sha256(ROOT / p) != prereg["sources"][p] for p in dependencies):
        raise ValueError("smoke validator/message dependencies changed")
    summaries = gate.get("summaries", [])
    if not 1 <= len(summaries) <= len(cfg["smoke"]["candidates"]):
        raise ValueError("missing smoke evidence")
    pairs = pairs or benchmark_pairs(cfg)
    for index, summary in enumerate(summaries):
        spec = cfg["smoke"]["candidates"][index]
        dest = out / str(index)
        manifest = json.loads((dest / "manifest.json").read_text())
        if (
            manifest.get("stage") != "smoke"
            or manifest["sources"] != prereg["sources"]
            or manifest["selection"]["task_ids"] != prereg["tasks"]
            or manifest["config"]["attempts"] != 1
            or manifest["config"]["writer"] != spec
        ):
            raise ValueError("smoke manifest/settings mismatch")
        identity = json.loads((dest / "generation_identity.json").read_text())
        seal = json.loads((dest / "generation_complete.json").read_text())
        if (
            identity != {"manifest": digest(manifest), "mode": "live"}
            or seal["mode"] != "live"
        ):
            raise ValueError("smoke evidence must be live")
        rows = list(sealed(dest, manifest).values())
        if len(rows) != 5 or {r["task_id"] for r in rows} != set(prereg["tasks"]):
            raise ValueError("need five distinct development calls")
        valid = 0
        for row in rows:
            if row["attempt"] != 1 or row["id"] != f"{row['task_id']}:1":
                raise ValueError("unexpected smoke attempt")
            request = messages(pairs[row["task_id"]]["backdoored"])
            if row["request_sha256"] != digest(
                {"messages": request, "writer": spec, "attempt": 1}
            ):
                raise ValueError("smoke request identity mismatch")
            response = row.get("response")
            if response is None:
                if row["status"] != "infrastructure_failure":
                    raise ValueError("missing smoke response")
                continue
            if response.get("model") != spec["model"] or response.get(
                "provider"
            ) not in (None, "tinker"):
                raise ValueError("smoke model/provider drift")
            checked = validate(response, pairs[row["task_id"]]["backdoored"])
            if any(row.get(k) != v for k, v in checked.items()):
                raise ValueError("smoke validation evidence mismatch")
            valid += checked["status"] == "ok"
        passed = valid >= 4
        if (
            summary["candidate"] != index
            or summary["writer"] != spec
            or summary["calls"] != 5
            or summary["valid"] != valid
            or summary["passed"] != passed
            or summary["receipts_sha256"] != digest(rows)
        ):
            raise ValueError("smoke summary disagrees with receipts")
        if passed != (index == len(summaries) - 1):
            raise ValueError("fallback must stop at first passing candidate")
    if gate["status"] != "passed" or gate["selected_writer"] != summaries[-1]["writer"]:
        raise ValueError("selected writer did not pass")
    return prereg


if __name__ == "__main__":
    main()
