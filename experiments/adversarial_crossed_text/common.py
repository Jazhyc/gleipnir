"""Frozen inputs, run identity and durable API reservations."""

from __future__ import annotations

import contextlib
import fcntl
import json
import subprocess
import threading
import time
from pathlib import Path

from experiments.adversarial_code_or_comments.common import (
    ROOT as ROOT,
)
from experiments.adversarial_code_or_comments.common import (
    instruction as instruction,
)
from experiments.adversarial_code_or_comments.common import (
    load_pairs as load_pairs,
)
from experiments.adversarial_code_or_comments.common import (
    order_key as order_key,
)
from experiments.adversarial_code_or_comments.common import (
    read_jsonl as read_jsonl,
)
from experiments.adversarial_code_or_comments.common import (
    sha as sha,
)
from experiments.adversarial_code_or_comments.common import (
    write_json as write_json,
)
from experiments.adversarial_code_or_comments.common import (
    write_jsonl as write_jsonl,
)
from experiments.adversarial_code_or_comments.llm import Client, endpoint
from gleipnir.adversarial.attacker import AttackerError, BudgetExceeded

HERE = Path(__file__).resolve().parent


def config():
    return json.loads((HERE / "config.json").read_text())


def digest(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output(path=None):
    out = Path(path) if path else ROOT / config()["output"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def append(path, record):
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


@contextlib.contextmanager
def exclusive(out, stage):
    with (out / f".{stage}.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{stage} already has a live runner") from None
        yield


def clean(paths):
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    ).strip()


def frozen(cfg, *, heldout=False, out=None):
    paths = [
        str(HERE.relative_to(ROOT)),
        "src/gleipnir/adversarial",
        "experiments/adversarial_code_or_comments",
    ]
    if not cfg.get("execution_enabled", False):
        raise SystemExit("execution disabled by user; implementation-only scope")
    if cfg["protocol_status"] != "frozen" or not clean(paths):
        raise SystemExit(
            "commit the frozen protocol and implementation before paid/GPU work"
        )
    if heldout:
        path = cfg["analysis"]["predictions_file"]
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True
        )
        if tracked.returncode or not clean([path]):
            raise SystemExit("commit predictions before held-out scoring")
        pred = json.loads((ROOT / path).read_text())
        if pred.get("dry_run") is not False:
            raise SystemExit("mock/unproven predictions cannot unlock live scoring")
        if out and pred.get("variants_manifest_sha256") != validate_variants(
            out, cfg, False
        ):
            raise SystemExit("predictions refer to another variant provenance chain")
        if out and pred["variants_sha256"] != digest(out / "variants.jsonl"):
            raise SystemExit("variants changed after predictions were frozen")


def prior(cfg):
    root = ROOT / cfg["prior"]["directory"]
    for name in ("selection", "triggers"):
        path = root / (name + (".json" if name == "selection" else ".jsonl"))
        if digest(path) != cfg["prior"][name + "_sha256"]:
            raise ValueError(f"prior {name} identity mismatch")
    selection = json.loads((root / "selection.json").read_text())
    triggers = {r["task_id"]: r for r in read_jsonl(root / "triggers.jsonl")}
    return selection, triggers


def bind(out, cfg, stage, inputs):
    """Reject stale resumes, including mock/live cache reuse."""
    identity = {
        "config_sha256": sha(json.dumps(cfg, sort_keys=True)),
        "sources": sources(),
        **inputs,
    }
    path = out / f"{stage}.identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise SystemExit(f"{stage} inputs/settings changed; use a new output directory")
    write_json(path, identity)


def sources():
    paths = list(HERE.glob("*.py")) + list((HERE / "prompts").glob("*.txt"))
    paths += list((ROOT / "src/gleipnir/adversarial").glob("*.py"))
    paths += [
        ROOT / "experiments/adversarial_code_or_comments" / (name + ".py")
        for name in (
            "common",
            "llm",
            "texts",
            "variants",
            "score",
            "analyze",
            "trigger_runner",
        )
    ]
    return {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(paths)
        if not p.name.startswith("test_")
    }


def validate_pipeline(out, cfg, mock):
    path = out / "pipeline.identity.json"
    if not path.exists():
        raise SystemExit("pipeline provenance is missing")
    expected = {
        "config_sha256": sha(json.dumps(cfg, sort_keys=True)),
        "sources": sources(),
        "dry_run": mock,
        "selection_sha256": digest(out / "selection.json"),
    }
    if json.loads(path.read_text()) != expected:
        raise SystemExit("pipeline provenance/settings/source/mock identity mismatch")


def seal_variants(out, cfg, mock):
    validate_pipeline(out, cfg, mock)
    if any(r.get("dry_run") != mock for r in read_jsonl(out / "calls.jsonl")):
        raise SystemExit("API call provenance disagrees with pipeline mode")
    files = (
        "selection.json",
        "calls.jsonl",
        "trigger_audit.jsonl",
        "trigger_hand_reads.jsonl",
        "texts.jsonl",
        "text_checks.jsonl",
        "hand_audit_sample.jsonl",
        "hand_audit.jsonl",
        "variant_failures.jsonl",
        "variants.jsonl",
    )
    manifest = {
        "dry_run": mock,
        "sources": sources(),
        "config_sha256": sha(json.dumps(cfg, sort_keys=True)),
        "files": {
            name: digest(out / name) if (out / name).exists() else None
            for name in files
        },
    }
    write_json(out / "variants.manifest.json", manifest)


def validate_variants(out, cfg, mock):
    validate_pipeline(out, cfg, mock)
    path = out / "variants.manifest.json"
    if not path.exists():
        raise SystemExit("sealed variant manifest is missing")
    manifest = json.loads(path.read_text())
    if manifest["dry_run"] != mock or manifest["sources"] != sources():
        raise SystemExit("variant mock/source identity mismatch")
    if manifest["config_sha256"] != sha(json.dumps(cfg, sort_keys=True)):
        raise SystemExit("variant configuration changed")
    for name, expected in manifest["files"].items():
        current = digest(out / name) if (out / name).exists() else None
        if current != expected:
            raise SystemExit(f"variant upstream artifact changed: {name}")
    return digest(path)


def prompt(name, cfg):
    path = HERE / "prompts" / (name + ".txt")
    if digest(path) != cfg["prompts_sha256"][name]:
        raise ValueError(f"{name} prompt hash mismatch")
    return path.read_text()


def spent(out, role):
    latest = {}
    for r in read_jsonl(out / "calls.jsonl"):
        if r["role"] == role:
            latest[r["request_id"]] = r
    return sum(r["charged_usd"] for r in latest.values())


class Calls:
    """Reserve on disk BEFORE dispatch. Interrupted requests are not replayed."""

    def __init__(self, out, cfg, role, dry_run=False, responder=None):
        if not dry_run:
            frozen(cfg)
        self.out, self.cfg, self.role = out, cfg, role
        self.dry_run = dry_run
        self.lock = threading.Lock()
        self.stop = threading.Event()
        from experiments.adversarial_crossed_text.campaign import Campaign

        self.campaign = Campaign(cfg, out, dry_run)
        self.client = Client(
            cfg["models"][role],
            cfg["budget_usd"][role],
            cfg,
            responder=responder if dry_run else None,
        )
        self.client.budget.spent_usd = spent(out, role)
        self.history = {r["request_id"]: r for r in read_jsonl(out / "calls.jsonl")}
        post = self.client.post

        def refusal_aware(body, key, timeout):
            data = post(body, key, timeout)
            for choice in data.get("choices", []):
                if (choice.get("message") or {}).get("refusal"):
                    choice["finish_reason"] = "refusal"
            return data

        self.client.post = refusal_aware

    def reserve(self, request_id, messages):
        if self.stop.is_set():
            raise SystemExit("dispatch stopped after an authentication failure")
        identity = sha(json.dumps(messages, sort_keys=True))
        if request_id in self.history:
            old = self.history[request_id]
            if old["prompt_sha256"] != identity:
                raise ValueError("request id reused for another prompt")
            return old, True
        worst = endpoint(self.cfg["models"][self.role]).worst_case_usd(
            sum(len(m["content"].encode()) for m in messages)
        )
        if spent(self.out, self.role) + worst > self.cfg["budget_usd"][self.role]:
            raise BudgetExceeded("durable role cap reached")
        record = {
            "request_id": request_id,
            "role": self.role,
            "prompt_sha256": identity,
            "status": "reserved",
            "charged_usd": 0.0 if self.dry_run else worst,
            "time": time.time(),
            "dry_run": self.dry_run,
        }
        campaign_id = self.campaign.request_id(self.out, request_id)
        campaign = self.campaign.reserve(
            campaign_id,
            "tinker",
            "crossed-" + self.role,
            record["charged_usd"],
            self.cfg["budget_usd"][self.role],
        )
        record["campaign_reservation"] = campaign
        append(self.out / "calls.jsonl", record)
        self.history[request_id] = record
        return record, False

    def chat(self, request_id, messages, attempt=1):
        with self.lock:
            record, reused = self.reserve(request_id, messages)
        if reused:
            return record.get("result", {"status": "interrupted_unknown"})
        worst = record["charged_usd"]
        try:
            result = {"status": "ok", **self.client.chat(messages, attempt)}
            charge = result["cost_usd"]
        except AttackerError as error:
            result = {
                "status": "infrastructure_failure",
                "error": str(error),
                "details": error.details,
            }
            charge = 0.0 if self.dry_run else worst
            if error.details.get("http_status") in (401, 403):
                self.stop.set()
        final = {**record, "status": "settled", "charged_usd": charge, "result": result}
        with self.lock:
            append(self.out / "calls.jsonl", final)
            self.history[request_id] = final
            self.campaign.settle(record["campaign_reservation"], charge)
        return result
