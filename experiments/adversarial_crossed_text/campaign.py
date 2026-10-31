"""One persistent cap ledger and VM deadline across both study directories."""

import fcntl
import json
import math
import time

from experiments.adversarial_crossed_text.common import ROOT, append, read_jsonl, sha


class Campaign:
    def __init__(self, cfg, out, mock=False):
        self.cfg = cfg
        self.path = out / ".mock-campaign" if mock else ROOT / cfg["campaign"]["output"]
        self.path.mkdir(parents=True, exist_ok=True)
        self.ledger = self.path / "spend.jsonl"

    def latest(self):
        return {r["id"]: r for r in read_jsonl(self.ledger)}

    def reserve(
        self, request_id, api, scope, worst, scope_cap, group=None, group_cap=None
    ):
        with (self.path / ".spend.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = self.latest()
            if request_id in records:
                raise SystemExit(
                    "campaign request already reserved; reconcile before reuse"
                )
            own = sum(r["charge"] for r in records.values() if r["scope"] == scope)
            total = sum(r["charge"] for r in records.values() if r["api"] == api)
            if (
                own + worst > scope_cap
                or total + worst > self.cfg["campaign"]["caps_usd"][api]
            ):
                raise SystemExit(
                    "campaign-wide cap reached, including prior directories"
                )
            grouped = sum(
                r["charge"]
                for r in records.values()
                if group is not None and r.get("group") == group
            )
            if group_cap is not None and grouped + worst > group_cap:
                raise SystemExit("campaign study/API cap reached")
            record = {
                "id": request_id,
                "api": api,
                "scope": scope,
                "charge": worst,
                "status": "reserved",
                "group": group,
            }
            append(self.ledger, record)
            return record

    def settle(self, record, cost):
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            cost = record["charge"]
        with (self.path / ".spend.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            append(self.ledger, {**record, "status": "settled", "charge": cost})

    def check_deadline(self):
        path = self.path / "vm.json"
        if not path.exists():
            raise SystemExit("VM campaign deadline must be recorded before GPU work")
        record = json.loads(path.read_text())
        if time.time() >= record["deadline_epoch"]:
            raise SystemExit("campaign VM deadline reached")
        return record

    def request_id(self, out, stage_id):
        return sha(str(out.resolve()) + ":" + stage_id)
