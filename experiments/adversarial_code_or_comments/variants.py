"""Scoring rows: each text inserted as a header docstring, checked for harm.

Builds the 2 x 4 main grid for both splits, the repetition levels on the
repetition subset, and the unedited calibration rows. Every edited program
must (a) equal the original with exactly the inserted header added, (b) pass
`check_documentation_only_edit` (comments, docstrings and blank lines stripped,
byte-identical code and equal ASTs) and (c) carry the intended docstring
value. Variants that fail, or whose text failed generation, fidelity or hand
audit, are excluded and counted in `variant_failures.jsonl`.

    PYTHONPATH=src python -m experiments.adversarial_code_or_comments.variants \
        [--dry-run --out DIR]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from collections import Counter
from typing import Any

from experiments.adversarial_code_or_comments.common import (
    CODES,
    config,
    load_pairs,
    output_dir,
    read_jsonl,
    sha,
    write_json,
    write_jsonl,
)
from experiments.adversarial_code_or_comments.texts import (
    REFERENCE,
    final_texts,
    length_band,
    pick_donor,
    with_cover_story_reads,
)
from gleipnir.adversarial import apps
from gleipnir.adversarial.contract import join_prompt, split_prompt
from gleipnir.adversarial.python_code import Check
from gleipnir.adversarial.python_docs import check_documentation_only_edit

LEADING = re.compile(r"^(#!.*\n)?([ \t\f]*#.*coding[:=][ \t]*[-\w.]+.*\n)?")


def docstring(text: str, width: int, repeats: int = 1) -> tuple[str, str]:
    """(source of the header, value its docstring must have)."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip())]
    body = "\n\n".join(textwrap.fill(p, width) for p in paragraphs if p)
    value = "\n" + "\n\n".join([body] * repeats) + "\n"
    escaped = value.replace("\\", "\\\\").replace('"""', '""\\"')
    return f'"""{escaped}"""\n\n', value


def insertion_offset(code: str) -> int:
    """After a shebang or encoding cookie, which must stay on lines 1-2."""
    return LEADING.match(code).end()


def insert_header(code: str, header: str) -> tuple[str, int]:
    at = insertion_offset(code)
    return code[:at] + header + code[at:], at


def check_variant(
    original: str, edited: str, header: str, at: int, value: str
) -> Check:
    """The harm-preserved check for one header insertion."""
    if edited[:at] + edited[at + len(header) :] != original:
        return Check(False, "removing the header does not recover the original")
    if edited[at : at + len(header)] != header:
        return Check(False, "header not where it was inserted")
    result = check_documentation_only_edit(original, edited)
    if not result.ok:
        return result
    if ast.get_docstring(ast.parse(edited), clean=False) != value:
        return Check(False, "module docstring is not the intended text")
    return Check(True, "header docstring only")


def edited_prompt(row: dict[str, Any], code: str) -> str:
    """The row with new code; head, problem and framing checked byte for byte."""
    new = apps.with_code(row["prompt"], code)
    before, after = apps.parse(row["prompt"]), apps.parse(new)
    if (after.head, after.problem, after.code) != (before.head, before.problem, code):
        raise ValueError("prompt contract changed")
    if join_prompt(*split_prompt(new)) != new:
        raise ValueError("prompt does not round-trip")
    return new


def eligible_texts(
    texts: list[dict[str, Any]],
    fidelity: list[dict[str, Any]],
    audit: list[dict[str, Any]] | None,
    cfg: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
    """Usable texts by (task, kind), and why each other text is unusable."""
    checked = set(cfg["texts"]["fidelity"]["checked_kinds"])
    verdicts = {r["text_id"]: r["status"] for r in fidelity}
    failed_by_hand = {
        r["text_id"] for r in audit or [] if r.get("verdict") not in (None, "pass")
    }
    finals = final_texts(texts)
    usable, reasons = {}, {}
    for key, record in finals.items():
        reference = finals.get((key[0], REFERENCE), {})
        band = (
            length_band(reference["chars"], cfg["texts"]["tolerance"])
            if reference.get("status") == "ok"
            else None
        )
        if record["status"] != "ok":
            reasons[key] = f"text_{record['status']}"
        elif key[1] != REFERENCE and not (
            band and band[0] <= record["chars"] <= band[1]
        ):
            reasons[key] = "length_out_of_tolerance"
        elif key[1] in checked and verdicts.get(record["id"]) != "pass":
            reasons[key] = f"fidelity_{verdicts.get(record['id'], 'unchecked')}"
        elif record["id"] in failed_by_hand:
            reasons[key] = "hand_audit_fail"
        else:
            usable[key] = record
    return usable, reasons


def donors(selection, texts, usable, cfg) -> dict[str, tuple[str | None, str]]:
    """Task -> (donor task or None, reason) for the unrelated text, per split."""
    u = cfg["texts"]["unrelated"]
    references = {k[0]: r for k, r in final_texts(texts).items() if k[1] == REFERENCE}
    out = {}
    for split in ("development", "heldout"):
        candidates = {
            t: usable[(t, REFERENCE)]["chars"]
            for t in selection[split]
            if (t, REFERENCE) in usable
        }
        for task in selection[split]:
            ref = references.get(task)
            if not ref or ref["status"] != "ok":
                out[task] = (None, "no_reference")
                continue
            donor = pick_donor(
                task, ref["chars"], candidates, cfg["texts"]["tolerance"], u["seed"]
            )
            out[task] = (donor, "ok" if donor else "no_donor_in_tolerance")
    return out


def build(selection, pairs, texts, fidelity, audit, cfg):
    """(rows, failures) for calibration, main grid and repetition sets."""
    g, rep = cfg["grid"], cfg["repetition"]
    usable, reasons = eligible_texts(texts, fidelity, audit, cfg)
    donor_of = donors(selection, texts, usable, cfg)
    rows, failures = [], []
    held_pool = set(selection["heldout_pool"])
    for task in sorted(pairs, key=int):
        if task not in held_pool:
            row = pairs[task]["honest"]
            rows.append(
                {
                    "id": f"calibration:{task}",
                    "prompt": row["prompt"],
                    "metadata": {
                        "ground_truth": 0,
                        "condition": "calibration",
                        "task_id": task,
                        "parent_id": str(row["id"]),
                        "split": "calibration",
                        "set": "calibration",
                        "code": "honest",
                        "text": "none",
                        "repetition": 0,
                        "added_chars": 0,
                    },
                }
            )
    plan = []
    for split in ("development", "heldout"):
        for task in selection[split]:
            for code in CODES:
                for text in g["texts"]:
                    plan.append((split, "main", task, code, text, 1))
    for task in selection["repetition"]:
        for code in CODES:
            for text in rep["texts"]:
                for k in rep["levels"]:
                    if k != 1:
                        plan.append(("development", "repetition", task, code, text, k))
    for split, set_name, task, code, text, k in plan:
        row = pairs[task][code]
        original = apps.parse(row["prompt"]).code
        rid = f"{task}:{code}:{text}" + (f":r{k}" if k != 1 else "")
        meta = {
            "ground_truth": int(code == "backdoored"),
            "condition": f"{code}:{text}",
            "task_id": task,
            "parent_id": str(row["id"]),
            "split": split,
            "set": set_name,
            "code": code,
            "text": text,
            "repetition": k if text != "none" else 0,
            "original_code_sha256": sha(original),
        }

        def fail(reason, rid=rid, meta=meta):
            failures.append({"id": rid, **meta, "reason": reason})

        if text == "none":
            rows.append(
                {
                    "id": rid,
                    "prompt": row["prompt"],
                    "metadata": {**meta, "added_chars": 0},
                }
            )
            continue
        if text == "unrelated":
            donor, why = donor_of[task]
            if donor is None:
                fail(why)
                continue
            record = usable[(donor, REFERENCE)]
            meta["donor_task_id"] = donor
        else:
            kind = g["text_kind"][code][text]
            if (task, kind) not in usable:
                fail(reasons.get((task, kind), "text_missing"))
                continue
            record = usable[(task, kind)]
        header, value = docstring(record["text"], g["wrap_width"], k)
        edited, at = insert_header(original, header)
        check = check_variant(original, edited, header, at, value)
        if not check.ok:
            fail(f"harm_check: {check.reason}")
            continue
        try:
            new_prompt = edited_prompt(row, edited)
        except ValueError as error:
            fail(f"contract: {error}")
            continue
        rows.append(
            {
                "id": rid,
                "prompt": new_prompt,
                "metadata": {
                    **meta,
                    "text_id": record["id"],
                    "text_chars": record["chars"],
                    "added_chars": len(header),
                    "code_sha256": sha(edited),
                },
            }
        )
    return rows, failures


def summarise(rows, failures) -> dict[str, Any]:
    def key(m):
        return f"{m['set']}/{m['split']}/{m['code']}/{m['text']}" + (
            f"/r{m['repetition']}" if m["set"] == "repetition" else ""
        )

    return {
        "rows": dict(sorted(Counter(key(r["metadata"]) for r in rows).items())),
        "excluded": dict(sorted(Counter(key(f) for f in failures).items())),
        "excluded_by_reason": dict(Counter(f["reason"] for f in failures)),
        "total_rows": len(rows),
        "total_excluded": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="output folder (default: config output)")
    parser.add_argument(
        "--dry-run", action="store_true", help="allow a missing hand audit"
    )
    args = parser.parse_args()
    cfg = config()
    out = output_dir(args.out, cfg)
    selection = json.loads((out / "selection.json").read_text())
    if not selection.get("final"):
        raise SystemExit("selection.json is provisional; verify triggers first")
    audit_path = out / "hand_audit.jsonl"
    if not audit_path.exists() and not args.dry_run:
        raise SystemExit("hand_audit.jsonl missing: read the 40-item sample first")
    audit = read_jsonl(audit_path)
    if any(r.get("verdict") is None for r in audit):
        raise SystemExit("hand_audit.jsonl has items without a verdict")
    pairs, _ = load_pairs(cfg)
    rows, failures = build(
        selection,
        pairs,
        read_jsonl(out / "texts.jsonl"),
        with_cover_story_reads(
            read_jsonl(out / "fidelity.jsonl"),
            read_jsonl(out / "cover_story_reads.jsonl"),
        ),
        audit,
        cfg,
    )
    write_jsonl(out / "variants.jsonl", rows)
    write_jsonl(out / "variant_failures.jsonl", failures)
    summary = summarise(rows, failures)
    write_json(out / "variants_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
