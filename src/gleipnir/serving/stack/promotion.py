"""Write ``promotion.json`` for the promoted serving runtime.

Development tool: ``python -m gleipnir.serving.stack.promotion MOVES BASE``
reads tab-separated ``recorded path<TAB>live path`` lines from ``MOVES``, takes
each recorded file's bytes from git revision ``BASE`` and writes the manifest
with ``status: candidate``. Regenerating it after an edit to a promoted file
invalidates the acceptance that the manifest records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

from gleipnir.repository import repo_root
from gleipnir.serving.sources import MANIFEST, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moves")
    parser.add_argument("base")
    parser.add_argument("--status", default="candidate")
    args = parser.parse_args()
    root = repo_root(__file__)
    triples = []
    for line in open(args.moves).read().splitlines():
        old, new = line.split("\t")
        blob = subprocess.run(
            ["git", "show", f"{args.base}:{old}"],
            capture_output=True,
            check=True,
            cwd=root,
        ).stdout
        triples.append((old, new, hashlib.sha256(blob).hexdigest()))
    manifest = build_manifest(root, triples, args.status)
    manifest["base_revision"] = subprocess.run(
        ["git", "rev-parse", args.base], capture_output=True, text=True, cwd=root
    ).stdout.strip()
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
