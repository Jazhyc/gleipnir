"""Download public inputs at one immutable Hub revision, never using credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/goodfire"
REPO = "Goodfire/reward-hack-data"
REVISION = "bd8fb473568b8ef1d12fc2325a5c8af939c15ad9"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wanted(path: str, metadata_only: bool = False) -> bool:
    if path.endswith("README.md"):
        return True  # Read every directory card, including the excluded DeepSWE.
    if path.startswith(("deepswe-", "synthetic_data/")):
        return False
    if "/sources/" in path or path == ".gitattributes":
        return False
    return not metadata_only or not path.endswith("traces.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000"
    with urllib.request.urlopen(url, timeout=120) as response:
        if response.headers.get("Link"):
            raise ValueError(
                "Hub tree is paginated; update downloader before proceeding"
            )
        tree = json.load(response)
    files = [
        x for x in tree if x["type"] == "file" and wanted(x["path"], args.metadata_only)
    ]

    def fetch(item: dict) -> dict:
        path = DATA / "raw" / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        lfs = item.get("lfs", {}).get("oid")

        def verified(candidate: Path) -> bool:
            if not candidate.exists() or candidate.stat().st_size != item["size"]:
                return False
            if lfs:
                return sha256(candidate) == lfs
            blob = hashlib.sha1(usedforsecurity=False)
            blob.update(f"blob {item['size']}\0".encode())
            with candidate.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    blob.update(block)
            return blob.hexdigest() == item["oid"]

        if not verified(path):
            tmp = path.with_suffix(path.suffix + ".partial")
            with (
                urllib.request.urlopen(
                    f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['path']}",
                    timeout=300,
                ) as response,
                tmp.open("wb") as f,
            ):
                shutil.copyfileobj(response, f)
            if not verified(tmp):
                raise ValueError(f"Hub object identity mismatch: {item['path']}")
            tmp.replace(path)
        return {
            "path": item["path"],
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "hub_git_oid": item["oid"],
            "hub_lfs_sha256": lfs,
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(fetch, files))
    receipt = {
        "repo": REPO,
        "revision": REVISION,
        "license": None,
        "permission_note": (
            "No dataset license stated. Abhinav authorized sprint experiments; "
            "no redistribution."
        ),
        "metadata_only": args.metadata_only,
        "files": receipts,
    }
    (DATA / "download_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files": len(receipts),
                "bytes": sum(r["bytes"] for r in receipts),
                "revision": REVISION,
            }
        )
    )


if __name__ == "__main__":
    main()
