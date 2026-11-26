"""Build image 1.0 as one layer of release code on top of image 0.2.

The layer holds the changed files of ``experiments/release_serving`` at their
image path (``/workspace/gleipnir/...``). Everything else stays 0.2's bytes.
``crane append`` pushes it; no Docker daemon and no GPU are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "ghcr.io/jazhyc/gleipnir-serve:0.2"
BASE_DIGEST = "sha256:ba6025997d875557e372d9ef445b809eb62e7f8fa95f90c08abf5f7b848894cf"
IMAGE_ROOT = "workspace/gleipnir"
FILES = (
    "experiments/release_serving/README.md",
    "experiments/release_serving/serve.py",
    "experiments/release_serving/targets.py",
    "experiments/release_serving/test_release_serving.py",
    "experiments/release_serving/tokenizer_hook/sitecustomize.py",
    "experiments/release_serving/image/identity.py",
    "experiments/release_serving/image/layer10.py",
    "experiments/release_serving/image/pod_fresh_check.sh",
)


def build(output: Path) -> dict:
    """Write a reproducible tarball (fixed owner and times) and describe it."""
    with tarfile.open(output, "w") as archive:
        directory = tarfile.TarInfo(
            f"{IMAGE_ROOT}/experiments/release_serving/tokenizer_hook"
        )
        directory.type, directory.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(directory)
        for relative in FILES:
            data = (ROOT / relative).read_bytes()
            info = tarfile.TarInfo(f"{IMAGE_ROOT}/{relative}")
            info.size = len(data)
            info.mode = 0o755 if relative.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    return {
        "files": {
            f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES
        },
        "layer_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "layer_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="tag to push, e.g. ...:1.0-rc1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crane", default="crane")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    digest = subprocess.run(
        [args.crane, "digest", BASE], check=True, capture_output=True, text=True
    ).stdout.strip()
    if digest != BASE_DIGEST:
        raise SystemExit(f"base image changed: {digest}")
    layer = args.out / "layer10.tar"
    manifest = {"base": BASE, "base_digest": digest, "image": args.image}
    manifest |= build(layer)
    pushed = subprocess.run(
        [
            args.crane,
            "append",
            "-b",
            f"{BASE}@{digest}".replace(":0.2@", "@"),
            "-f",
            str(layer),
            "-t",
            args.image,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest["pushed"] = pushed
    (args.out / "image_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == "__main__":
    main()
