"""Build image 1.2 as one layer on top of image 1.1: an entrypoint with shell access.

The new ``entrypoint.sh`` serves as before for ``4b``/``27b`` (or no
argument), runs any other command as given (``/start.sh``, ``bash``), and
starts SSH in the background when ``PUBLIC_KEY`` is set. The layer holds it at
``/opt/gleipnir-release/entrypoint.sh`` (the image entrypoint) and at its
repository path, plus this builder and the release README. Everything else
stays 1.1's bytes; ``crane append`` pushes it without Docker or a GPU.
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
BASE = "ghcr.io/jazhyc/gleipnir-serve:1.1"
BASE_DIGEST = "sha256:272f4a4dffc9fd1c1779632d43634bf32dfc63bdf00c5e65951dd0d295f27008"
IMAGE_ROOT = "workspace/gleipnir"
ENTRYPOINT = "experiments/release_serving/image/entrypoint.sh"
# (repository file, image path relative to the image root)
FILES = (
    (ENTRYPOINT, "opt/gleipnir-release/entrypoint.sh"),
    *(
        (f, f"{IMAGE_ROOT}/{f}")
        for f in (
            ENTRYPOINT,
            "experiments/release_serving/README.md",
            "experiments/release_serving/image/layer12.py",
        )
    ),
)


def build(output: Path) -> dict:
    """Write a reproducible tarball (fixed owner and times) and describe it."""
    with tarfile.open(output, "w") as archive:
        for relative, target in FILES:
            data = (ROOT / relative).read_bytes()
            info = tarfile.TarInfo(target)
            info.size = len(data)
            info.mode = 0o755 if relative.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    return {
        "files": {
            target: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative, target in FILES
        },
        "layer_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "layer_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="tag to push, e.g. ...:1.2-rc1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crane", default="crane")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    digest = subprocess.run(
        [args.crane, "digest", BASE], check=True, capture_output=True, text=True
    ).stdout.strip()
    if digest != BASE_DIGEST:
        raise SystemExit(f"base image changed: {digest}")
    layer = args.out / "layer12.tar"
    manifest = {"base": BASE, "base_digest": digest, "image": args.image}
    manifest |= build(layer)
    pushed = subprocess.run(
        [
            args.crane,
            "append",
            "-b",
            f"{BASE}@{digest}".replace(":1.1@", "@"),
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
