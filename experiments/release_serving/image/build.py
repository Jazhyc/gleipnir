"""Assemble and push the serving image without a Docker daemon (build Pod).

Each layer is a tarball of absolute paths from the build Pod, written with
their original locations so recorded receipts, cache keys and interpreter
paths stay valid. ``crane append`` adds the layers to the pinned base image
and ``crane mutate`` sets the entrypoint. The layer manifest (paths, sizes and
SHA-256 of each tarball) is written next to the tarballs for the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE = "runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster"
ENTRYPOINT = "/opt/gleipnir-release/entrypoint.sh"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def layer(name: str, paths: list[str], out: Path) -> dict:
    """Tar absolute ``paths`` (files or trees) relative to ``/``."""
    listing = out / f"{name}.list"
    listing.write_text("\0".join(p.lstrip("/") for p in paths) + "\0")
    tarball = out / f"{name}.tar"
    subprocess.run(
        ["tar", "-C", "/", "--null", "-T", str(listing), "-cf", str(tarball)],
        check=True,
    )
    return {
        "name": name,
        "entries": len(paths),
        "bytes": tarball.stat().st_size,
        "sha256": digest(tarball),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON {layer name: [absolute paths]}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--image", required=True, help="e.g. ghcr.io/jazhyc/gleipnir-serve:0.1"
    )
    parser.add_argument("--crane", default="crane")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.spec.read_text())
    layers = [layer(name, paths, args.out) for name, paths in spec.items()]
    staged = f"{args.image}-staged"
    command = [args.crane, "append", "-b", BASE, "-t", staged]
    for item in layers:
        command += ["-f", str(args.out / f"{item['name']}.tar")]
    subprocess.run(command, check=True)
    result = subprocess.run(
        [
            args.crane,
            "mutate",
            staged,
            "-t",
            args.image,
            "--entrypoint",
            ENTRYPOINT,
            "--cmd",
            "4b",
            "--workdir",
            "/workspace/gleipnir",
            "--exposed-ports",
            "8000/tcp",
            "--label",
            "org.opencontainers.image.source=https://github.com/Jazhyc/gleipnir",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = {
        "base": BASE,
        "image": args.image,
        "pushed_digest": result.stdout.strip(),
        "layers": layers,
    }
    (args.out / "image_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == "__main__":
    main()
