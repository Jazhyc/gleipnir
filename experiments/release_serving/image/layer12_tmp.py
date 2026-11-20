"""Build image 1.2-rc2: rc1 plus one layer that makes ``/tmp`` mode 1777.

The 1.1 and 1.2-rc1 images ship ``/tmp`` as mode 755, so tools that write
temporary files as another user, such as ``apt-get``, fail inside the
container. The layer holds a ``tmp`` directory entry with mode 1777 (overlay
takes a directory's metadata from its top layer) plus this builder and the
release README. Applied to both ``1.2-rc1`` (gzip) and ``1.2-rc1-zstd``, it
gives two tags with the same uncompressed layers.
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
IMAGE_ROOT = "workspace/gleipnir"
BASES = {
    "gzip": (
        "ghcr.io/jazhyc/gleipnir-serve:1.2-rc1",
        "sha256:7725d8a0107667d5c76c610063dcb9b053e2dec8bd4b7fe718c7ea8ca3e66449",
        "ghcr.io/jazhyc/gleipnir-serve:1.2-rc2",
    ),
    "zstd": (
        "ghcr.io/jazhyc/gleipnir-serve:1.2-rc1-zstd",
        "sha256:06f12fee522e3714bc5d633b08317d2fd219fd994e29c3192e20178d478fe1a9",
        "ghcr.io/jazhyc/gleipnir-serve:1.2-rc2-zstd",
    ),
}
FILES = (
    "experiments/release_serving/README.md",
    "experiments/release_serving/image/layer12_tmp.py",
)


def build(output: Path) -> dict:
    """Write a reproducible tarball (fixed owner and times) and describe it."""
    with tarfile.open(output, "w") as archive:
        tmp = tarfile.TarInfo("tmp")
        tmp.type, tmp.mode = tarfile.DIRTYPE, 0o1777
        archive.addfile(tmp)
        for relative in FILES:
            data = (ROOT / relative).read_bytes()
            info = tarfile.TarInfo(f"{IMAGE_ROOT}/{relative}")
            info.size = len(data)
            info.mode = 0o644
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crane", default="crane")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    layer = args.out / "layer12_tmp.tar"
    manifest = build(layer)
    manifest["pushed"] = {}
    for base, expected, target in BASES.values():
        digest = subprocess.run(
            [args.crane, "digest", base], check=True, capture_output=True, text=True
        ).stdout.strip()
        if digest != expected:
            raise SystemExit(f"base image changed: {base} {digest}")
        repository = base.rsplit(":", 1)[0]
        manifest["pushed"][target] = subprocess.run(
            [
                args.crane,
                "append",
                "-b",
                f"{repository}@{digest}",
                "-f",
                str(layer),
                "-t",
                target,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    (args.out / "image_manifest_rc2.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == "__main__":
    main()
