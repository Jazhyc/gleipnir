"""Install hash-locked optional Lens wheels without changing the serving lock."""

import hashlib
import json
import subprocess
from pathlib import Path

from gleipnir.serving.runtime import copy_dependency_tree


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    requirements = Path(__file__).with_name("requirements.txt")
    source = root / ".cache/interpretability/vllm-lens-1.3.0"
    target = Path("/tmp/gleipnir-vllm-lens-1.3.0")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(root / ".venv-vllm031/bin/python"),
            "--no-deps",
            "--require-hashes",
            "--link-mode",
            "copy",
            "--target",
            str(source),
            "-r",
            str(requirements),
        ],
        check=True,
    )
    copy_dependency_tree(source, target)
    files = {}
    for path in source.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            relative = path.relative_to(source)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashlib.sha256((target / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Lens overlay staging drift: {relative}")
            files[str(relative)] = digest
    receipt = {
        "lens_version": "1.3.0",
        "vllm_override": "0.31.0",
        "upstream_requires_vllm": "0.30.0",
        "generation_plugin_disabled": True,
        "source": str(source),
        "target": str(target),
        "files_sha256": files,
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    }
    out = root / "results/b200_vllm_lens/staging.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print("lens_staged", len(files), target, flush=True)


if __name__ == "__main__":
    main()
