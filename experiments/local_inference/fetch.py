"""Stage pinned public inputs and record the resolved release revision."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from experiments.tool_trajectory_monitoring import (
    prepare_qwen_reasoning_id_benchmark as source,
)
from gleipnir.qwen35_adapter_rebase import sha256_file

ROOT = Path("results/local_inference")
CONFIG = Path("experiments/local_inference/config.json")


def main() -> None:
    config = json.loads(CONFIG.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = ROOT / "downloads.json"
    if identity.exists():
        info = json.loads(identity.read_text())
    else:
        info = {"adapter_revision": HfApi().model_info(config["adapter_id"]).sha}
    identity.write_text(json.dumps(info, indent=2) + "\n")

    def model(which: str) -> None:
        revision = (
            config["base_revision"] if which == "base" else info["adapter_revision"]
        )
        snapshot_download(
            config[f"{which}_id"],
            revision=revision,
            local_dir=ROOT / which,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
        )
        print(f"downloaded {which} at {revision}", flush=True)

    def dataset(which: str) -> None:
        upper = which.upper()
        path = hf_hub_download(
            getattr(source, f"{upper}_REPO"),
            getattr(source, f"{upper}_FILE"),
            revision=getattr(source, f"{upper}_REVISION"),
            repo_type="dataset",
        )
        target = Path("data/tool_trajectory_monitoring/source/id_evaluation") / (
            "stride_test.parquet" if which == "stride" else "gloom_exfiltration.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if sha256_file(Path(path)) != getattr(source, f"{upper}_SHA256"):
            raise ValueError(f"{which} source checksum mismatch")
        shutil.copyfile(path, target)
        print(f"verified {which} source", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(model, x) for x in ("base", "adapter")]
        futures += [pool.submit(dataset, x) for x in ("stride", "gloom")]
        for future in futures:
            future.result()
    snapshot_download(
        source.TOKENIZER_ID,
        revision=source.TOKENIZER_REVISION,
        allow_patterns=["*.json", "*.txt", "*.jinja", "*.model"],
    )


if __name__ == "__main__":
    main()
