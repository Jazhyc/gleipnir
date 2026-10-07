"""Native Gigatoken encoding for the validated text-only Qwen serving frontend."""

from __future__ import annotations

import importlib.metadata
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

VERSION = "0.10.0"
ENTRYPOINT = "experiments.b200_inference_benchmark.frontend_server"


class FrontendControl:
    """Opt-in experiment control; switch only between drained request passes."""

    def __init__(self) -> None:
        self.mode = "native"
        self.generation = 0
        self.active = 0
        self.calls = {"hf": 0, "native": 0}
        self.seconds = {"hf": 0.0, "native": 0.0}
        self.lock = threading.Lock()

    def toggle(self) -> dict:
        with self.lock:
            if self.active:
                raise RuntimeError("drain active encodes before switching frontend")
            self.mode = "hf" if self.mode == "native" else "native"
            self.generation += 1
            return self._snapshot()

    def _snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "generation": self.generation,
            "active_encodes": self.active,
            "calls": dict(self.calls),
            "encode_seconds": dict(self.seconds),
        }

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def encode(self, native: Any, hf: Any, text: str, **kwargs) -> list[int]:
        with self.lock:
            mode = self.mode
            self.active += 1
        before = time.perf_counter()
        try:
            return (native if mode == "native" else hf).encode(text, **kwargs)
        finally:
            elapsed = time.perf_counter() - before
            with self.lock:
                self.active -= 1
                self.calls[mode] += 1
                self.seconds[mode] += elapsed


class NativeEncoder:
    """Replace encoding only; retain the original HF tokenizer for other work."""

    def __init__(self, tokenizer: Any, *, factory: Any = None) -> None:
        configuration = json.loads(tokenizer.backend_tokenizer.to_str())
        processor = configuration.get("post_processor")
        if processor is not None and processor.get("type") != "ByteLevel":
            raise ValueError("native encoder requires a token-neutral postprocessor")
        self.truncation_side = tokenizer.truncation_side
        if self.truncation_side not in {"left", "right"}:
            raise ValueError("unsupported truncation side")
        if factory is None:
            import gigatoken

            factory = gigatoken.Tokenizer
        self.backend = factory(tokenizer)
        self.calls = 0

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("native serving encoder requires text")
        if type(add_special_tokens) is not bool or type(truncation) is not bool:
            raise ValueError("unsupported tokenization flags")
        if max_length is not None and (type(max_length) is not int or max_length < 0):
            raise ValueError("invalid tokenization length")
        if truncation and max_length is None:
            raise ValueError("truncation requires an explicit length")
        ids = self.backend.encode(text)
        if truncation and len(ids) > max_length:
            ids = (
                ids[:0]
                if max_length == 0
                else (
                    ids[-max_length:]
                    if self.truncation_side == "left"
                    else ids[:max_length]
                )
            )
        self.calls += 1
        return ids.tolist()


def configure_frontend(
    root: Path, frontend: dict, command: list[str], env: dict
) -> dict:
    """Bind CPU-only startup separately from the unchanged GPU compilation key."""
    from gleipnir.serving_runtime import sha

    if frontend["backend"] != "gigatoken_native":
        raise ValueError("unsupported frontend backend")
    validation = root / frontend["validation"]
    receipt = json.loads(validation.read_text())
    if receipt.get("exact") is not True or receipt.get("version") != VERSION:
        raise ValueError("native Gigatoken validation failed")
    source = root / "src/gleipnir/serving/gigatoken.py"
    if receipt["source_sha256"] != sha(source):
        raise ValueError("native Gigatoken validation source drift")
    package = Path(frontend["package_path"])
    if not (package / "gigatoken").is_dir():
        raise ValueError("isolated Gigatoken package missing")
    for path, digest in receipt["package_files"].items():
        if sha(package / path) != digest:
            raise ValueError("native Gigatoken package drift")
    index = command.index("-m") + 1
    if command[index] != "experiments.b200_attention_gdn_serving.server":
        raise ValueError("frontend wrapper requires the validated GPU registry loader")
    command[index] = ENTRYPOINT
    env["PYTHONPATH"] = f"{package}:{env['PYTHONPATH']}"
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(frontend["receipt_path"])
    if frontend.get("ab_control"):
        env["GLEIPNIR_GIGATOKEN_AB"] = "1"
    else:
        env.pop("GLEIPNIR_GIGATOKEN_AB", None)
    return {
        **frontend,
        "version": VERSION,
        "validation_sha256": sha(validation),
        "source_sha256": sha(source),
        "package_files": receipt["package_files"],
        "entrypoint_sha256": sha(
            root / "experiments/b200_inference_benchmark/frontend_server.py"
        ),
        "gpu_registry_module": "experiments.b200_attention_gdn_serving.server",
    }


def install_native_encoder() -> None:
    """Install before constructing renderers; keep vLLM's existing thread pool."""
    from vllm.renderers.base import BaseRenderer

    if importlib.metadata.version("gigatoken") != VERSION:
        raise ValueError("native frontend Gigatoken version drift")
    original_init = BaseRenderer.__init__
    receipt_path = Path(os.environ["GLEIPNIR_GIGATOKEN_RECEIPT"])
    control = (
        FrontendControl() if os.environ.get("GLEIPNIR_GIGATOKEN_AB") == "1" else None
    )
    receipt: dict = {}

    def publish() -> None:
        value = {**receipt, **({"ab_control": control.snapshot()} if control else {})}
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(receipt_path)

    if control is not None:

        def switch(signum, frame):
            try:
                control.toggle()
            except RuntimeError:
                return
            publish()

        signal.signal(signal.SIGUSR1, switch)

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._gleipnir_encoder = (
            NativeEncoder(self.tokenizer) if self.tokenizer else None
        )
        if self._gleipnir_encoder is not None:
            receipt.update(
                {
                    "backend": "gigatoken_native",
                    "version": VERSION,
                    "pid": os.getpid(),
                    "is_hf_compat": False,
                    "renderer_workers": self.model_config.renderer_num_workers,
                    "truncation_side": self._gleipnir_encoder.truncation_side,
                    "original_tokenizer_class": type(self.tokenizer).__name__,
                    "metadata_and_decode": "original HF tokenizer",
                }
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            publish()
            print("native_gigatoken_frontend_initialized", receipt, flush=True)

    def encode(self, text, **kwargs):
        if self._gleipnir_encoder is None:
            raise ValueError("native frontend tokenizer is unavailable")
        if control is not None:
            return control.encode(
                self._gleipnir_encoder, self.tokenizer, text, **kwargs
            )
        return self._gleipnir_encoder.encode(text, **kwargs)

    BaseRenderer._encode = encode
    BaseRenderer.__init__ = initialize
