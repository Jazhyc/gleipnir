"""BF16 precision control around the unchanged causal two-logit monitor."""

from __future__ import annotations

import json
from pathlib import Path

CLASSIFIER_KEYS = ("classifier_from_token", "is_causal", "method", "num_labels")


def command(parent: list[str], sources: dict[str, str]) -> list[str]:
    """Keep model, tokenizer, pooling and scheduler; remove low precision."""
    result = parent.copy()
    result[result.index("-m") + 1] = "gleipnir.serving.bf16_server"
    result[result.index("--worker-cls") + 1] = "gleipnir.serving.bf16_worker.Bf16Worker"
    for flag in ("--quantization", "--profiler-config", "--kv-cache-dtype"):
        if flag in result:
            i = result.index(flag)
            del result[i : i + 2]
    i = result.index("--additional-config") + 1
    old = json.loads(result[i])
    result[i] = json.dumps(
        {
            "gleipnir_frost_fp4": sources,
            "monitor_score": old["monitor_score"],
            "runtime_migration": old["runtime_migration"],
            "serving_condition": {
                "attention_precision": "bf16",
                "attention_projection_precision": "bf16",
                "gdn_projection_precision": "bf16",
                "mlp_precision": "bf16",
                "merged_model": result[result.index("--model") + 1],
                "gdn_backend": "flashinfer",
                "quantization": None,
            },
        },
        sort_keys=True,
    )
    result += ["--kv-cache-dtype", "auto"]
    return with_model_text_config(result)


def with_model_text_config(command: list[str]) -> list[str]:
    """Rebuild the override's text config from the served model's own config.

    The selected command embeds the 4B text config plus the classifier keys;
    a different model keeps the classifier keys and its own architecture. For
    the 4B the content is unchanged and the command is returned as is.
    """
    result = command.copy()
    i = result.index("--hf-overrides") + 1
    overrides = json.loads(result[i])
    model = Path(result[result.index("--model") + 1])
    text = json.loads((model / "config.json").read_text())["text_config"]
    rebuilt = {
        **text,
        **{key: overrides["text_config"][key] for key in CLASSIFIER_KEYS},
    }
    if rebuilt != overrides["text_config"]:
        result[i] = json.dumps({**overrides, "text_config": rebuilt})
    return result
