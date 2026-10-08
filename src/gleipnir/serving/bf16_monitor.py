"""BF16 precision control around the unchanged causal two-logit monitor."""

from __future__ import annotations

import json


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
    return result
