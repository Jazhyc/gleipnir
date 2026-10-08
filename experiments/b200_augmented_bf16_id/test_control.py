"""BF16 command intervention preserves the fixed monitor workload and boundary."""

import json

from gleipnir.serving.bf16_monitor import command


def test_bf16_command_removes_all_low_precision_and_retains_scoring_contract():
    # Build a representative command without requiring ignored receipts in CI.
    parent = [
        "python",
        "-m",
        "original.server",
        "--model",
        "/tmp/merged",
        "--worker-cls",
        "original.Worker",
        "--quantization",
        "fp4",
        "--runner",
        "pooling",
        "--convert",
        "classify",
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "32768",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--hf-overrides",
        '{"is_causal":true}',
        "--pooler-config",
        '{"pooling_type":"LAST"}',
        "--attention-backend",
        "FLASHINFER",
        "--scheduler-cls",
        "same.Scheduler",
        "--additional-config",
        json.dumps(
            {
                "monitor_score": {"token_ids": [15, 16]},
                "runtime_migration": {"vllm": "0.31.0"},
            }
        ),
    ]
    observed = command(parent, {"source.py": "sha"})
    assert parent[parent.index("--quantization") + 1] == "fp4"
    assert "--quantization" not in observed
    for flag in (
        "--model",
        "--dtype",
        "--runner",
        "--hf-overrides",
        "--pooler-config",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--scheduler-cls",
    ):
        assert observed[observed.index(flag) + 1] == parent[parent.index(flag) + 1]
    additional = json.loads(observed[observed.index("--additional-config") + 1])
    condition = additional["serving_condition"]
    assert condition["quantization"] is None
    assert all(
        condition[k] == "bf16"
        for k in (
            "attention_precision",
            "attention_projection_precision",
            "gdn_projection_precision",
            "mlp_precision",
        )
    )
    assert observed[observed.index("--kv-cache-dtype") + 1] == "auto"
    assert observed[observed.index("--worker-cls") + 1].endswith("Bf16Worker")
