import json
from pathlib import Path

import pytest

from experiments.release_serving import fetch
from experiments.release_serving.targets import ROOT, TARGETS


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_recorded_command_serves_the_recorded_merge(key: str) -> None:
    target = TARGETS[key]
    command = json.loads((ROOT / target.server_receipt).read_text())["command"]
    assert command[command.index("--model") + 1] == str(target.merged_path)
    assert {"--host", "--port", "--additional-config"} <= set(command)
    assert "--runner" in command and command[command.index("--runner") + 1] == "pooling"


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_release_hashes_cover_weights_and_config(key: str) -> None:
    expected = fetch.expected_files(TARGETS[key].merged_artifact)
    assert "config.json" in expected
    assert any(name.endswith(".safetensors") for name in expected)
    assert all(len(digest) == 64 for digest in expected.values())


def test_verified_requires_every_file_to_match(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}")
    good = {"a.json": fetch.file_hash(tmp_path / "a.json")}
    assert fetch.verified(tmp_path, good)
    assert not fetch.verified(tmp_path, {**good, "b.json": "0" * 64})
    assert not fetch.verified(tmp_path, {"a.json": "0" * 64})


def _long_context_case(tmp_path: Path, key: str):
    import dataclasses

    from experiments.release_serving.serve import extend_context

    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    source = tmp_path / "worker.py"
    source.write_text("pass\n")
    target = TARGETS[key]
    spec = dataclasses.replace(
        target.long_context,
        receipt=str(receipt),
        receipt_sha256=fetch.file_hash(receipt),
        sources=(str(source),) if target.long_context.sources else (),
    )
    target = dataclasses.replace(target, long_context=spec)
    command = json.loads((ROOT / TARGETS[key].server_receipt).read_text())["command"]
    return extend_context, target, command


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_recorded_context_replays_the_recorded_command(
    key: str, tmp_path: Path
) -> None:
    extend_context, target, command = _long_context_case(tmp_path, key)
    original, env = list(command), {}
    extend_context(command, env, target, 32768)
    assert command == original and env == {}


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_long_context_changes_only_limit_worker_and_receipt(
    key: str, tmp_path: Path
) -> None:
    extend_context, target, command = _long_context_case(tmp_path, key)
    original, env = list(command), {}
    extend_context(command, env, target, 262144)
    changed = {
        original[i - 1]
        for i, (a, b) in enumerate(zip(original, command, strict=True))
        if a != b
    }
    assert changed <= {"--max-model-len", "--worker-cls", "--additional-config"}
    assert command[command.index("--max-model-len") + 1] == "262144"
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"
    before = json.loads(original[original.index("--additional-config") + 1])
    after = json.loads(command[command.index("--additional-config") + 1])
    spec = target.long_context
    if spec.receipt_env:
        assert env == {spec.receipt_env: spec.receipt}
        assert command[command.index("--worker-cls") + 1] == spec.worker[1]
        assert set(after["gleipnir_frost_fp4"]) - set(
            before["gleipnir_frost_fp4"]
        ) == set(spec.sources)
    else:
        assert env == {} and "--worker-cls" not in changed
        assert after[spec.additional_key]["validation_sha256"] == spec.receipt_sha256
    assert {k: v for k, v in after.items() if k != spec.additional_key} == {
        **before,
        "gleipnir_frost_fp4": after["gleipnir_frost_fp4"],
    }


def test_long_context_rejects_a_changed_receipt(tmp_path: Path) -> None:
    import dataclasses

    extend_context, target, command = _long_context_case(tmp_path, "4b")
    stale = dataclasses.replace(target.long_context, receipt_sha256="0" * 64)
    with pytest.raises(ValueError, match="receipt changed"):
        extend_context(
            command, {}, dataclasses.replace(target, long_context=stale), 262144
        )
    with pytest.raises(ValueError, match="context limit"):
        extend_context(command, {}, target, 2 * 262144)


def test_repeat_relative_gate() -> None:
    from experiments.release_serving.gate import gate

    recorded = {str(i): i / 10 for i in range(10)}
    pass_a = {k: v + (0.01 if int(k) % 2 else -0.01) for k, v in recorded.items()}
    pass_b = {k: v + (-0.01 if int(k) % 2 else 0.01) for k, v in recorded.items()}
    result = gate(pass_a, pass_b, recorded, 0.910, 0.912)
    assert result["mae_ratio"] == pytest.approx(0.5) and result["passed"]
    assert not gate(pass_a, pass_a, recorded, 0.910, 0.912)["passed"]
    assert not gate(pass_a, pass_b, recorded, 0.900, 0.912)["passed"]


@pytest.mark.parametrize("key", sorted(TARGETS))
@pytest.mark.parametrize("limit", [32768, 262144])
def test_prefix_caching_is_opt_in_flags_and_twin_worker(
    key: str, limit: int, tmp_path: Path
) -> None:
    from experiments.release_serving.serve import enable_prefix_caching

    extend_context, target, command = _long_context_case(tmp_path, key)
    extend_context(command, {}, target, limit)
    assert "--no-enable-prefix-caching" in command
    worker = command[command.index("--worker-cls") + 1]
    enable_prefix_caching(command, target)
    assert "--no-enable-prefix-caching" not in command
    index = command.index("--enable-prefix-caching")
    assert command[index + 1 : index + 3] == ["--mamba-cache-mode", "align"]
    served = command[command.index("--worker-cls") + 1]
    assert served == target.prefix_workers.get(worker, worker)


def test_graphs_are_added_only_for_targets_that_name_sizes() -> None:
    from experiments.release_serving.serve import GRAPHS, add_graphs

    sizes = TARGETS["4b"].graph_sizes
    assert len(sizes) == 78 and list(sizes) == sorted(set(sizes))
    assert sizes[-1] == 4096 and {287, 288, 767, 768, 1535} <= set(sizes)
    assert not [n for n in sizes if 256 < n < 768 and n % 32 not in (0, 31)]
    command = ["python", "--port", "1"]
    add_graphs(command, TARGETS["4b"])
    assert command[:3] == ["python", "--port", "1"] and command[3] == GRAPHS
    assert json.loads(command[4]) == {"cudagraph_capture_sizes": list(sizes)}
    with pytest.raises(ValueError, match="already sets"):
        add_graphs(command, TARGETS["4b"])
    recorded = ["python", GRAPHS, "{}"]
    add_graphs(recorded, TARGETS["27b"])
    assert recorded == ["python", GRAPHS, "{}"]


def test_native_tokenizer_puts_the_hook_first_on_the_path(tmp_path: Path) -> None:
    from experiments.release_serving.serve import (
        TOKENIZER_HOOK,
        enable_native_tokenizer,
    )

    env = {"PYTHONPATH": "src:."}
    enable_native_tokenizer(env, tmp_path)
    assert env["PYTHONPATH"] == f"{TOKENIZER_HOOK}:src:."
    assert env["GLEIPNIR_TOKENIZER_HOOK"] == "1"
    assert (TOKENIZER_HOOK / "sitecustomize.py").is_file()


def test_only_a_compatible_gpu_is_routed_through_the_hardware_hook() -> None:
    from experiments.release_serving.serve import (
        HARDWARE_HOOK,
        enable_compatible_gpu,
    )

    for gpu in ("NVIDIA B200", "NVIDIA RTX PRO 6000"):
        env = {"PYTHONPATH": "src:."}
        assert not enable_compatible_gpu(env, gpu)
        assert env == {"PYTHONPATH": "src:."}
    env = {"PYTHONPATH": "hook:src:."}
    assert enable_compatible_gpu(env, "NVIDIA B300 SXM6 AC")
    assert env == {
        "PYTHONPATH": f"{HARDWARE_HOOK}:hook:src:.",
        "GLEIPNIR_GPU_ALIAS": "NVIDIA B200",
        "GLEIPNIR_GPU_CAPABILITY": "10.0",
    }


def test_hardware_hook_aliases_the_gpu_and_runs_the_next_hook(tmp_path: Path) -> None:
    import subprocess
    import sys

    from experiments.release_serving.serve import enable_compatible_gpu

    (tmp_path / "torch").mkdir()
    (tmp_path / "torch/__init__.py").write_text("")
    (tmp_path / "torch/cuda.py").write_text(
        "get_device_name = lambda d=None: 'NVIDIA B300 SXM6 AC'\n"
        "get_device_capability = lambda d=None: (10, 3)"
    )
    (tmp_path / "gleipnir.py").write_text(
        "import torch.cuda as c\ncapability = lambda: c.get_device_capability()"
    )
    (tmp_path / "next").mkdir()
    (tmp_path / "next/sitecustomize.py").write_text("print('chained')")
    env = {"PYTHONPATH": f"{tmp_path / 'next'}:{tmp_path}"}
    enable_compatible_gpu(env, "NVIDIA B300 SXM6 AC")
    code = (
        "import torch.cuda as c, gleipnir as g;"
        "print(c.get_device_name(), c.get_device_capability(), g.capability())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.stdout.splitlines() == ["chained", "NVIDIA B200 (10, 3) (10, 0)"]


def test_attention_compile_uses_the_target_plan_and_limit() -> None:
    from experiments.release_serving.serve import attention_compile_command

    module = "experiments.release_serving.compile_attention"
    small = attention_compile_command(TARGETS["4b"], 262144)
    assert small[1:] == ["-m", module, "--length", "262144"]
    large = attention_compile_command(TARGETS["27b"], 32768)
    assert large[1:] == ["-m", module, "--length", "32768", "--heads", "24", "4"]


ENTRYPOINT = ROOT / "experiments/release_serving/image/entrypoint.sh"


def test_entrypoint_runs_other_commands_as_given() -> None:
    import subprocess

    result = subprocess.run(
        ["bash", str(ENTRYPOINT), "echo", "shell"], capture_output=True, text=True
    )
    assert result.returncode == 0 and result.stdout == "shell\n"


def test_entrypoint_serves_the_release_models_and_starts_ssh_only_with_a_key() -> None:
    text = ENTRYPOINT.read_text()
    assert '"" | 4b | 27b) ;;' in text
    assert '[ -n "${PUBLIC_KEY:-}" ] && [ -x /start.sh ]' in text
    assert text.rstrip().endswith(
        'exec "$PY" -m experiments.release_serving.serve "$MODEL" "$@"'
    )


def test_layer12_replaces_the_image_entrypoint() -> None:
    from experiments.release_serving.image import layer12

    targets = [
        target for source, target in layer12.FILES if source == layer12.ENTRYPOINT
    ]
    assert "opt/gleipnir-release/entrypoint.sh" in targets
    assert f"{layer12.IMAGE_ROOT}/{layer12.ENTRYPOINT}" in targets
    assert layer12.BASE.endswith(":1.1")
