"""CPU-only campaign inventory, immutable binding and resolved training job."""

from __future__ import annotations

import fcntl
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gleipnir.data.injection_audit import audit_rows
from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    validate_targets,
    write_json,
)

REGISTRY = {"id": ("id",), "apps": ("benchmark", "honest_controls")}


@dataclass(frozen=True)
class Campaign:
    """Separate artifact workspace from the checkout whose sources are executed."""

    root: Path
    config_path: Path
    config: dict
    source_root: Path = Path(__file__).resolve().parents[4]

    @classmethod
    def load(cls, root: Path, config_path: Path) -> Campaign:
        root, config_path = root.resolve(), config_path.resolve()
        config = yaml.safe_load(config_path.read_text())
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", config["campaign_id"]):
            raise ValueError("campaign_id must be a simple directory name")
        names = config["evaluations"]
        if not names or len(set(names)) != len(names) or set(names) - REGISTRY.keys():
            raise ValueError(
                f"evaluations must be distinct registered names: {list(REGISTRY)}"
            )
        if (
            config["num_train_epochs"] != 1
            or config["profile"] != "qwen35_4b_b200_fp4_mlp"
        ):
            raise ValueError(
                "runner currently supports the validated one-epoch B200 FP4 profile"
            )
        if config.get("evaluation", {}).get("promote", False):
            raise ValueError(
                "held-out campaign evaluations cannot promote a checkpoint"
            )
        if config["model"]["id"] != "Qwen/Qwen3.5-4B":
            raise ValueError("runner currently supports the validated Qwen3.5-4B model")
        for name, spec in config.get("baselines", {}).items():
            if spec["kind"] not in names or spec["input"] not in config["inputs"]:
                raise ValueError(
                    f"baseline {name} requires its registered evaluation "
                    "and declared input"
                )
            if not spec.get("qualification"):
                raise ValueError(f"baseline {name} needs comparison qualifications")
        if (
            config["model"]["initial_tensor_sha256"]
            == config["inputs"]["initial_weights"]["sha256"]
        ):
            raise ValueError(
                "initializer tensor fingerprint must not be the file checksum"
            )
        if config["evaluation"]["port"] != 8010:
            raise ValueError("validated serving recipe uses port 8010")
        for name in ("batch_rows", "concurrency", "timeout_seconds"):
            if config["evaluation"][name] <= 0:
                raise ValueError(f"evaluation.{name} must be positive")
        for name in ("hypothesis", "intervention", "selection", "stop_condition"):
            if not str(config.get(name, "")).strip():
                raise ValueError(f"missing frozen experiment field: {name}")
        return cls(root, config_path, config)

    @property
    def output(self) -> Path:
        return self.root / "results" / self.config["campaign_id"]

    @property
    def data(self) -> Path:
        return self.root / "data" / self.config["campaign_id"]

    @property
    def logs(self) -> Path:
        return self.root / "logs/runpod" / self.config["campaign_id"]

    @property
    def adapter(self) -> Path:
        return self.output / "4b/monitor"

    @property
    def serving(self) -> Path:
        return self.root / "results/b200_attention_gdn_serving"

    def input(self, name: str) -> Path:
        return self.root / self.config["inputs"][name]["path"]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(s for name in self.config["evaluations"] for s in REGISTRY[name])

    def profile(self) -> dict:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(self.source_root / "src/gleipnir/configs/systems_screen"),
        ):
            return OmegaConf.to_container(
                compose(config_name=self.config["profile"]), resolve=True
            )

    def job(self) -> dict:
        config = self.config
        recipe = self.profile()["recipe"]
        recipe["startup_validation_reference"] = str(self.input("startup_reference"))
        return {
            **recipe,
            "job_name": config["campaign_id"],
            "model": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "seed": config["seed"],
            "student_rows": str(self.input("training")),
            "soft_targets": str(self.input("teacher")),
            "selection_manifest": None,
            "train_rows": config["training_rows"],
            "num_train_epochs": 1.0,
            "max_steps": -1,
            "learning_rate": config["learning_rate"],
            "soft_loss_weight": config["soft_loss_weight"],
            "direct_loss_weight": config["direct_loss_weight"],
            "completion_loss_weight": 0.0,
            "save_steps": 1000000,
            "expected_initial_master_sha256": config["model"]["initial_tensor_sha256"],
            "output_dir": str(self.adapter),
            "causal_adapter_dir": str(self.adapter / "causal_adapter"),
            "model_dir": str(self.adapter / "model"),
            "hydra_log_dir": str(self.logs / "hydra"),
        }

    def sources(self) -> dict[str, str]:
        paths = []
        for directory in (
            "src/gleipnir",
            "experiments/deception_distillation",
            "experiments/tool_trajectory_monitoring",
            "experiments/training_procedure_screen",
            "experiments/b200_vllm031",
            "experiments/b200_attention_precision",
        ):
            paths.extend(
                p
                for p in (self.source_root / directory).rglob("*")
                if p.suffix in (".py", ".yaml", ".json")
            )
        return {
            str(p.relative_to(self.source_root)): file_hash(p) for p in sorted(paths)
        }

    def inventory(self, *, runtime: bool = False) -> dict:
        """Collect every missing/drifted prerequisite before doing GPU work."""
        errors, checked = [], {}
        required = {
            "training",
            "teacher",
            "token_audit",
            "initial_weights",
            "initial_adapter_config",
            "startup_reference",
            "serving_selection",
        }
        required |= {s for s in self.splits} | {s + "_workload" for s in self.splits}
        if self.config.get("augmentation_audit", False):
            required |= {"clean_training", "ledger", "templates"}
        for name in sorted(required - self.config["inputs"].keys()):
            errors.append(f"undeclared input: {name}")
        specs = {
            name: (self.input(name), spec["sha256"])
            for name, spec in self.config["inputs"].items()
        }
        selection_path = specs.get("serving_selection", (Path("/nonexistent"), None))[0]
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text())
            for path, sha in selection["artifact_bindings"].items():
                specs[f"serving:{path}"] = (self.root / path, sha)
            for key in ("merged_artifact", "host_parent", "recipe_summary"):
                path = selection[key]
                specs.setdefault(f"serving:{path}", (self.root / path, None))
            merge_path = self.root / selection["merged_artifact"]
            if merge_path.is_file():
                for path, sha in json.loads(merge_path.read_text())[
                    "source_files_sha256"
                ].items():
                    specs[f"base:{path}"] = (
                        self.root / self.config["base_model"] / path,
                        sha,
                    )
        if selection_path.is_file():
            report_path = self.root / selection["recipe_summary"]
            if report_path.is_file():
                command = json.loads(report_path.read_text())["command"]
                additional = json.loads(
                    command[command.index("--additional-config") + 1]
                )
                for key, value in additional["serving_condition"].items():
                    if key.endswith(("_validation", "_reference")):
                        refs = value.values() if isinstance(value, dict) else [value]
                        for path in refs:
                            if str(path).startswith("results/"):
                                specs.setdefault(
                                    f"serving:{path}", (self.root / path, None)
                                )
                for path, sha in additional["gleipnir_frost_fp4"].items():
                    # Diagnostic sources can be restored from their frozen archive;
                    # runtime arithmetic must already match the selected recipe.
                    if (
                        Path(path).name.startswith("test_")
                        or Path(path).name.endswith(("_canary.py", "_compare.py"))
                        or Path(path).name in {"run.py", "fp4_gemm_tune.py"}
                        or path == "src/gleipnir/serving/reference.py"
                    ):
                        continue
                    specs[f"runtime:{path}"] = (self.source_root / path, sha)
        archive = self.root / "results/b200_vllm031/pre_migration_sources.tar.gz"
        specs["serving:diagnostic_archive"] = (archive, None)
        for name, (path, expected) in specs.items():
            if not path.is_file():
                errors.append(f"missing {name}: {path}")
                continue
            actual = file_hash(path)
            checked[str(path)] = actual
            if expected is not None and actual != expected:
                errors.append(f"checksum mismatch {name}: {path}")
        initial = self.root / self.config["model"]["initial_adapter"]
        for name, filename in (
            ("initial_weights", "adapter_model.safetensors"),
            ("initial_adapter_config", "adapter_config.json"),
        ):
            if (
                name in specs
                and (initial / filename).resolve() != specs[name][0].resolve()
            ):
                errors.append(f"model.initial_adapter disagrees with inputs.{name}")
        if runtime:
            for name in ("training_python", "serving_python"):
                if not Path(self.config[name]).is_file():
                    errors.append(f"missing executable {name}: {self.config[name]}")
            if self.root != self.source_root:
                errors.append("GPU run must use the source checkout as its workspace")
        return {"passed": not errors, "errors": errors, "files_sha256": checked}

    def check(self) -> dict:
        binding = json.loads((self.data / "manifest.json").read_text())
        if file_hash(self.config_path) != binding["config_sha256"]:
            raise ValueError("campaign configuration drift")
        for path, expected in binding["files_sha256"].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f"campaign input drift: {path}")
        for name, expected in binding["source_sha256"].items():
            if file_hash(self.source_root / name) != expected:
                raise ValueError(f"campaign source drift: {name}")
        if (
            digest(json.dumps(self.profile(), sort_keys=True))
            != binding["profile_sha256"]
        ):
            raise ValueError("resolved training profile drift")
        return binding

    def prepare(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._prepare()

    def _prepare(self) -> dict:
        inventory = self.inventory()
        if not inventory["passed"]:
            raise ValueError(
                "campaign prerequisites failed:\n" + "\n".join(inventory["errors"])
            )
        if (self.data / "manifest.json").exists():
            return self.check()
        config = self.config
        rows = read_rows(self.input("training"))
        validate_targets(rows, read_rows(self.input("teacher")))
        job = self.job()
        if (
            len(rows) != config["training_rows"]
            or math.ceil(len(rows) / config["logical_batch_size"])
            != config["expected_steps"]
        ):
            raise ValueError("training population/update count changed")
        if job["effective_batch_size"] != config["logical_batch_size"]:
            raise ValueError("logical batch size disagrees with selected profile")
        if (
            job["startup_validation_reference_sha256"]
            != config["inputs"]["startup_reference"]["sha256"]
        ):
            raise ValueError("startup reference disagrees with selected profile")
        audit = json.loads(self.input("token_audit").read_text())
        if (
            audit["total"] != config["expected_training_tokens"]
            or audit["rows"] != len(rows)
            or audit["maximum"] > job["max_length"]
            or audit["truncated"]
        ):
            raise ValueError("token audit population/context changed")
        augmentation = None
        if config.get("augmentation_audit", False):
            augmentation = audit_rows(
                read_rows(self.input("clean_training")),
                rows,
                read_rows(self.input("ledger")),
                json.loads(self.input("templates").read_text()),
            )
            if augmentation["injected_rows"] != config["injected_rows"]:
                raise ValueError("augmentation coverage changed")
        # Every evaluation population is checked before training, including its
        # cached rendered prompts and hard-label lineage.
        from .evaluation import validate_workloads

        validate_workloads(self)
        from .evaluation import attach_inputs

        for spec in config.get("baselines", {}).values():
            path = self.input(spec["input"])
            if spec["kind"] == "id":
                saved = (
                    json.loads(path.read_text())
                    if path.suffix == ".json"
                    else read_rows(path)
                )
                attach_inputs(saved, read_rows(self.input("id")))
                if any(
                    not math.isfinite(r["score"]) or not 0 <= r["score"] <= 1
                    for r in saved
                ):
                    raise ValueError("invalid baseline score")
            else:
                saved = json.loads(path.read_text())
                for key in spec.get("keys", []):
                    saved = saved[key]
                if not isinstance(saved, dict) or not saved:
                    raise ValueError("invalid APPS baseline summary")
        link = self.data / "monitor/student_rows.jsonl"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.resolve() != self.input("training").resolve():
                raise ValueError("prepared training population changed")
        else:
            link.symlink_to(self.input("training"))
        sources = self.sources()
        for name in sources:
            path = self.output / "executed_sources" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.source_root / name).read_bytes())
        binding = {
            **inventory,
            "config_sha256": file_hash(self.config_path),
            "source_sha256": sources,
            "profile_sha256": digest(json.dumps(self.profile(), sort_keys=True)),
            "augmentation_audit": augmentation,
            "token_audit": audit,
            "selection": config["selection"],
        }
        (self.output / "config.yaml").write_bytes(self.config_path.read_bytes())
        write_json(self.output / "config.json", config)
        write_json(self.output / "resolved_profile.json", self.profile())
        write_json(self.adapter / "job.json", job)
        write_json(self.adapter / "token_audit.json", audit)
        write_json(self.data / "manifest.json", binding)
        return binding
