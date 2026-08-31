"""Load and render the versioned teacher and student prompt contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PromptRole = Literal["teacher", "student"]
PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"
MANIFEST_PATH = PROMPT_DIRECTORY / "manifest.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"prompt manifest field {key!r} must be a nonempty string")
    return value


def _instruction_path(prompt_directory: Path, filename: str) -> Path:
    path = (prompt_directory / filename).resolve()
    if path.parent != prompt_directory.resolve():
        raise ValueError(f"prompt instruction must be inside {prompt_directory}")
    return path


@dataclass(frozen=True)
class PromptTemplate:
    """One instruction prefix and the common action-only trajectory envelope."""

    prompt_set_id: str
    role: PromptRole
    instruction: str
    trajectory_open: str
    trajectory_close: str
    decision_prefix: str
    negative_label: str
    positive_label: str
    uses: tuple[str, ...]

    @property
    def cache_prefix(self) -> str:
        """Return the invariant leading substring for provider input caching."""
        return f"{self.instruction}\n{self.trajectory_open}\n"

    @property
    def template_sha256(self) -> str:
        """Hash every prompt component that can change a rendered request."""
        identity = json.dumps(
            {
                "prompt_set_id": self.prompt_set_id,
                "role": self.role,
                "instruction": self.instruction,
                "trajectory_open": self.trajectory_open,
                "trajectory_close": self.trajectory_close,
                "decision_prefix": self.decision_prefix,
                "negative_label": self.negative_label,
                "positive_label": self.positive_label,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return _sha256(identity)

    def render(self, trajectory: str) -> str:
        """Render one generation prompt without normalizing the trajectory."""
        if not isinstance(trajectory, str) or not trajectory.strip():
            raise ValueError("trajectory must be a nonempty string")
        separator = "" if trajectory.endswith("\n") else "\n"
        return (
            f"{self.cache_prefix}{trajectory}{separator}"
            f"{self.trajectory_close}\n"
        )

    def decision_context(self, trajectory: str) -> str:
        """Render the context whose next-token logits are compared at 0 versus 1."""
        return f"{self.render(trajectory)}{self.decision_prefix}"

    def completion(self, label: str) -> str:
        """Return the exact assistant target used for hard-label supervision."""
        if label not in {self.negative_label, self.positive_label}:
            raise ValueError(
                f"label must be {self.negative_label!r} or {self.positive_label!r}"
            )
        return f"{self.decision_prefix}{label}"

    def rendered_sha256(self, trajectory: str) -> str:
        return _sha256(self.render(trajectory))


@dataclass(frozen=True)
class PromptSet:
    """The paired full-teacher and compact-student prompt templates."""

    prompt_set_id: str
    teacher: PromptTemplate
    student: PromptTemplate

    def for_role(self, role: PromptRole) -> PromptTemplate:
        if role == "teacher":
            return self.teacher
        if role == "student":
            return self.student
        raise ValueError(f"unknown prompt role: {role!r}")


def _uses(manifest: dict[str, Any], key: str) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"prompt manifest field {key!r} must be a string list")
    return tuple(value)


def load_prompt_set(prompt_directory: Path = PROMPT_DIRECTORY) -> PromptSet:
    """Load the human-editable prompt files and their shared interface."""
    manifest_path = prompt_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("prompt manifest must contain a JSON object")

    prompt_set_id = _require_string(manifest, "prompt_set_id")
    trajectory_open = _require_string(manifest, "trajectory_open")
    trajectory_close = _require_string(manifest, "trajectory_close")
    decision_prefix = _require_string(manifest, "decision_prefix")
    negative_label = _require_string(manifest, "negative_label")
    positive_label = _require_string(manifest, "positive_label")
    if negative_label == positive_label:
        raise ValueError("negative and positive labels must be different")

    def build(role: PromptRole) -> PromptTemplate:
        filename = _require_string(manifest, f"{role}_instruction")
        instruction = _instruction_path(prompt_directory, filename).read_text(
            encoding="utf-8"
        )
        if not instruction.strip():
            raise ValueError(f"{role} instruction must not be empty")
        return PromptTemplate(
            prompt_set_id=prompt_set_id,
            role=role,
            instruction=instruction,
            trajectory_open=trajectory_open,
            trajectory_close=trajectory_close,
            decision_prefix=decision_prefix,
            negative_label=negative_label,
            positive_label=positive_label,
            uses=_uses(manifest, f"{role}_uses"),
        )

    return PromptSet(
        prompt_set_id=prompt_set_id,
        teacher=build("teacher"),
        student=build("student"),
    )
