"""Disjoint adapter-level evaluation assignment for persistent GPU engines."""


def adapter_lanes(names: list[str], gpu_count: int = 2) -> list[list[str]]:
    """Round-robin equal-work full-suite evaluations without duplicate ownership."""
    if gpu_count < 1 or len(names) != len(set(names)) or not names:
        raise ValueError("require unique nonempty jobs and positive GPU count")
    return [names[i::gpu_count] for i in range(min(gpu_count, len(names)))]
