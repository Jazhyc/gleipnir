"""Compatibility entrypoint for the historical injection-presence census."""

from gleipnir.evaluation.concept_census import (
    ASSISTANT_SUFFIX,
    CONTROLS,
    DEFAULT_CONFIG,
    digest,
    extract_score,
    file_hash,
    main,
    now,
    read_predictions,
    recover_trajectory,
    render_prompt,
    run,
    summarize,
    validate_cache,
    write_json,
)

__all__ = [
    "ASSISTANT_SUFFIX",
    "CONTROLS",
    "DEFAULT_CONFIG",
    "digest",
    "extract_score",
    "file_hash",
    "main",
    "now",
    "read_predictions",
    "recover_trajectory",
    "render_prompt",
    "run",
    "summarize",
    "validate_cache",
    "write_json",
]

if __name__ == "__main__":
    main()
