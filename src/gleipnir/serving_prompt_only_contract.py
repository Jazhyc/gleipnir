"""Admission rules for complete-prompt, one-decision-token monitoring."""


def validate_prompt_only_request(max_tokens: int | None, n: int = 1) -> None:
    """Reject requests that would require a second model forward."""
    if max_tokens != 1 or n != 1:
        raise ValueError("prompt-only monitoring requires max_tokens=1 and n=1")


def validate_complete_prompts(lengths: list[int], computed: list[int]) -> None:
    """Require fresh, unchunked prompts inside the validated serving envelope."""
    if (
        not 1 <= len(lengths) <= 128
        or len(lengths) != len(computed)
        or any(n <= 0 or n > 32768 for n in lengths)
        or sum(lengths) > 32768
        or any(n != 0 for n in computed)
    ):
        raise ValueError("prompt-only monitoring requires complete fresh prompts")
