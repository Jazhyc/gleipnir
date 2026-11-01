"""Preserve Tinker's structured refusal before the shared completion adapter."""

from experiments.adversarial_code_or_comments.llm import Client as BaseClient


class Client(BaseClient):
    def chat(self, messages: list[dict], attempt: int = 1) -> dict:
        post = self.post
        metadata = {}

        def capture(body, key, timeout):
            data = post(body, key, timeout)
            choices = data.get("choices") or []
            message = (choices[0].get("message") or {}) if choices else {}
            if "refusal" in message:
                metadata["refusal"] = message["refusal"]
            metadata["cost_status"] = (
                "actual"
                if (data.get("usage") or {}).get("cost") is not None
                else "unknown_reserved"
            )
            return data

        self.post = capture
        try:
            return {**super().chat(messages, attempt), **metadata}
        finally:
            self.post = post
