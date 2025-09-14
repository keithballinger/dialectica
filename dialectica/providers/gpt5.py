from __future__ import annotations

import os
from .base import DryRunProvider, Provider
from ..errors import ProviderError
from ..utils import is_dry_run


class GPT5Provider:
    name = "GPT5"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("GPT5_MODEL", "gpt5")
        self._dry = DryRunProvider(self.name, self.model)

    def complete(self, prompt: str) -> str:
        if is_dry_run():
            return self._dry.complete(prompt)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT5_API_KEY")
        if not api_key:
            raise ProviderError("Missing OPENAI_API_KEY for GPT5 provider. Set it in .env.")
        try:
            # Prefer new OpenAI client
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content or ""
        except ImportError as e:
            raise ProviderError("openai package not installed. Run: pip install openai") from e
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"GPT5 API error: {e}") from e


def get_provider() -> Provider:
    return GPT5Provider()
