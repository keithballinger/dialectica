from __future__ import annotations

import os
from .base import DryRunProvider, Provider
from ..errors import ProviderError
from ..utils import is_dry_run


class GrokProvider:
    name = "Grok"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("GROK_MODEL", "grok-4")
        self._dry = DryRunProvider(self.name, self.model)

    def complete(self, prompt: str) -> str:
        if is_dry_run():
            return self._dry.complete(prompt)
        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            raise ProviderError("Missing XAI_API_KEY for Grok provider. Set it in .env.")
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content or ""
        except ImportError as e:
            raise ProviderError("openai package not installed. Run: pip install openai") from e
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"Grok (xAI) API error: {e}") from e


def get_provider() -> Provider:
    return GrokProvider()
