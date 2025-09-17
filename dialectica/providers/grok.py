from __future__ import annotations

import os
from pathlib import Path
from .base import DryRunProvider, Provider
from ..errors import ProviderError
from ..utils import is_dry_run


class GrokProvider:
    name = "Grok"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("GROK_MODEL", "grok-4")
        self._dry = DryRunProvider(self.name, self.model)
        # Do not set temperature; use model defaults

    def _client(self):
        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            raise ProviderError("Missing XAI_API_KEY for Grok provider. Set it in .env.")
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ProviderError("openai package not installed. Run: pip install openai") from e
        return OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str) -> str:
        if is_dry_run():
            return self._dry.complete(prompt)
        try:
            client = self._client()
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content or ""
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"Grok (xAI) API error: {e}") from e

    def complete_with_json_object(self, prompt: str) -> str:
        """Attempt OpenAI-compatible JSON mode via response_format=json_object.
        If the API rejects response_format, this call may raise; callers should fallback.
        """
        if is_dry_run():
            # Minimal placeholder JSON
            items = [{"index": i, "criteria": {"overall": {"score": 7}}} for i in range(1, 11)]
            import json
            return json.dumps({"items": items}, indent=2)
        try:
            client = self._client()
            completion = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            content = completion.choices[0].message.content or ""
            return content
        except Exception as e:
            raise ProviderError(f"Grok JSON mode error: {e}") from e


def get_provider() -> Provider:
    return GrokProvider()
