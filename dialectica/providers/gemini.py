from __future__ import annotations

import os
from .base import DryRunProvider, Provider
from ..errors import ProviderError
from ..utils import is_dry_run


class GeminiProvider:
    name = "Gemini"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
        self._dry = DryRunProvider(self.name, self.model)

    def complete(self, prompt: str) -> str:
        if is_dry_run():
            return self._dry.complete(prompt)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("Missing GOOGLE_API_KEY for Gemini provider. Set it in .env.")
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            model_name = self.model
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            # google-generativeai returns candidates; .text is convenient accessor
            return getattr(resp, "text", "") or "\n".join(getattr(resp, "candidates", []) or [])
        except ImportError as e:
            raise ProviderError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            ) from e
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"Gemini API error: {e}") from e


def get_provider() -> Provider:
    return GeminiProvider()
