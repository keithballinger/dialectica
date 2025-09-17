from __future__ import annotations

import json
import os
from pathlib import Path
from .base import DryRunProvider, Provider
from ..errors import ProviderError
from ..utils import is_dry_run


class GeminiProvider:
    name = "Gemini"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
        self._dry = DryRunProvider(self.name, self.model)
        # Do not set temperature; use model defaults

    def _model(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("Missing GOOGLE_API_KEY for Gemini provider. Set it in .env.")
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ProviderError("google-generativeai package not installed. Run: pip install google-generativeai") from e
        genai.configure(api_key=api_key)
        # Defer generation_config to call sites
        return genai

    def complete(self, prompt: str) -> str:
        if is_dry_run():
            return self._dry.complete(prompt)
        try:
            genai = self._model()
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", "") or ""
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"Gemini API error: {e}") from e

    def complete_json(self, prompt: str) -> str:
        """Request JSON response by setting response_mime_type to application/json.
        Not all SDK versions support response_schema; we at least enforce JSON MIME.
        Returns a JSON string (resp.text) or raises ProviderError.
        """
        if is_dry_run():
            items = [{"index": i, "criteria": {"overall": {"score": 7}}} for i in range(1, 11)]
            return json.dumps({"items": items}, indent=2)
        try:
            genai = self._model()
            gen_cfg = genai.GenerationConfig(response_mime_type="application/json")
            model = genai.GenerativeModel(self.model, generation_config=gen_cfg)
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if text:
                return text
            # Fallback: try candidates list
            cand = getattr(resp, "candidates", None)
            if cand and isinstance(cand, list):
                # Try to extract first JSON-like text
                for c in cand:
                    t = getattr(c, "content", None) or getattr(c, "text", None)
                    if isinstance(t, str) and t.strip():
                        return t
            raise ProviderError("Gemini JSON completion returned no text")
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"Gemini JSON API error: {e}") from e


def get_provider() -> Provider:
    return GeminiProvider()
