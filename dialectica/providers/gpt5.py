from __future__ import annotations

import json
import os
from pathlib import Path
from .base import DryRunProvider, Provider
from ..errors import ProviderError
from ..utils import is_dry_run


class GPT5Provider:
    name = "GPT5"

    def __init__(self, model: str | None = None):
        # Per user: model is exactly "gpt-5"
        self.model = model or os.environ.get("GPT5_MODEL", "gpt-5")
        self._dry = DryRunProvider(self.name, self.model)

    def complete(self, prompt: str) -> str:
        if is_dry_run():
            return self._dry.complete(prompt)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT5_API_KEY")
        if not api_key:
            raise ProviderError("Missing OPENAI_API_KEY for GPT5 provider. Set it in .env.")
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            # Default to chat completions for free-form Markdown outputs
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content or ""
        except ImportError as e:
            raise ProviderError("openai package not installed. Run: pip install openai") from e
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"GPT5 API error: {e}") from e

    def complete_with_json_schema(self, prompt: str, schema_path: Path, schema_name: str) -> str:
        """Use OpenAI chat completions with JSON mode.
        Returns a JSON string that should validate against the schema.
        """
        if is_dry_run():
            # In dry-run, just return a minimal valid-looking JSON payload
            try:
                schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
                # Create a placeholder items list with indexes and only overall
                items = [{"index": i, "criteria": {"overall": {"score": 7}}} for i in range(1, 11)]
                return json.dumps({"items": items}, indent=2)
            except Exception:
                return json.dumps({"items": []})

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT5_API_KEY")
        if not api_key:
            raise ProviderError("Missing OPENAI_API_KEY for GPT5 provider. Set it in .env.")
        try:
            from openai import OpenAI  # type: ignore

            # Load schema for reference in the prompt
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

            # Add schema info to the prompt
            system_prompt = f"You are a helpful assistant designed to output JSON that matches this schema: {json.dumps(schema)}"

            client = OpenAI(api_key=api_key)
            # Use standard chat completions with JSON mode
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content or "{}"
        except ImportError as e:
            raise ProviderError("openai package not installed. Run: pip install openai") from e
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"GPT5 JSON schema API error: {e}") from e


def get_provider() -> Provider:
    return GPT5Provider()
