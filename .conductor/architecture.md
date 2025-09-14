# Architecture — Dialectica

## Overview
Sequential Python CLI orchestrating three LLM providers via a common interface. All artifacts are Markdown files saved under `./runs/<timestamp>/`.

## Modules (Proposed)
- `dialectica/cli/`: CLI commands and argument parsing
  - `ideas.py` — generate ideas command
  - `score.py` — provider scoring command
  - `select.py` — interactive idea selection
  - `draft.py` — drafting loop to consensus
  - `run.py` — combined flows (e.g., `run all`)
- `dialectica/providers/`: provider adapters
  - `base.py` — provider interface and shared types
  - `gpt5.py`, `gemini.py`, `grok.py` — concrete adapters
- `dialectica/pipeline/`:
  - `runner.py` — orchestrates sequential steps, creates run folder
  - `prompts.py` — prompt templates and constraint injection
  - `artifacts.py` — file naming, writing Markdown artifacts, transcripts
  - `consensus.py` — checks termination condition
- `dialectica/errors.py` — custom exceptions

## Provider Interface
```python
class Provider(Protocol):
    name: str
    model: str
    def complete(self, prompt: str) -> str: ...
```
- Strictly sequential calls. Implement retry/backoff for transient errors.
- No streaming or JSON parsing; plain text in, Markdown out.

## Prompts
- Compose from:
  - System behavior (truth-seeking, judgment-first rule)
  - Constraints from `./constraints/*.md` (inserted verbatim under “Constraints of Paper”)
  - Task-specific instructions (ideas, scoring rubric, critique+rewrite)

## Runs & Artifacts
- `./runs/<timestamp>/` structure:
  - `kickoff_prompt.md`
  - `ideas_gpt5.md`
  - `ratings_grok4.md`, `ratings_gemini.md`
  - `selected_idea.md`
  - `drafts/round_01_gpt5.md`, `round_02_gemini.md`, `round_03_grok4.md`, ...
  - `consensus.md`, `paper.md`
  - `transcripts/<provider>/step_<n>.md` (optional)
- Naming: include round numbers and provider names for readability.

## Consensus Logic
- “Publishable with minor revisions” from all three models on the latest draft.
- Optionally enforce a max cycle cap (TBD) to prevent runaway loops; prompt user if reached.

## Config & Secrets
- `.env` only. Load keys and model names from environment variables.
- No additional config layers; no telemetry.

## Error Handling
- ProviderError, PromptError, ArtifactError for precise failure modes.
- Recover by resuming from last artifact; design commands to be idempotent when possible.

## Open Questions
- CLI name and command layout confirmation.
- Interactive UX: simple prompts vs TUI.
- Whether to include a `resume` command in initial release.

## Provider Wiring Examples (Libraries)

These examples illustrate the intended libraries and minimal usage patterns for each provider. The CLI uses plain prompt strings (single user message) and expects Markdown text responses.

### GPT5 (OpenAI client)

Library: `openai` (new SDK)

```python
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # model via env GPT5_MODEL
resp = client.chat.completions.create(
    model=os.environ.get("GPT5_MODEL", "gpt5"),
    messages=[{"role": "user", "content": prompt}],
)
text = resp.choices[0].message.content
```

### Gemini 2.5 Pro (Google Generative AI)

Library: `google-generativeai`

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])  # model via env GEMINI_MODEL
model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
resp = model.generate_content(prompt)
text = resp.text
```

### Grok4 (xAI, OpenAI-compatible)

Library: `openai` (new SDK) with base_url

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["XAI_API_KEY"],
    base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"),
)
resp = client.chat.completions.create(
    model=os.environ.get("GROK_MODEL", "grok-4"),
    messages=[{"role": "user", "content": prompt}],
)
text = resp.choices[0].message.content
```

All adapters honor the `DIALECTICA_DRY_RUN=1` flag to return deterministic placeholders for offline testing.

# Provider examples

## Grok

### Pip install
pip install xai-sdk

### code sample
import os

from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
api_key=os.getenv("XAI_API_KEY"),
timeout=3600, # Override default timeout with longer timeout for reasoning models
)

chat = client.chat.create(model="grok-4")
chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
chat.append(user("What is the meaning of life, the universe, and everything?"))

response = chat.sample()
print(response.content)

## GPT5

### pip install
pip install openai

### code sample

from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)

## Gemini

### pip install
pip install -q -U google-genai

### code samples
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)
