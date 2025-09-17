# Architecture — Dialectica

## Overview
Sequential Python CLI orchestrating multiple LLM providers via a common interface. Ideas/ratings prefer structured JSON; drafts remain Markdown. All artifacts are saved under `./runs/<label-or-timestamp>/`.

## Modules (Current)
- `dialectica/cli/`: CLI commands and argument parsing
  - `ideas.py` — generate ideas command
  - `score.py` — provider scoring command
  - `select.py` — interactive idea selection
  - `draft.py` — drafting loop to consensus
  - `run.py` — combined flows (e.g., `run all` with `--auto-select`, `--from-ideas`, `--all-ideas`)
  - `branch.py` — branch from existing ideas into a fresh run
- `dialectica/providers/`: provider adapters
  - `base.py` — provider interface and shared types
  - `gpt5.py`, `gemini.py`, `grok.py` — concrete adapters
- `dialectica/pipeline/`:
  - `runner.py` — orchestrates sequential steps, creates run folder
  - `prompts.py` — prompt templates and constraint injection
  - `artifacts.py` — file naming, writing Markdown artifacts, transcripts
  - `consensus.py` — checks termination condition
  - `config.py` — config load/merge/validate, write `run.yml`
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
  - Constraints from files, inline text, or STDIN (merged verbatim under “Constraints of Paper”)
  - Field/domain pack for style/examples
  - Task instructions (ideas with Smart layperson, scoring rubric, critique+rewrite)

## Runs & Artifacts
- `./runs/<timestamp or label>/` structure:
  - `kickoff_prompt.md`
  - `constraints.md`, `constraints_sources.txt`
  - `ideas.json` (ideas_v1)
  - `ratings.json` (ratings_v1 combined; includes per-rater scores)
  - `selected_idea.md`
  - `drafts/round_01_gpt5.md`, `round_02_gemini.md`, `round_03_grok4.md`, ...
  - `judgments/round_XX_<provider>.md` (written when a provider outputs Publish)
  - `draft_prompt_round_XX.md` per round
  - `provenance.md` (for branched runs)
  - `consensus.md`, `paper.md` (full final draft), `paper_only.md` (paper body only), `paper_annotated.md` (annotations for smart layperson)
  - `transcripts/<provider>/step_<n>.md` (optional)
  - `run.yml` — resolved snapshot (providers, roles, rubric, policies, prompts, ui, provenance, sources)
- Naming: include round numbers and provider names for readability.

## Consensus Logic
- Done when at least two models output “Publish” on their latest evaluation of the current draft.
- If a model outputs Publish, that round records a judgment file only (no new draft). The latest draft remains unchanged.
- Publish judgments tied to older drafts do not count; a new draft invalidates older Publish judgments until models re-evaluate.
- Max cycle cap ensures progress; user can resume to extend cycles.

## Config & Secrets
- `dialectica.yml` (project defaults) + environment variables + CLI. Resolved config written to `runs/<id>/run.yml`.
- `.env` holds secrets (API keys). No telemetry by default.
## JSON Usage (No Fallback)
- Ideas and ratings: structured JSON (ideas_v1, ratings_v1) required from providers. Any invalid JSON/timeout/error aborts the run immediately; artifacts record failure status.
- Drafts: Markdown only (pretty-rendered in TUI).
## Criteria
- A `criteria.json` defines an overview and key/value pairs of criteria names to descriptions; TUI uses it to render score legends and explanations.

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
