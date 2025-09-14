# Dialectica CLI

Python 3.11+ CLI to orchestrate GPT5, Gemini 2.5 Pro, and Grok4 to generate, evaluate, and iteratively draft a scientific paper under explicit constraints. Artifacts are saved as Markdown under `./runs/<timestamp>/`.

## Quick Start

- Clone and enter the repo
- Copy `.env.example` to `.env` and edit as needed
- Optionally leave `DIALECTICA_DRY_RUN=1` enabled for offline testing

### Generate ideas (dry-run)

```
python -m dialectica run ideas --constraints constraints/quantum_ibm_cost.md --name test-run
```

### Score ideas

```
python -m dialectica run score
```

### Select an idea (interactive)

```
python -m dialectica select
```

### Draft to consensus (default max 10 cycles)

```
python -m dialectica draft --ask-to-continue
```

### One-shot flow (stops for selection)

```
python -m dialectica run all --constraints constraints/quantum_ibm_cost.md --ask-to-continue
```

### Resume a run

```
python -m dialectica resume
```

## Notes
- Set `DIALECTICA_DRY_RUN=0` to require real providers (not yet implemented in this skeleton).
- Provider adapters live under `dialectica/providers/` and can be wired to real APIs.
- Prompts and artifact flows live under `dialectica/pipeline/`.
