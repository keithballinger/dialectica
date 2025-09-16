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

## CLI Overview

- `run ideas`:
  - Purpose: Generate 10 ideas from constraints.
  - Flags: `--constraints <file[,file2]>`, `--name <label>`
- `run score`:
  - Purpose: Score ideas with GPT5, Grok4, Gemini.
  - Flags: `--run <dir>`
- `select`:
  - Purpose: Choose the idea (interactive) or auto-pick.
  - Flags: `--run <dir>`, `--auto`, `--seed <int>`
- `draft`:
  - Purpose: Draft/critique loop to consensus.
  - Flags: `--run <dir>`, `--ask-to-continue`, `--max-cycles <int>`
- `run all` (combined flow):
  - Purpose: Ideas → (optional) score → select → draft.
  - Flags: `--constraints <files>` | `--from-ideas <path>`, `--name <label>`,
    `--auto-select`, `--seed <int>`, `--idea <n>`, `--all-ideas`, `--max-cycles <int>`
- `branch`:
  - Purpose: Start a new run from an existing ideas file and selected idea.
  - Flags: `--from-run <dir>` | `--from-ideas <path>`, `--idea <n>`, `--name <label>`, `--start`, `--max-cycles <int>`
- `resume`:
  - Purpose: Continue a run from its last phase.
  - Flags: `--run <dir>`, `--ask-to-continue`, `--max-cycles <int>`

## Notes
- Set `DIALECTICA_DRY_RUN=0` to call real providers (ensure valid API keys/models in `.env`).
- Provider adapters live under `dialectica/providers/` and can be wired to real APIs.
- Prompts and artifact flows live under `dialectica/pipeline/`.

## Common Flows

Auto full flow (no pauses)

```
python -m dialectica run all \
  --constraints constraints/quantum_ibm_cost.md \
  --auto-select --seed 42 --max-cycles 1000 --name auto
```

Batch all ideas (separate runs per idea)

```
python -m dialectica run all \
  --constraints constraints/quantum_ibm_cost.md \
  --all-ideas --max-cycles 1000 --name batch
```

Start from an existing ideas file (auto-select)

```
python -m dialectica run all \
  --from-ideas runs/compsci/ideas_gpt5.md \
  --auto-select --max-cycles 1000 --name from-compsci
```

Branch from a previous run’s ideas (pick specific idea and start)

```
python -m dialectica branch \
  --from-run runs/compsci --idea 7 --name compsci-idea-7 --start --max-cycles 1000
```

Makefile shortcuts

```
make run-all-auto           # auto full flow with quantum_ibm_cost constraints
make run-all-batch          # batch all ideas with quantum_ibm_cost constraints
make run-from-ideas-auto    # from IDEAS (see Makefile vars), auto-select
make branch-compsci-7       # branch from runs/compsci idea #7 and start
```

Dry-run variants (deterministic placeholders)

```
make run-all-auto-dry
make run-all-batch-dry
make run-from-ideas-auto-dry
```
