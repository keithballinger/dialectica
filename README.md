# Dialectica CLI

Python 3.11+ CLI to orchestrate GPT5, Gemini 2.5 Pro, and Grok4 to generate, evaluate, and iteratively draft a scientific paper under explicit constraints. Artifacts are saved as Markdown under `./runs/<timestamp>/`.

## Quick Start

- Clone and enter the repo
- Copy `.env.example` to `.env` and edit as needed
- Optionally leave `DIALECTICA_DRY_RUN=1` enabled for offline testing
- Install dependencies: `pip install -r requirements.txt`

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

## Constraint Files

Constraints can be specified in two formats:

**Markdown/Text (.md, .txt):**
```markdown
- In the field of computer science, focused on Large Language Model inference
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models
```

**JSON (.json):**
```json
{
  "overview": "Research into using small LLMs for agentic coding",
  "constraints": {
    "testable": "Can validate with code",
    "novelty": "Highly novel"
  }
}
```

Both formats are supported. Multiple constraint files can be combined using comma-separated paths.

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

## Literature Search Feature

Dialectica can automatically search academic databases (Semantic Scholar, arXiv) to find related papers and integrate them into the idea generation and drafting process.

### Enabling Literature Search

Set the environment variable in your `.env` file:
```
DIALECTICA_LITERATURE_SEARCH=1
```

Or use CLI flags:
```bash
python -m dialectica run ideas --constraints constraints/compsci.md --literature-search

python -m dialectica run all --constraints constraints/compsci.md --literature-search --auto-select
```

### What It Does

1. **During Idea Generation**: Searches for related papers based on your constraints and field
2. **Literature Review Artifact**: Saves `literature_review.md` in the run directory with paper summaries
3. **Context for AI**: Provides literature context to GPT-5 to ensure ideas are novel and build on existing work
4. **Citations in Drafts**: First draft includes citations to related work in markdown format `[Author et al., Year]`

### Configuration

- `DIALECTICA_LITERATURE_SEARCH=1` - Enable literature search (default: 0)
- `SEMANTIC_SCHOLAR_API_KEY=...` - Optional API key for higher rate limits (default: public access)

### Requirements

The `arxiv` package is required for arXiv search (included in `requirements.txt`):
```bash
pip install arxiv>=2.1.0
```

## Notes
- Set `DIALECTICA_DRY_RUN=0` to call real providers (ensure valid API keys/models in `.env`).
- Provider adapters live under `dialectica/providers/` and can be wired to real APIs.
- Prompts and artifact flows live under `dialectica/pipeline/`.
- Literature search uses free public APIs (Semantic Scholar, arXiv) with no authentication required.

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
