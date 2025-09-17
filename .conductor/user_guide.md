# User Guide — Dialectica

## Overview
Dialectica is a Python CLI that coordinates GPT5, Gemini 2.5 Pro, and Grok4 to invent a novel, testable scientific theory and iteratively draft a short Markdown paper until all models judge “Publish.” All artifacts are saved to `./runs/<timestamp>/` as Markdown files.

## Requirements
- macOS
- Python 3.11+
- Accounts/keys for: GPT5, Gemini 2.5 Pro, Grok4
- `.env` at project root with provider keys and model identifiers

## Installation
1) `git clone <your-repo-url>`
2) `cd dialectica`
3) Create `.env` with entries such as:
   - `OPENAI_API_KEY=...` (for GPT5)
   - `GOOGLE_API_KEY=...` (for Gemini 2.5 Pro)
   - `XAI_API_KEY=...` (for Grok4)
   - Optionally, explicit model names if needed: `GPT5_MODEL=...`, `GEMINI_MODEL=...`, `GROK_MODEL=...`
4) Ensure Python dependencies are installed once implemented (see repo README when available).

## Concepts
- Constraints: Markdown files under `./constraints/` that define the “Constraints of Paper” section injected verbatim into prompts.
- Run: A single end-to-end session saved under `./runs/<timestamp>/` with all prompts, responses, and drafts.
- Consensus: At least two models output “Publish” on the latest draft (Publish tied to older drafts does not count).
- Pausing: By default, the CLI pauses for idea selection; optionally also after each drafting round.

## Directory Layout
- `./constraints/` — Your constraint files (e.g., `quantum_basic.md`, `quantum_ibm_cost.md`).
- `./runs/<timestamp>/` — Artifacts per run:
  - `kickoff_prompt.md`
  - `ideas_gpt5.md`
  - `ratings_grok4.md`
  - `ratings_gemini.md`
  - `selected_idea.md`
  - `drafts/round_01_gpt5.md`, `round_02_gemini.md`, `round_03_grok4.md`, ...
  - `judgments/round_XX_<provider>.md` (when a provider outputs Publish)
  - `consensus.md` (termination summary)
  - `paper.md` (final draft file, may include critique scaffolding)
  - `paper_only.md` (just the final paper body)
  - `paper_annotated.md` (paper with inserted notes for a smart layperson; all math terms, variables, and equations explained)
  - `transcripts/<provider>/...` (optional, provider dialogues)

## CLI Usage (Proposed)

Note: Command names and flags will be finalized during implementation. The following reflects the intended UX.

- Initialize a run with constraints and generate ideas:
  - `dialectica run ideas --constraints constraints/quantum_basic.md --count 10`
  - Add inline constraints: `--constraints-text "Extra bounds here"`
  - Or pipe from STDIN: `--constraints-stdin` (e.g., `cat extra.md | dialectica run ideas --constraints constraints/quantum_basic.md --constraints-stdin`)
  - Override field/domain: `--field compsci`, `--domain domain_compsci`
  - Creates a new run folder with `kickoff_prompt.md` and `ideas_gpt5.md`.

- Score all ideas (single-run flow):
  - `dialectica run score`
  - Produces `ratings.json` (ratings_v1 combined). Any invalid JSON/timeout aborts and writes status + error artifacts.

- Select an idea (interactive):
  - `dialectica select`
  - Presents ideas with both scores; writes `selected_idea.md`.

- Start drafting loop:
  - `dialectica draft --ask-to-continue`
  - GPT5 writes the first draft (`round_01_gpt5.md`).
  - Alternates critique+rewrite: Gemini (`round_02_gemini.md`), Grok4 (`round_03_grok4.md`), GPT5 (`round_04_gpt5.md`), etc.
  - Each critique begins with a judgment line: Reject, Major Revisions, Minor Revisions, or Publish.
  - If two models say Publish on the latest draft, the run finalizes even if the third requests revisions.
  - If a model outputs Publish, it records a judgment only (no rewritten draft for that round).
  - If `--ask-to-continue` is provided, the CLI prompts before each next round.

- Auto-run after selection (no pauses except selection):
  - `dialectica run all`
  - Executes ideas (ideas.json) → score (ratings.json) → select (interactive) → drafting to consensus.
  - Use `--auto-select --seed 42` to auto-pick highest total score.
  - Use `--from-ideas runs/compsci/ideas_gpt5.md` to seed from an existing ideas file.
  - To include extra constraints: `--constraints-text` or `--constraints-stdin`.

- Run all ideas:
  - `dialectica run all --all-ideas --constraints constraints/quantum_ibm_cost.md --name batch`
  - Generates ideas (ideas.json), then creates separate runs for each idea and drafts each to completion (no scoring).

- Start from an existing ideas file:
  - `dialectica run all --from-ideas runs/compsci/ideas_gpt5.md --auto-select --max-cycles 1000 --name from-compsci`
  - Copies the ideas file, scores them, auto-selects, and drafts without pauses.
  - Alternatively, pick a specific idea directly with `--idea 7`.

- Branch from previous ideas (reuse idea set, keep old papers intact):
  - `dialectica branch --from-run runs/<old_run> --idea 5 --name retry-idea-5 --start --max-cycles 1000`
  - Creates a new run, copies ideas from the old run, selects idea #5, and immediately starts drafting.
  - Use `--from-ideas <path>` to branch from a specific ideas file.

## Prompts & Constraints
- Place one or more Markdown files in `./constraints/`.
- Use `--constraints` to include one file or a comma-separated list.
- The constraints section is inserted verbatim into the prompts under “Constraints of Paper.”

## Environment & Config
- `.env` at repository root is the single source of configuration:
  - Provider keys and model names.
  - Optional flags in future (e.g., default constraints path).
- No additional config files are required.

## Artifacts & Naming
- Every meaningful step is saved as a Markdown file in the run folder with a timestamp-based name that includes the provider and round.
- Draft files always contain both the critique (judgment first) and the revised draft.
- The final agreed paper is saved as `paper.md`.

## Error Handling & Recovery
- If a provider call fails, the CLI shows a readable message and suggests `dialectica resume` (planned) to continue the same run from the last successful step.
- All steps are sequential; no concurrent calls are made.

## Privacy & Logging
- No special privacy handling. All prompts and responses are stored in Markdown locally under `./runs/`.

## FAQs
- Can I edit constraints mid-run? Yes; rerun from ideas if constraints changed materially.
- Can I skip Grok or Gemini? No; all three providers are required for this workflow.
- Can I export to PDF/LaTeX? Output is Markdown-only by design; you can convert externally if desired.

## Next Steps
- Confirm command names and flags.
- Implement CLI and provider adapters.
- Iterate this guide to match actual behavior as features land.
