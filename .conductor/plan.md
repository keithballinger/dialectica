# Dialectica Plan

Status legend: [ ] not started · [~] in progress · [x] done

## Vision
Build a Python CLI that orchestrates GPT5, Gemini 2.5 Pro, and Grok4 to invent a novel, testable scientific theory under specified constraints, and iteratively draft a short paper in Markdown until all models judge “Publish.” All artifacts are saved under `./runs/<timestamp>/` as Markdown.

## Phases

- [ ] Phase 0: Decisions & Setup
  - [ ] Confirm CLI name and namespace (default proposal: `dialectica`).
  - [ ] Confirm interactive UI style (simple prompts vs TUI).
  - [ ] Confirm package layout (plain `pip` project) and Python version (proposal: 3.11+).
  - [ ] Define minimal README and example `.env` template.

- [ ] Phase 1: Scaffolding
  - [ ] Initialize Python project structure.
  - [ ] Add CLI entrypoint and command structure.
  - [ ] Load configuration from `.env` (provider keys, model names).
  - [ ] Add config-first loader and `run.yml` snapshot.

- [ ] Phase 2: Provider Abstraction
  - [ ] Define provider interface (request/response, metadata, error types).
  - [ ] Implement GPT5 client adapter.
  - [ ] Implement Gemini 2.5 Pro client adapter.
  - [ ] Implement Grok4 client adapter.
  - [ ] Centralized rate-limit/backoff policies (sequential execution only).

- [ ] Phase 3: Constraints & Prompting
  - [ ] Support external constraints from `./constraints/*.md` files.
  - [ ] Add inline constraints via `--constraints-text` and `--constraints-stdin`.
  - [ ] Resolve `field` and domain pack from constraints (CLI overrides allowed).
  - [ ] Template the “Constraints of Paper” section into prompts.
  - [ ] Store kickoff prompt in run folder.
  - [ ] Add criteria JSON (overview + criteria map) and use it in prompts and TUI.

- [ ] Phase 4: Idea Generation & Scoring
  - [ ] Generate 10 ideas (GPT5) as JSON (ideas_v1). Abort on invalid JSON.
  - [ ] Score ideas (GPT5, Grok4, Gemini 2.5 Pro) as JSON (ratings_v1). Abort on invalid JSON.
  - [ ] Save a single `ideas.json` and `ratings.json` (combined); no Markdown ratings artifacts.
  - [ ] Interactive idea selection; persist selection.
  - [ ] Auto-selection: sum three scores; tie-break with seedable RNG.

- [ ] Phase 5: Drafting Loop & Consensus
  - [ ] Initial draft by ChatGPT/GPT5 (once).
  - [ ] Alternating critique+rewrite cycles: Gemini → Grok → GPT5 → ...
  - [ ] Always start critique with judgment: reject/major revisions/minor revisions/publish.
  - [ ] Option `--ask-to-continue` to pause after each round.
  - [ ] Terminate when ≥2 models output: “Publish.”
  - [ ] Persist each round (critique + new draft) as separate Markdown files.
  - [ ] Write `paper_only.md` and `paper_annotated.md` on finalize.

- [ ] Phase 6: Artifacts & Output
  - [ ] Save `paper.md` as the latest agreed draft.
  - [ ] Save `paper_only.md` (body) and `paper_annotated.md` (lay notes).
  - [ ] Save `consensus.md` summarizing the final judgments.
  - [ ] Save full transcripts per model as Markdown.

- [ ] Phase 7: UX & Ergonomics
  - [ ] TUI main screen revamp: ideas list with scores, idea details with criteria descriptions, timeline of drafts/ratings, pretty-rendered Markdown drafts.
  - [ ] Clear progress logs to stdout.
  - [ ] Human-readable file naming in runs directory.
  - [ ] Fail-fast on invalid JSON; show status and logs in TUI.

- [ ] Phase 8: Documentation
  - [ ] Update user guide as features land.
  - [ ] Add architecture notes for contributors.
  - [ ] Keep `.conductor/status.md` current.
  - [ ] Document config usage, `run.yml`, field/domain resolution.

## Non-Goals
- No budgets or cost enforcement.
- No concurrency; strictly sequential calls.
- No JSON artifacts; Markdown only.
- No telemetry.
- No IBM Quantum API integration; constraints only by prompt.

## Definition of Done
- CLI runs on macOS via git clone and Python install.
- Commands cover idea generation, scoring, selection, drafting loop, and termination on consensus.
- All artifacts saved under `./runs/<timestamp>/` as Markdown.
- Providers abstracted with a clean interface and clear error handling.
- Documentation complete and consistent with behavior.
