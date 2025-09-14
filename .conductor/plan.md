# Dialectica Plan

Status legend: [ ] not started · [~] in progress · [x] done

## Vision
Build a Python CLI that orchestrates GPT5, Gemini 2.5 Pro, and Grok4 to invent a novel, testable scientific theory under specified constraints, and iteratively draft a short paper in Markdown until consensus of “publishable with minor revisions.” All artifacts are saved under `./runs/<timestamp>/` as Markdown.

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

- [ ] Phase 2: Provider Abstraction
  - [ ] Define provider interface (request/response, metadata, error types).
  - [ ] Implement GPT5 client adapter.
  - [ ] Implement Gemini 2.5 Pro client adapter.
  - [ ] Implement Grok4 client adapter.
  - [ ] Centralized rate-limit/backoff policies (sequential execution only).

- [ ] Phase 3: Constraints & Prompting
  - [ ] Support external constraints from `./constraints/*.md` files.
  - [ ] Template the “Constraints of Paper” section into prompts.
  - [ ] Store kickoff prompt in run folder.

- [ ] Phase 4: Idea Generation & Scoring
  - [ ] Generate 10 ideas (GPT5).
  - [ ] Score ideas (Grok4) with 1–10 and short rationale.
  - [ ] Score ideas (Gemini 2.5 Pro) with 1–10 and short rationale.
  - [ ] Save ideas and both scoring reports as Markdown.
  - [ ] Interactive idea selection; persist selection.

- [ ] Phase 5: Drafting Loop & Consensus
  - [ ] Initial draft by ChatGPT/GPT5 (once).
  - [ ] Alternating critique+rewrite cycles: Gemini → Grok → GPT5 → ...
  - [ ] Always start critique with judgment: reject/major revisions/minor revisions.
  - [ ] Option `--ask-to-continue` to pause after each round.
  - [ ] Terminate when all models agree: “publishable with minor revisions.”
  - [ ] Persist each round (critique + new draft) as separate Markdown files.

- [ ] Phase 6: Artifacts & Output
  - [ ] Save `paper.md` as the latest agreed draft.
  - [ ] Save `consensus.md` summarizing the final judgments.
  - [ ] Save full transcripts per model as Markdown.

- [ ] Phase 7: UX & Ergonomics
  - [ ] Clear progress logs to stdout.
  - [ ] Human-readable file naming in runs directory.
  - [ ] Helpful errors and recovery (retry, resume run).

- [ ] Phase 8: Documentation
  - [ ] Update user guide as features land.
  - [ ] Add architecture notes for contributors.
  - [ ] Keep `.conductor/status.md` current.

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
