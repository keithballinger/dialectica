# Status — Dialectica

As of: TBD

Project Status: In Progress

Current Phase: Phase 3/4 — Config + Constraints + Scoring

Current Task: Add config loader + run.yml snapshot; add inline/STDIN constraints and field/domain resolution; prep JSON schemas for ratings/ideas

Next Action: Implement CLI flags (`--constraints-text`, `--constraints-stdin`, `--field`, `--domain`); write `run.yml` on kickoff; update prompts to reflect domain/field

Notes:
- Models: GPT5, Gemini 2.5 Pro, Grok4 (keys via `.env`).
- Artifacts: Markdown under `./runs/<label-or-timestamp>/` including paper_only.md and paper_annotated.md.
