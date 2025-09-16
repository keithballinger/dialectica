# Workflow — Dialectica

## Guiding Principles
- Predictable steps, small increments, visible artifacts.
- Stop for human choice at key gates (idea selection, optional round pauses).

## Task Workflow
1) Select task from `plan.md`.
2) Mark task In Progress in `status.md`.
3) Implement change (code/docs) in small scope.
4) Verify locally (lint/tests if present).
5) Update docs/user guide if behavior changed.
6) Mark task Done in `status.md` with date.

## Run Workflow (Product Behavior)
1) Kickoff: choose constraints via files under `./constraints/`, optional inline text, or stdin.
   - CLI: `--constraints <files> [,files]`, `--constraints-text "..."`, `--constraints-stdin`.
   - Field auto-detects from the primary constraints file (e.g., `compsci.md` → field `compsci`), or override with `--field`. Domain pack auto-maps from field, or override with `--domain`.
   - Either generate 10 ideas with the idea generator, or seed from an existing ideas file using `--from-ideas`.
2) Scoring (single-run flow): obtain ratings from configured raters (≥1) across configured rubric criteria (overall required). Save ratings files and `scoring_prompt.md`.
3) Selection (single-run flow):
   - Manual: `dialectica select` shows titles for picking.
   - Auto: sum across raters × criteria (weights in rubric); tie-break random (optional `--seed`).
4) Drafting: initial_drafter writes the first draft once; alternate critique+rewrite in critics order.
5) Judgment First: each critique starts with Reject/Major Revisions/Minor Revisions/Publish.
   - If Publish, record a judgment only (no new draft).
6) Termination: stop when at least two models output “Publish” for the latest draft (Publish tied to older drafts is ignored).
7) Batch mode: `--all-ideas` drafts a separate paper for every idea (no scoring), each in its own run.
8) Output: save `paper.md` (full), `paper_only.md` (body), `paper_annotated.md` (lay annotations), `consensus.md`, and all intermediate Markdown files under the run folder.

## Quality Gates
- Prompts include constraints verbatim from files plus any inline/STDIN segments.
- Field and domain pack resolved and persisted to `run.yml`.
- Ideas include Smart layperson sections and structured fields.
- Ratings prefer JSON with schema validation (ideas/ratings only); drafts remain Markdown.
- On invalid JSON, retry up to 2 times with guidance; then fail with artifacts for diagnosis.
- All provider responses captured to Markdown with timestamps and model identifiers.
- Runs are reproducible with `run.yml` snapshot, kickoff prompt, constraints sources, and selected idea.

## Definition of Done (per task)
- Behavior matches the user guide.
- Documentation updated.
- No broken commands or missing artifacts for supported flows.
