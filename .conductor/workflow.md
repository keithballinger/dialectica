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
1) Kickoff: choose constraints file(s) under `./constraints/` and generate 10 ideas with GPT5.
2) Scoring: get Grok4 and Gemini 2.5 Pro ratings (1–10 + short rationale) and save Markdown reports.
3) Selection: pause for human to pick the idea (interactive prompt).
4) Drafting: GPT5 writes initial draft once; alternate critique+rewrite Gemini → Grok → GPT5.
5) Judgment First: each critique starts with Reject/Major Revisions/Minor Revisions/Publish.
   - If a model outputs Publish, it records a judgment only (no new draft).
6) Pausing: if `--ask-to-continue`, stop after each round for confirmation.
7) Termination: stop when all three models output “Publish.”
8) Output: save `paper.md`, `consensus.md`, and all intermediate Markdown files under the run folder.

## Quality Gates
- Prompts include the constraints verbatim from chosen files.
- All provider responses captured to Markdown with timestamps and model identifiers.
- Runs are reproducible with saved kickoff prompt and selected idea.

## Definition of Done (per task)
- Behavior matches the user guide.
- Documentation updated.
- No broken commands or missing artifacts for supported flows.
