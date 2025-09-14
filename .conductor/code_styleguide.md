# Code Style Guide — Dialectica (Python)

## Core Principle
Clarity Over Cleverness: prioritize readability, explicit typing, and small, composable functions.

## Base
- Python 3.11+
- Follow PEP 8 and PEP 257 (docstrings).
- Use type hints everywhere; `mypy`-clean target (even if tooling added later).

## Structure
- Package layout: `dialectica/` with submodules `cli/`, `providers/`, `pipeline/`, `prompts/`, `artifacts/`.
- Single entrypoint script (e.g., `dialectica/__main__.py`).
- Avoid deep inheritance; prefer small dataclasses and composition.

## Naming
- Modules: `lower_snake_case.py`
- Classes: `CapWords`
- Functions/vars: `lower_snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Docstrings
- One-line summary, then details if needed.
- For public functions: arguments, returns, raises.

## Errors
- Define custom exceptions in `dialectica/errors.py`.
- Use precise exception types; no broad `except Exception`.

## CLI
- Prefer `argparse` or `typer`; short, explicit commands.
- No interactive TUI unless specified; default to simple prompts.

## I/O & Artifacts
- Read/write Markdown files with deterministic naming.
- No JSON artifacts as per project spec.

## Tests (if added later)
- Use pytest, record/replay for API calls, isolate side effects.
