from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..utils import load_dotenv
from ..pipeline import runner
from ..pipeline.artifacts import ensure_runs_dir, latest_run_dir, load_state, save_state


def cmd_run_ideas(args: argparse.Namespace) -> None:
    print("[phase] run ideas")
    constraints = [Path(p) for p in args.constraints.split(",")] if args.constraints else []
    for p in constraints:
        if not Path(p).exists():
            raise SystemExit(f"Constraint file not found: {p}")
    run_dir = runner.kickoff_run(constraints, name=args.name)
    runner.generate_ideas(run_dir, constraints, count=args.count)
    print(f"Ideas generated. Run folder: {run_dir}")


def cmd_run_score(args: argparse.Namespace) -> None:
    print("[phase] run score")
    run_dir = Path(args.run) if args.run else (latest_run_dir() or Path(""))
    if not run_dir or not run_dir.exists():
        raise SystemExit("Run directory not found. Use --run to specify.")
    constraints = _infer_constraints_from_kickoff(run_dir)
    runner.score_ideas(run_dir, constraints)
    print(f"Scoring complete. Run folder: {run_dir}")


def cmd_select(args: argparse.Namespace) -> None:
    print("[phase] select idea")
    run_dir = Path(args.run) if args.run else (latest_run_dir() or Path(""))
    if not run_dir or not run_dir.exists():
        raise SystemExit("Run directory not found. Use --run to specify.")
    ideas_text = (run_dir / "ideas_gpt5.md").read_text(encoding="utf-8")
    ideas = runner.parse_ideas_list(ideas_text)
    if not ideas:
        raise SystemExit("No ideas found to select from.")
    print("Select an idea:")
    for i, idea in enumerate(ideas, start=1):
        print(f"{i}. {idea}")
    while True:
        try:
            idx = int(input("Enter number: ").strip())
            if 1 <= idx <= len(ideas):
                break
            print("Out of range. Try again.")
        except ValueError:
            print("Invalid input. Enter an integer.")
    runner.save_selected_idea(run_dir, idx)
    print(f"Selected idea #{idx}.")


def cmd_draft(args: argparse.Namespace) -> None:
    print("[phase] draft to consensus")
    run_dir = Path(args.run) if args.run else (latest_run_dir() or Path(""))
    if not run_dir or not run_dir.exists():
        raise SystemExit("Run directory not found. Use --run to specify.")
    constraints = _infer_constraints_from_kickoff(run_dir)
    state = load_state(run_dir)
    if state.get("phase") != "drafting":
        # start fresh drafting from selected idea
        runner.first_draft(run_dir, constraints)
        state = load_state(run_dir)

    current_round = int(state.get("round", "1")) + 1
    next_provider = state.get("next", "gemini")
    cycles_remaining = int(args.max_cycles)

    print(f"Starting rounds at {current_round} → {next_provider}; max cycles: {cycles_remaining}")
    while cycles_remaining > 0:
        current_round, next_provider = runner.next_round(
            run_dir, constraints, current_round, next_provider
        )
        cycles_remaining -= 1
        print(f"[progress] cycles remaining: {cycles_remaining}")
        if runner.check_consensus_and_finalize(run_dir):
            print("Consensus reached. Paper finalized.")
            return
        if args.ask_to_continue:
            answer = input("Continue to next round? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                save_state(run_dir, phase="drafting", round=current_round - 1, next=next_provider)
                print("Paused.")
                return
    # reached cycle limit
    print("Reached max cycles. You can resume later.")
    save_state(run_dir, phase="drafting", round=current_round - 1, next=next_provider)


def cmd_run_all(args: argparse.Namespace) -> None:
    print("[phase] run all")
    constraints = [Path(p) for p in args.constraints.split(",")] if args.constraints else []
    for p in constraints:
        if not Path(p).exists():
            raise SystemExit(f"Constraint file not found: {p}")
    run_dir = runner.kickoff_run(constraints, name=args.name)
    runner.generate_ideas(run_dir, constraints, count=10)
    runner.score_ideas(run_dir, constraints)
    # Interactive selection
    cmd_select(argparse.Namespace(run=str(run_dir)))
    # Draft to consensus
    cmd_draft(argparse.Namespace(run=str(run_dir), ask_to_continue=args.ask_to_continue, max_cycles=args.max_cycles))
    print(f"All done. Run folder: {run_dir}")


def cmd_resume(args: argparse.Namespace) -> None:
    print("[phase] resume run")
    run_dir = Path(args.run) if args.run else (latest_run_dir() or Path(""))
    if not run_dir or not run_dir.exists():
        raise SystemExit("Run directory not found. Use --run to specify.")
    state = load_state(run_dir)
    phase = state.get("phase", "")
    print(f"Current phase: {phase or '(unknown)'}")
    if phase in {"kickoff", "ideas"}:
        # Resume from scoring
        constraints = _infer_constraints_from_kickoff(run_dir)
        runner.score_ideas(run_dir, constraints)
        cmd_select(argparse.Namespace(run=str(run_dir)))
        cmd_draft(argparse.Namespace(run=str(run_dir), ask_to_continue=args.ask_to_continue, max_cycles=args.max_cycles))
    elif phase == "scored":
        cmd_select(argparse.Namespace(run=str(run_dir)))
        cmd_draft(argparse.Namespace(run=str(run_dir), ask_to_continue=args.ask_to_continue, max_cycles=args.max_cycles))
    elif phase == "selected":
        cmd_draft(argparse.Namespace(run=str(run_dir), ask_to_continue=args.ask_to_continue, max_cycles=args.max_cycles))
    elif phase == "drafting":
        cmd_draft(argparse.Namespace(run=str(run_dir), ask_to_continue=args.ask_to_continue, max_cycles=args.max_cycles))
    elif phase == "complete":
        print("Run already complete.")
    else:
        print("Unknown state. Attempting to continue drafting.")
        cmd_draft(argparse.Namespace(run=str(run_dir), ask_to_continue=args.ask_to_continue, max_cycles=args.max_cycles))


def _infer_constraints_from_kickoff(run_dir: Path) -> list[Path]:
    state = load_state(run_dir)
    raw = state.get("constraints", "")
    paths = [Path(p) for p in raw.split("|") if p.strip()]
    if paths:
        return paths
    # Fallback to sources file if present
    src = run_dir / "constraints_sources.txt"
    if src.exists():
        items = [Path(ln.strip()) for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if items:
            return items
    # Last-resort: scan constraints dir
    cdir = Path("constraints")
    if cdir.exists():
        files = sorted([p for p in cdir.iterdir() if p.is_file() and p.suffix in {".md", ".txt"}])
        if files:
            return files
    return []


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ensure_runs_dir()
    parser = argparse.ArgumentParser(prog="dialectica", description="Dialectica CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run group
    p_run = sub.add_parser("run", help="Run high-level flows: ideas, score, all")
    sub_run = p_run.add_subparsers(dest="run_cmd", required=True)

    # run ideas
    p_ideas = sub_run.add_parser("ideas", help="Generate ideas with GPT5")
    p_ideas.add_argument("--constraints", type=str, required=True, help="Comma-separated constraint file paths")
    p_ideas.add_argument("--count", type=int, default=10)
    p_ideas.add_argument("--name", type=str, help="Optional name/label for the run directory")
    p_ideas.set_defaults(func=cmd_run_ideas)

    # run score
    p_score = sub_run.add_parser("score", help="Score ideas with Grok and Gemini")
    p_score.add_argument("--run", type=str, help="Run directory (default: latest)")
    p_score.set_defaults(func=cmd_run_score)

    # select
    p_select = sub.add_parser("select", help="Select an idea interactively")
    p_select.add_argument("--run", type=str, help="Run directory (default: latest)")
    p_select.set_defaults(func=cmd_select)

    # draft
    p_draft = sub.add_parser("draft", help="Run drafting loop to consensus")
    p_draft.add_argument("--run", type=str, help="Run directory (default: latest)")
    p_draft.add_argument("--ask-to-continue", action="store_true")
    p_draft.add_argument("--max-cycles", type=int, default=10)
    p_draft.set_defaults(func=cmd_draft)

    # run all
    p_all = sub_run.add_parser("all", help="Run ideas → score → select → draft")
    p_all.add_argument("--constraints", type=str, required=True, help="Comma-separated constraint file paths")
    p_all.add_argument("--ask-to-continue", action="store_true")
    p_all.add_argument("--max-cycles", type=int, default=10)
    p_all.add_argument("--name", type=str, help="Optional name/label for the run directory")
    p_all.set_defaults(func=cmd_run_all)

    # resume
    p_resume = sub.add_parser("resume", help="Resume a previous run")
    p_resume.add_argument("--run", type=str, help="Run directory (default: latest)")
    p_resume.add_argument("--ask-to-continue", action="store_true")
    p_resume.add_argument("--max-cycles", type=int, default=10)
    p_resume.set_defaults(func=cmd_resume)

    args = parser.parse_args(argv)
    args.func(args)
