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
    stdin_text = None
    if getattr(args, "constraints_stdin", False):
        import sys
        stdin_text = sys.stdin.read()
    run_dir = runner.kickoff_run(
        constraints,
        name=args.name,
        constraints_inline_text=args.constraints_text,
        constraints_stdin_text=stdin_text,
        field=args.field,
        domain_pack=args.domain,
    )
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
    if getattr(args, "auto", False):
        chosen = runner.auto_select_by_sum(run_dir, seed=args.seed)
        print(f"Auto-selected idea #{chosen} by highest total score.")
        return
    ideas_text = (run_dir / "ideas_gpt5.md").read_text(encoding="utf-8")
    ideas = runner.parse_ideas_list(ideas_text)
    if not ideas:
        raise SystemExit("No ideas found to select from.")
    titles = runner.idea_titles_for_display(ideas_text)
    print("Select an idea:")
    for i, title in enumerate(titles, start=1):
        print(f"{i}. {title}")
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
    # Seed run with ideas (either from file or by generation)
    if getattr(args, "from_ideas", None):
        run_dir = runner.create_run_from_existing_ideas(Path(args.from_ideas), constraints, name=args.name)
    else:
        stdin_text = None
        if getattr(args, "constraints_stdin", False):
            import sys
            stdin_text = sys.stdin.read()
        run_dir = runner.kickoff_run(
            constraints,
            name=args.name,
            constraints_inline_text=args.constraints_text,
            constraints_stdin_text=stdin_text,
            field=args.field,
            domain_pack=args.domain,
        )
        runner.generate_ideas(run_dir, constraints, count=10)
    # Only score when not processing all ideas
    if not getattr(args, "all_ideas", False):
        runner.score_ideas(run_dir, constraints)
    # Selection and drafting logic
    if getattr(args, "all_ideas", False):
        # Iterate through all ideas; create separate child runs without scoring
        src_ideas = run_dir / "ideas_gpt5.md"
        ideas_text = src_ideas.read_text(encoding="utf-8")
        all_ideas = runner.parse_ideas_list(ideas_text)
        idxs = list(range(1, len(all_ideas) + 1))
        print(f"Processing all {len(idxs)} ideas")
        for i in idxs:
            name = f"{(args.name or 'batch')}-idea-{i}"
            child = runner.create_run_from_existing_ideas(src_ideas, constraints, name=name)
            runner.save_selected_idea(child, i)
            cmd_draft(argparse.Namespace(run=str(child), ask_to_continue=False, max_cycles=args.max_cycles))
        print("Batch processing complete.")
    else:
        # Single selection flow in this run
        if getattr(args, "idea", None):
            runner.save_selected_idea(run_dir, int(args.idea))
        else:
            if getattr(args, "auto_select", False):
                chosen = runner.auto_select_by_sum(run_dir, seed=args.seed)
                print(f"Auto-selected idea #{chosen} by highest total score.")
            else:
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
    p_ideas.add_argument("--constraints-text", type=str, help="Inline constraints string to append")
    p_ideas.add_argument("--constraints-stdin", action="store_true", help="Read additional constraints from STDIN")
    p_ideas.add_argument("--field", type=str, help="Override inferred field (e.g., compsci, quantum)")
    p_ideas.add_argument("--domain", type=str, help="Override domain pack (e.g., domain_compsci)")
    p_ideas.set_defaults(func=cmd_run_ideas)

    # run score
    p_score = sub_run.add_parser("score", help="Score ideas with Grok and Gemini")
    p_score.add_argument("--run", type=str, help="Run directory (default: latest)")
    p_score.set_defaults(func=cmd_run_score)

    # select
    p_select = sub.add_parser("select", help="Select an idea (interactive or auto)")
    p_select.add_argument("--run", type=str, help="Run directory (default: latest)")
    p_select.add_argument("--auto", action="store_true", help="Auto-select highest total score (Grok+Gemini+GPT5)")
    p_select.add_argument("--seed", type=int, help="Random seed for tie-breaking")
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
    p_all.add_argument("--auto-select", action="store_true", help="Auto-select highest total score (Grok+Gemini+GPT5)")
    p_all.add_argument("--seed", type=int, help="Random seed for auto-select tie-breaking")
    p_all.add_argument("--from-ideas", type=str, help="Path to an ideas_gpt5.md file to seed the run")
    p_all.add_argument("--idea", type=int, help="Select a specific idea (1-10) and draft immediately")
    p_all.add_argument("--all-ideas", action="store_true", help="Draft a separate paper for every idea; creates separate runs per idea")
    p_all.add_argument("--constraints-text", type=str, help="Inline constraints string to append")
    p_all.add_argument("--constraints-stdin", action="store_true", help="Read additional constraints from STDIN")
    p_all.add_argument("--field", type=str, help="Override inferred field (e.g., compsci, quantum)")
    p_all.add_argument("--domain", type=str, help="Override domain pack (e.g., domain_compsci)")
    p_all.set_defaults(func=cmd_run_all)

    # resume
    p_resume = sub.add_parser("resume", help="Resume a previous run")
    p_resume.add_argument("--run", type=str, help="Run directory (default: latest)")
    p_resume.add_argument("--ask-to-continue", action="store_true")
    p_resume.add_argument("--max-cycles", type=int, default=10)
    p_resume.set_defaults(func=cmd_resume)

    # branch from previous ideas
    def cmd_branch(args: argparse.Namespace) -> None:
        print("[phase] branch from ideas")
        if bool(args.from_run) == bool(args.from_ideas):
            raise SystemExit("Specify exactly one of --from-run or --from-ideas")
        if args.from_run:
            src_run = Path(args.from_run)
            if not src_run.exists():
                raise SystemExit("--from-run not found")
            ideas_path = src_run / "ideas_gpt5.md"
            if not ideas_path.exists():
                raise SystemExit("ideas_gpt5.md not found in source run")
            constraints = _infer_constraints_from_kickoff(src_run)
        else:
            ideas_path = Path(args.from_ideas)
            if not ideas_path.exists():
                raise SystemExit("--from-ideas file not found")
            # Use current default inference or require --constraints? We'll infer from current project
            constraints = _infer_constraints_from_kickoff(Path(args.run)) if args.run else _infer_constraints_from_kickoff(latest_run_dir() or Path(""))
            if not constraints:
                # fallback to constraints dir
                constraints = _infer_constraints_from_kickoff(Path("."))
        new_run = runner.create_run_from_existing_ideas(ideas_path, constraints, name=args.name)
        # Save selection and optionally start drafting
        runner.save_selected_idea(new_run, int(args.idea))
        if args.start:
            cmd_draft(argparse.Namespace(run=str(new_run), ask_to_continue=args.ask_to_continue, max_cycles=args.max_cycles))
        else:
            print(f"Branched run created at: {new_run}. Next: run 'dialectica draft --run {new_run}'.")

    p_branch = sub.add_parser("branch", help="Create a new run from existing ideas and start post-selection")
    p_branch.add_argument("--from-run", type=str, help="Source run directory with ideas_gpt5.md")
    p_branch.add_argument("--from-ideas", type=str, help="Path to an ideas_gpt5.md file")
    p_branch.add_argument("--idea", type=int, required=True, help="1-based index of idea to select")
    p_branch.add_argument("--name", type=str, help="Optional name/label for the new run directory")
    p_branch.add_argument("--start", action="store_true", help="Immediately start drafting in the new run")
    p_branch.add_argument("--ask-to-continue", action="store_true")
    p_branch.add_argument("--max-cycles", type=int, default=10)
    p_branch.set_defaults(func=cmd_branch)

    args = parser.parse_args(argv)
    args.func(args)
