from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .prompts import (
    compose_kickoff_prompt,
    compose_ideas_prompt,
    compose_scoring_prompt,
    compose_first_draft_prompt,
    compose_critique_rewrite_prompt,
)
from .artifacts import (
    create_run_dir,
    write_markdown,
    read_text,
    latest_run_dir,
    load_state,
    save_state,
)
from .consensus import parse_judgment, latest_judgments_by_provider, all_publish
from ..providers import gpt5, gemini, grok
from ..log import step, info, done


def read_constraints(paths: Sequence[Path]) -> str:
    parts: list[str] = []
    for p in paths:
        parts.append(f"From: {p}\n\n" + read_text(p).strip())
    return "\n\n".join(parts).strip()


def kickoff_run(constraints_files: Sequence[Path], name: str | None = None) -> Path:
    step("Kickoff run")
    info(f"Label: {name or '(none)'}")
    info(f"Constraints: {len(constraints_files)} file(s)")
    run_dir = create_run_dir(name=name)
    info(f"Run directory: {run_dir}")
    constraints_text = read_constraints(constraints_files)
    kickoff_prompt = compose_kickoff_prompt(constraints_text)
    write_markdown(run_dir / "kickoff_prompt.md", kickoff_prompt)
    # Persist constraints
    write_markdown(run_dir / "constraints.md", constraints_text)
    write_markdown(
        run_dir / "constraints_sources.txt",
        "\n".join(str(p) for p in constraints_files) + "\n",
    )
    save_state(
        run_dir,
        phase="kickoff",
        constraints="|".join(str(p) for p in constraints_files),
        name=name or "",
    )
    done("Kickoff artifacts written")
    return run_dir


def generate_ideas(run_dir: Path, constraints_files: Sequence[Path], count: int = 10) -> Path:
    step("Generate ideas")
    info("Reading constraints")
    constraints_text = read_constraints(constraints_files)
    info("Composing ideas prompt")
    prompt = compose_ideas_prompt(constraints_text)
    info("Calling GPT5 provider")
    provider = gpt5.get_provider()
    result = provider.complete(prompt)
    info("Saving ideas and prompt")
    write_markdown(run_dir / "ideas_gpt5.md", result)
    write_markdown(run_dir / "ideas_prompt.md", prompt)
    save_state(run_dir, phase="ideas")
    done("Ideas generated")
    return run_dir


def score_ideas(run_dir: Path, constraints_files: Sequence[Path]) -> None:
    step("Score ideas")
    info("Loading ideas")
    ideas_text = read_text(run_dir / "ideas_gpt5.md")
    info("Reading constraints")
    constraints_text = read_constraints(constraints_files)
    info("Composing scoring prompt")
    prompt = compose_scoring_prompt(constraints_text, ideas_text)

    info("Calling Grok4 for ratings")
    ratings_grok = grok.get_provider().complete(prompt)
    write_markdown(run_dir / "ratings_grok4.md", ratings_grok)

    info("Calling Gemini 2.5 Pro for ratings")
    ratings_gem = gemini.get_provider().complete(prompt)
    write_markdown(run_dir / "ratings_gemini.md", ratings_gem)

    info("Saving scoring prompt")
    write_markdown(run_dir / "scoring_prompt.md", prompt)
    save_state(run_dir, phase="scored")
    done("Scoring complete")


def parse_ideas_list(ideas_text: str) -> list[str]:
    """Parse numbered or bulleted ideas into a list of idea strings.

    Supported formats per line:
    - "1) Some idea ..."
    - "1. Some idea ..."
    - "- Some idea ..."
    The text after the enumerator is returned verbatim (trimmed).
    """
    import re

    lines = [ln.rstrip() for ln in ideas_text.splitlines() if ln.strip()]
    items: list[str] = []
    pat = re.compile(r"^\s*(\d+)[\.)]\s+(.*\S)\s*$")
    for ln in lines:
        m = pat.match(ln)
        if m:
            items.append(m.group(2).strip())
            continue
        if ln.lstrip().startswith("- "):
            items.append(ln.lstrip()[2:].strip())
    # Keep at most 10 ideas if more present
    return items[:10]


def save_selected_idea(run_dir: Path, index: int) -> None:
    ideas = parse_ideas_list(read_text(run_dir / "ideas_gpt5.md"))
    if not ideas:
        raise ValueError("No ideas found to select from.")
    if not (1 <= index <= len(ideas)):
        raise ValueError("Selected index out of range.")
    selected = ideas[index - 1]
    write_markdown(run_dir / "selected_idea.md", f"Selected idea #{index}:\n\n{selected}\n")
    save_state(run_dir, phase="selected", selected=index)
    done(f"Idea #{index} saved")


def first_draft(run_dir: Path, constraints_files: Sequence[Path]) -> Path:
    step("Initial draft (GPT5)")
    info("Reading constraints and selected idea")
    constraints_text = read_constraints(constraints_files)
    selected = read_text(run_dir / "selected_idea.md")
    info("Composing first-draft prompt")
    prompt = compose_first_draft_prompt(constraints_text, selected)
    info("Calling GPT5 for first draft")
    draft = gpt5.get_provider().complete(prompt)
    info("Saving draft and prompt")
    write_markdown(run_dir / "drafts" / f"round_01_gpt5.md", draft)
    write_markdown(run_dir / "draft_prompt_round_01.md", prompt)
    save_state(run_dir, phase="drafting", round=1, next="gemini")
    done("First draft saved")
    return run_dir


def next_round(run_dir: Path, constraints_files: Sequence[Path], round_num: int, provider_name: str) -> tuple[int, str]:
    step(f"Drafting round {round_num} → {provider_name}")
    info("Reading constraints and latest draft")
    constraints_text = read_constraints(constraints_files)
    latest = latest_draft_path(run_dir)
    latest_text = read_text(latest)
    info("Composing critique+rewrite prompt")
    prompt = compose_critique_rewrite_prompt(constraints_text, latest_text)

    if provider_name == "gemini":
        info("Calling Gemini")
        out = gemini.get_provider().complete(prompt)
        fname = f"round_{round_num:02d}_gemini.md"
        nxt = "grok"
        provider_key = "gemini"
    elif provider_name == "grok":
        info("Calling Grok4")
        out = grok.get_provider().complete(prompt)
        fname = f"round_{round_num:02d}_grok4.md"
        nxt = "gpt5"
        provider_key = "grok4"
    else:
        info("Calling GPT5")
        out = gpt5.get_provider().complete(prompt)
        fname = f"round_{round_num:02d}_gpt5.md"
        nxt = "gemini"
        provider_key = "gpt5"

    j = parse_judgment(out) or "minor revisions"
    if j == "publish":
        info("Judgment is Publish — recording judgment only (no new draft)")
        write_markdown(run_dir / "judgments" / fname, out)
        write_markdown(run_dir / f"draft_prompt_round_{round_num:02d}.md", prompt)
        save_state(run_dir, phase="drafting", round=round_num, next=nxt)
        done(f"Round {round_num} judgment saved; next → {nxt}")
    else:
        info("Saving draft and prompt")
        write_markdown(run_dir / "drafts" / fname, out)
        write_markdown(run_dir / f"draft_prompt_round_{round_num:02d}.md", prompt)
        save_state(run_dir, phase="drafting", round=round_num, next=nxt)
        done(f"Round {round_num} saved; next → {nxt}")
    return round_num + 1, nxt


def latest_draft_path(run_dir: Path) -> Path:
    drafts = sorted((run_dir / "drafts").glob("round_*_*.md"))
    if not drafts:
        raise FileNotFoundError("No drafts found.")
    return drafts[-1]


def read_latest_three_judgments(run_dir: Path) -> list[str | None]:
    # Deprecated for minor-revisions consensus; maintained for possible diagnostics
    drafts = sorted((run_dir / "drafts").glob("round_*_*.md"))[-3:]
    judgments = []
    for p in drafts:
        j = parse_judgment(read_text(p))
        judgments.append(j)
    return judgments


def check_consensus_and_finalize(run_dir: Path) -> bool:
    latest = latest_judgments_by_provider(run_dir)
    info("Judgments by provider: " + ", ".join([f"{k}={v or '?'}" for k, v in latest.items()]))
    if all_publish(run_dir):
        final_text = read_text(latest_draft_path(run_dir))
        write_markdown(run_dir / "paper.md", final_text)
        write_markdown(
            run_dir / "consensus.md",
            "Consensus: All models returned Publish on the latest draft.\n",
        )
        save_state(run_dir, phase="complete")
        done("Consensus reached (Publish) and paper finalized")
        return True
    return False
