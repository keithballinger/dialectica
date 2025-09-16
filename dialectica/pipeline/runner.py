from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .prompts import (
    compose_kickoff_prompt,
    compose_ideas_prompt,
    compose_scoring_prompt,
    compose_first_draft_prompt,
    compose_critique_rewrite_prompt,
    compose_annotation_prompt,
)
from .artifacts import (
    create_run_dir,
    write_markdown,
    read_text,
    latest_run_dir,
    load_state,
    save_state,
    write_run_snapshot,
)
from .consensus import parse_judgment, latest_judgments_by_provider, all_publish, two_publish
from ..providers import gpt5, gemini, grok
from ..log import step, info, done


def read_constraints(paths: Sequence[Path]) -> str:
    parts: list[str] = []
    for p in paths:
        parts.append(f"From: {p}\n\n" + read_text(p).strip())
    return "\n\n".join(parts).strip()


def create_run_from_existing_ideas(
    source_ideas_path: Path, constraints_files: Sequence[Path], name: str | None = None
) -> Path:
    """Create a new run initialized with existing ideas from another run.

    Writes kickoff prompt, constraints artifacts, and copies ideas into the new run.
    Returns the new run directory path.
    """
    step("Bootstrap new run from existing ideas")
    if not source_ideas_path.exists():
        raise FileNotFoundError(f"Ideas file not found: {source_ideas_path}")

    run_dir = create_run_dir(name=name)
    info(f"Run directory: {run_dir}")

    constraints_text = read_constraints(constraints_files)
    kickoff_prompt = compose_kickoff_prompt(constraints_text)
    write_markdown(run_dir / "kickoff_prompt.md", kickoff_prompt)
    write_markdown(run_dir / "constraints.md", constraints_text)
    write_markdown(
        run_dir / "constraints_sources.txt",
        "\n".join(str(p) for p in constraints_files) + "\n",
    )
    # Copy ideas
    ideas_text = read_text(source_ideas_path)
    write_markdown(run_dir / "ideas_gpt5.md", ideas_text)
    # Record provenance
    write_markdown(
        run_dir / "provenance.md",
        f"Branched from ideas: {source_ideas_path}\n",
    )
    save_state(
        run_dir,
        phase="ideas",
        constraints="|".join(str(p) for p in constraints_files),
        name=name or "",
        branched_from=str(source_ideas_path),
    )
    done("Run bootstrapped from ideas")
    return run_dir


def _infer_field_from_paths(paths: Sequence[Path]) -> str:
    return (paths[0].stem if paths else "general").lower()


def _pack_from_field(field: str) -> str:
    mapping = {
        "compsci": "domain_compsci",
        "quantum": "domain_quantum",
        "info_theory": "domain_info_theory",
        "information_theory": "domain_info_theory",
        "game_theory_econ": "domain_game_theory_econ",
        "econ": "domain_game_theory_econ",
    }
    return mapping.get(field.lower(), f"domain_{field.lower()}")


def kickoff_run(
    constraints_files: Sequence[Path],
    name: str | None = None,
    constraints_inline_text: str | None = None,
    constraints_stdin_text: str | None = None,
    field: str | None = None,
    domain_pack: str | None = None,
) -> Path:
    step("Kickoff run")
    info(f"Label: {name or '(none)'}")
    info(f"Constraints: {len(constraints_files)} file(s)")
    run_dir = create_run_dir(name=name)
    info(f"Run directory: {run_dir}")

    inline_segments: list[str] = []
    if constraints_inline_text:
        inline_segments.append(constraints_inline_text)
    if constraints_stdin_text:
        inline_segments.append(constraints_stdin_text)

    constraints_text = ("\n\n".join([
        f"From: {p}\n\n" + read_text(p).strip() for p in constraints_files
    ] + [f"From: inline#{i}\n\n" + seg.strip() for i, seg in enumerate(inline_segments, start=1)])).strip()

    kickoff_prompt = compose_kickoff_prompt(constraints_text)
    write_markdown(run_dir / "kickoff_prompt.md", kickoff_prompt)
    # Persist constraints
    write_markdown(run_dir / "constraints.md", constraints_text)
    write_markdown(
        run_dir / "constraints_sources.txt",
        ("\n".join(str(p) for p in constraints_files) + ("\n" if constraints_files else "")),
    )
    if inline_segments:
        write_markdown(run_dir / "constraints_inline.txt", "\n\n".join(inline_segments))

    resolved_field = field or _infer_field_from_paths(constraints_files)
    resolved_pack = domain_pack or _pack_from_field(resolved_field)

    save_state(
        run_dir,
        phase="kickoff",
        constraints="|".join(str(p) for p in constraints_files),
        name=name or "",
        field=resolved_field,
        domain=resolved_pack,
    )
    # Snapshot minimal run config
    snapshot = {
        "field": resolved_field,
        "prompts": {"pack": resolved_pack},
        "constraints": {
            "files": [str(p) for p in constraints_files],
            "inline_present": bool(inline_segments),
        },
    }
    write_run_snapshot(run_dir, snapshot)
    done("Kickoff artifacts written")
    return run_dir


def generate_ideas(run_dir: Path, constraints_files: Sequence[Path], count: int = 10) -> Path:
    step("Generate ideas")
    info("Reading constraints")
    constraints_text = read_text(run_dir / "constraints.md")
    info("Composing ideas prompt")
    field = load_state(run_dir).get("field", "general")
    prompt = compose_ideas_prompt(constraints_text, field=field)
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
    constraints_text = read_text(run_dir / "constraints.md")
    info("Composing scoring prompt (JSON schema)")
    rubric = [
        "overall",
        "novelty",
        "experimental_feasibility",
        "clarity",
        "theoretical_soundness",
        "computational_feasibility",
        "reproducibility",
        "falsifiability_rigor",
        "impact_potential",
        "scalability",
    ]
    prompt = compose_scoring_prompt(constraints_text, ideas_text, rubric)

    info("Calling GPT5 for ratings (with retries)")
    ok_gpt, ratings_gpt = _ratings_with_retries(gpt5.get_provider(), prompt)
    write_markdown(run_dir / "ratings_gpt5.md", ratings_gpt)
    if not ok_gpt:
        raise RuntimeError("GPT5 ratings invalid after retries; see ratings_gpt5.md for payload")

    info("Calling Grok4 for ratings (with retries)")
    ok_grok, ratings_grok = _ratings_with_retries(grok.get_provider(), prompt)
    write_markdown(run_dir / "ratings_grok4.md", ratings_grok)
    if not ok_grok:
        raise RuntimeError("Grok ratings invalid after retries; see ratings_grok4.md for payload")

    info("Calling Gemini 2.5 Pro for ratings (with retries)")
    ok_gem, ratings_gem = _ratings_with_retries(gemini.get_provider(), prompt)
    write_markdown(run_dir / "ratings_gemini.md", ratings_gem)
    if not ok_gem:
        raise RuntimeError("Gemini ratings invalid after retries; see ratings_gemini.md for payload")

    info("Saving scoring prompt")
    write_markdown(run_dir / "scoring_prompt.md", prompt)
    save_state(run_dir, phase="scored")
    done("Scoring complete")


def parse_ideas_list(ideas_text: str) -> list[str]:
    """Parse ideas as either single-line items or structured blocks.

    Structured blocks start with 'n) <Title>' on its own line, followed by
    lines until the next 'm) <...>' header or EOF. We return the full block
    including the header line. If no block headers are found, fall back to
    single-line parsing using enumerated or '-' lines.
    """
    import re

    lines = ideas_text.splitlines()
    header_pat = re.compile(r"^\s*(\d+)[\.)]\s+(.+)$")

    blocks: list[str] = []
    current: list[str] = []
    found_headers = False
    for ln in lines:
        if header_pat.match(ln):
            found_headers = True
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        if found_headers:
            current.append(ln.rstrip())
    if found_headers and current:
        blocks.append("\n".join(current).strip())

    if found_headers:
        return blocks[:10]

    # Fallback: simple single-line items
    items: list[str] = []
    single_pat = re.compile(r"^\s*(\d+)[\.)]\s+(.*\S)\s*$")
    for ln in (l.strip() for l in lines if l.strip()):
        m = single_pat.match(ln)
        if m:
            items.append(m.group(2).strip())
        elif ln.startswith("- "):
            items.append(ln[2:].strip())
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


def idea_titles_for_display(ideas_text: str) -> list[str]:
    """Return short titles suitable for displaying a numbered selection list."""
    import re

    ideas = parse_ideas_list(ideas_text)
    titles: list[str] = []
    header_pat = re.compile(r"^\s*(\d+)[\.)]\s+(.+)$")
    for block in ideas:
        first = block.splitlines()[0] if block else ""
        m = header_pat.match(first)
        if m:
            titles.append(m.group(2).strip())
        else:
            # fallback to first 80 chars
            titles.append((first or block).strip()[:80])
    return titles


def parse_ratings(text: str) -> list[int]:
    """Parse ratings in strict lines like 'n) Score: x/10 — ...' or 'n. Score: x/10 — ...'.
    Returns a list of length up to 10 where index i-1 holds score for idea i.
    Missing ideas default to 0.
    """
    import re

    scores = [0] * 10
    pat = re.compile(r"^\s*(\d+)[\.)]\s*Score:\s*(\d+)\s*/\s*10\b", re.IGNORECASE)
    for ln in (l.strip() for l in text.splitlines() if l.strip()):
        m = pat.match(ln)
        if not m:
            continue
        n = int(m.group(1))
        x = int(m.group(2))
        if 1 <= n <= 10:
            scores[n - 1] = x
    return scores


def auto_select_by_sum(run_dir: Path, seed: int | None = None) -> int:
    """Auto-select idea by summing GPT5+Grok+Gemini scores; break ties randomly.
    Returns the selected 1-based index and writes selected_idea.md.
    """
    import random

    if seed is not None:
        random.seed(seed)
    try:
        totals = total_scores(run_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError("Missing ratings files; run scoring first.") from e
    max_total = max(totals)
    candidates = [i + 1 for i, t in enumerate(totals) if t == max_total]
    chosen = random.choice(candidates)
    info(f"Auto-select totals: {totals}")
    info(f"Top total: {max_total}; candidates: {candidates}; chosen: {chosen}")
    save_selected_idea(run_dir, chosen)
    return chosen


def total_scores(run_dir: Path) -> list[int]:
    """Return total scores [len=10] by summing raters over rubric criteria.

    Supports JSON ratings_v1 (preferred). Falls back to single-line format.
    """
    files = [
        ("gpt5", run_dir / "ratings_gpt5.md"),
        ("grok4", run_dir / "ratings_grok4.md"),
        ("gemini", run_dir / "ratings_gemini.md"),
    ]
    weights = {
        "overall": 1.0,
        "novelty": 0.8,
        "experimental_feasibility": 0.8,
        "clarity": 0.6,
        "theoretical_soundness": 0.7,
        "computational_feasibility": 0.6,
        "reproducibility": 0.6,
        "falsifiability_rigor": 0.7,
        "impact_potential": 0.5,
        "scalability": 0.5,
    }
    totals = [0.0] * 10
    for _, path in files:
        if not path.exists():
            continue
        text = read_text(path)
        scores = _scores_from_ratings_text(text, weights)
        for i in range(min(10, len(scores))):
            totals[i] += scores[i]
    return [int(round(x)) for x in totals]


def indices_by_threshold(run_dir: Path, threshold: int | None) -> list[int]:
    """Return 1-based indices of ideas meeting threshold; if None, return all present ideas."""
    ideas = parse_ideas_list(read_text(run_dir / "ideas_gpt5.md"))
    n = min(len(ideas), 10)
    if threshold is None:
        return list(range(1, n + 1))
    totals = total_scores(run_dir)
    return [i for i in range(1, n + 1) if totals[i - 1] >= threshold]


def first_draft(run_dir: Path, constraints_files: Sequence[Path]) -> Path:
    step("Initial draft (GPT5)")
    info("Reading constraints and selected idea")
    constraints_text = read_text(run_dir / "constraints.md")
    selected = read_text(run_dir / "selected_idea.md")
    info("Composing first-draft prompt")
    field = load_state(run_dir).get("field", "general")
    prompt = compose_first_draft_prompt(constraints_text, selected, field=field)
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
    constraints_text = read_text(run_dir / "constraints.md")
    latest = latest_draft_path(run_dir)
    latest_text = read_text(latest)
    info("Composing critique+rewrite prompt")
    field = load_state(run_dir).get("field", "general")
    prompt = compose_critique_rewrite_prompt(constraints_text, latest_text, field=field)

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
    # Summary of current judgments
    jmap = latest_judgments_by_provider(run_dir)
    info("Judgments now: " + ", ".join([f"gpt5={jmap.get('gpt5') or '?'}", f"gemini={jmap.get('gemini') or '?'}", f"grok4={jmap.get('grok4') or '?'}"]))
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
    if all_publish(run_dir) or two_publish(run_dir):
        full = read_text(latest_draft_path(run_dir))
        only = extract_revised_markdown(full)
        write_markdown(run_dir / "paper.md", full)
        write_markdown(run_dir / "paper_only.md", only)
        write_markdown(
            run_dir / "consensus.md",
            "Consensus: At least two models returned Publish on the latest draft.\n",
        )
        save_state(run_dir, phase="complete")
        done("Consensus reached (two Publish) and paper finalized")
        # Create an annotated paper for a smart layperson
        try:
            step("Annotating final paper for smart layperson")
            annot = annotate_paper(only)
            write_markdown(run_dir / "paper_annotated.md", annot)
            done("Annotated paper written")
        except Exception as e:
            info(f"Annotation skipped due to error: {e}")
        return True
    return False


def extract_revised_markdown(text: str) -> str:
    """Extract the final paper content from a draft file that may include judgment/critique.

    Heuristics:
    - If a line equals 'Revised Draft' (case-insensitive, optional leading markdown header like '#', '##'),
      return content after that line.
    - Else, if starts with a judgment line, try to find the first markdown heading (e.g., '# ')
      after the critique section and return from there.
    - Else, return the whole text.
    """
    import re

    lines = text.splitlines()
    # Look for explicit Revised Draft heading
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(?:#+\s*)?revised\s+draft\s*$", ln.strip(), flags=re.IGNORECASE):
            return "\n".join(lines[i + 1:]).strip() + "\n"

    # If the first line is a judgment, try to find the first top-level header after some lines
    if lines and re.match(r"^(reject|major\s+revisions|minor\s+revisions|publish)\b", lines[0].strip(), flags=re.IGNORECASE):
        # find first header line '# ' later in the file
        for j in range(1, len(lines)):
            if lines[j].lstrip().startswith('#'):
                return "\n".join(lines[j:]).strip() + "\n"

    # Default: return full text
    return text if text.endswith("\n") else text + "\n"


def annotate_paper(paper_only_markdown: str) -> str:
    """Use GPT5 to produce an annotated version of the paper for a smart layperson."""
    from ..providers import gpt5

    prompt = compose_annotation_prompt(paper_only_markdown)
    return gpt5.get_provider().complete(prompt)


# ---- Ratings JSON helpers ----

def _strip_code_fences(s: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s.strip()


def _parse_ratings_json(text: str) -> list[dict]:
    raw = _strip_code_fences(text)
    obj = json.loads(raw)
    items = obj.get("items")
    if not isinstance(items, list) or len(items) < 1:
        raise ValueError("ratings_v1: items missing or not a list")
    return items


def _scores_from_ratings_text(text: str, weights: dict[str, float]) -> list[float]:
    try:
        items = _parse_ratings_json(text)
        scores = [0.0] * 10
        for it in items:
            idx = int(it.get("index", 0))
            crit = it.get("criteria", {})
            total = 0.0
            if isinstance(crit, dict):
                for name, w in weights.items():
                    entry = crit.get(name, {})
                    sc = entry.get("score") if isinstance(entry, dict) else None
                    if isinstance(sc, (int, float)):
                        total += float(sc) * float(w)
            if 1 <= idx <= 10:
                scores[idx - 1] = total
        return scores
    except Exception:
        line_scores = parse_ratings(text)
        return [float(x) for x in line_scores]


def _ratings_with_retries(provider, prompt_base: str, max_attempts: int = 3) -> tuple[bool, str]:
    guidance = "\nReturn valid JSON only, no extra text. Ensure it matches ratings_v1 and includes 'overall'."
    prompt = prompt_base
    last_out = ""
    for attempt in range(1, max_attempts + 1):
        out = provider.complete(prompt)
        last_out = out
        try:
            _parse_ratings_json(out)
            return True, out
        except Exception:
            if attempt == max_attempts:
                break
            prompt = prompt_base + "\n\n" + guidance
    return False, last_out
