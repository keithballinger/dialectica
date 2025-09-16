from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
import re

Judgment = Literal["reject", "major revisions", "minor revisions", "publish"]


def parse_judgment(text: str) -> Judgment | None:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    norm = first_line.strip().lower()
    if re.search(r"^reject\b", norm):
        return "reject"
    if re.search(r"^major\s+revisions\b", norm):
        return "major revisions"
    if re.search(r"^minor\s+revisions\b", norm):
        return "minor revisions"
    if re.search(r"^publish\b", norm):
        return "publish"
    return None

def _extract_round(path: Path) -> int:
    # filenames like round_03_gpt5.md
    m = re.search(r"round_(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def latest_judgments_by_provider(run_dir: Path) -> dict[str, Judgment | None]:
    providers = {"gpt5": "gpt5", "gemini": "gemini", "grok4": "grok4"}
    out: dict[str, Judgment | None] = {p: None for p in providers}
    drafts_dir = run_dir / "drafts"
    judgments_dir = run_dir / "judgments"
    # Only count evaluations at or after the latest draft round
    latest_draft = max(drafts_dir.glob("round_*_*.md"), key=_extract_round, default=None)
    draft_round = _extract_round(latest_draft) if latest_draft else -1
    for key in providers.values():
        # collect candidate files: drafts and judgments for this provider
        cand = list(drafts_dir.glob(f"round_*_{key}.md")) + list(judgments_dir.glob(f"round_*_{key}.md"))
        # filter to those at/after latest draft round (to avoid stale Publish on older drafts)
        cand = [c for c in cand if _extract_round(c) >= draft_round]
        if not cand:
            continue
        latest = max(cand, key=_extract_round)
        try:
            text = latest.read_text(encoding="utf-8")
        except Exception:
            continue
        out[key] = parse_judgment(text)
    return out


def all_publish(run_dir: Path) -> bool:
    j = latest_judgments_by_provider(run_dir)
    vals = [j.get(k) for k in ("gpt5", "gemini", "grok4")]
    return all(v == "publish" for v in vals)


def two_publish(run_dir: Path) -> bool:
    j = latest_judgments_by_provider(run_dir)
    vals = [j.get(k) for k in ("gpt5", "gemini", "grok4")]
    return sum(1 for v in vals if v == "publish") >= 2
