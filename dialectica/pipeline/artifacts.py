from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..errors import ArtifactError
from ..utils import timestamp


RUNS_DIR = Path("runs")


def ensure_runs_dir() -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    return RUNS_DIR


def _safe_name(name: str) -> str:
    allowed = [c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip()]
    base = "".join(allowed).strip("-") or "run"
    return base[:64]


def create_run_dir(name: str | None = None) -> Path:
    ensure_runs_dir()
    if name:
        base = _safe_name(name)
        candidate = RUNS_DIR / base
        if candidate.exists():
            candidate = RUNS_DIR / f"{base}-{timestamp()}"
        run_dir = candidate
    else:
        run_dir = RUNS_DIR / timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "drafts").mkdir(parents=True, exist_ok=True)
    (run_dir / "judgments").mkdir(parents=True, exist_ok=True)
    (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def latest_run_dir() -> Optional[Path]:
    if not RUNS_DIR.exists():
        return None
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


STATE_FILE = "state.txt"


def load_state(run_dir: Path) -> dict:
    state_path = run_dir / STATE_FILE
    state: dict[str, str] = {}
    if not state_path.exists():
        return state
    with open(state_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.strip().split(":", 1)
            state[k.strip()] = v.strip()
    return state


def save_state(run_dir: Path, **entries: str | int) -> None:
    state = load_state(run_dir)
    for k, v in entries.items():
        state[str(k)] = str(v)
    lines = [f"{k}: {v}" for k, v in sorted(state.items())]
    write_markdown(run_dir / STATE_FILE, "\n".join(lines) + "\n")


def append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)


def _yaml_dump(obj, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(_yaml_dump(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {v}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for it in obj:
            if isinstance(it, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_yaml_dump(it, indent + 1))
            else:
                lines.append(f"{sp}- {it}")
        return "\n".join(lines)
    return f"{sp}{obj}"


def write_run_snapshot(run_dir: Path, snapshot: dict) -> None:
    content = _yaml_dump(snapshot) + "\n"
    write_markdown(run_dir / "run.yml", content)
