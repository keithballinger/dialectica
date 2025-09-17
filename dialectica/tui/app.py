from __future__ import annotations

from pathlib import Path
from typing import Optional


def run_tui(run_dir: Optional[Path] = None) -> int:
    """Run the TUI launcher for Dialectica"""
    from .launcher import run_launcher
    return run_launcher()