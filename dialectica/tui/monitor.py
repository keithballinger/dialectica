from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import threading
import time

from textual.app import App, ComposeResult
from textual.widgets import Static, Header, Footer, Label, Button, TabbedContent, TabPane, TextArea, Log, DataTable, Tree
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.reactive import reactive
from textual import events
from textual.message import Message

from ..pipeline.artifacts import load_state, save_state, latest_run_dir


class RunMonitor(App):
    TITLE = "Dialectica Run Monitor"
    CSS = """
    #status-panel {
        height: 8;
        border: thick $accent;
        padding: 1;
        background: $surface;
    }

    #progress-bar {
        height: 1;
        margin: 1 0;
    }

    .status-label {
        margin: 0 2 0 0;
    }

    #control-buttons {
        height: 3;
        margin: 1 0;
    }

    Button {
        margin: 0 1;
    }

    #tabs-container {
        border: solid $primary;
    }

    TextArea {
        scrollbar-size: 1 1;
    }

    #judgments-table {
        height: 100%;
    }

    #files-tree {
        height: 100%;
        border: solid $secondary;
        padding: 1;
    }

    .judgment-publish {
        color: $success;
    }

    .judgment-minor {
        color: $warning;
    }

    .judgment-major {
        color: $warning-darken-2;
    }

    .judgment-reject {
        color: $error;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("p", "pause_resume", "Pause/Resume"),
        ("escape", "quit", "Exit"),
    ]

    def __init__(self, run_dir: Optional[Path] = None):
        super().__init__()
        self.run_dir = Path(run_dir) if run_dir else latest_run_dir()
        if not self.run_dir or not self.run_dir.exists():
            raise ValueError("No valid run directory found")

        self.state = {}
        self.current_draft = ""
        self.judgments: List[Dict[str, str]] = []
        self.refresh_thread = None
        self.stop_refresh = threading.Event()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical():
            # Status Panel
            with Container(id="status-panel"):
                yield Label("[bold]Run Status[/bold]")
                with Horizontal():
                    yield Label("Phase:", classes="status-label")
                    yield Label("Unknown", id="phase-value")
                    yield Label("Round:", classes="status-label")
                    yield Label("--", id="round-value")
                    yield Label("Next:", classes="status-label")
                    yield Label("--", id="next-value")

                yield Static("", id="progress-bar")

                # Control buttons
                with Horizontal(id="control-buttons"):
                    yield Button("Pause", id="pause-btn", variant="warning")
                    yield Button("Resume", id="resume-btn", variant="success", disabled=True)
                    yield Button("Abort", id="abort-btn", variant="error")
                    yield Button("Refresh", id="refresh-btn", variant="primary")

            # Main content tabs
            with TabbedContent(id="tabs-container"):
                with TabPane("Current Draft", id="draft-tab"):
                    yield TextArea(
                        "",
                        id="draft-viewer",
                        read_only=True,
                        language="markdown"
                    )

                with TabPane("Judgments", id="judgments-tab"):
                    yield DataTable(id="judgments-table")

                with TabPane("Files", id="files-tab"):
                    yield Tree("Run Files", id="files-tree")

                with TabPane("Log", id="log-tab"):
                    yield Log(id="run-log", highlight=True)

        yield Footer()

    def on_mount(self) -> None:
        self.load_run_state()
        self.setup_judgments_table()
        self.populate_files_tree()
        self.start_auto_refresh()

    def load_run_state(self) -> None:
        try:
            self.state = load_state(self.run_dir)
            self.update_status_display()
            self.load_current_draft()
            self.load_judgments()
        except Exception as e:
            self.query_one("#run-log", Log).write(f"[red]Error loading state: {e}[/]")

    def update_status_display(self) -> None:
        phase = self.state.get("phase", "unknown")
        round_num = self.state.get("round", "--")
        next_provider = self.state.get("next", "--")

        self.query_one("#phase-value", Label).update(phase.title())
        self.query_one("#round-value", Label).update(str(round_num))
        self.query_one("#next-value", Label).update(next_provider)

        # Update progress bar
        progress_bar = self.query_one("#progress-bar", Static)
        if phase == "complete":
            progress_bar.update("[green]████████████████████████[/] Complete!")
        elif phase == "drafting":
            filled = min(int(round_num) * 2, 20) if isinstance(round_num, (int, str)) and str(round_num).isdigit() else 0
            bar = "█" * filled + "░" * (20 - filled)
            progress_bar.update(f"[cyan]{bar}[/] Drafting...")
        else:
            progress_bar.update("[dim]░░░░░░░░░░░░░░░░░░░░[/] Waiting...")

        # Update pause/resume button states
        pause_file = self.run_dir / "control_pause"
        if pause_file.exists():
            self.query_one("#pause-btn", Button).disabled = True
            self.query_one("#resume-btn", Button).disabled = False
        else:
            self.query_one("#pause-btn", Button).disabled = False
            self.query_one("#resume-btn", Button).disabled = True

    def load_current_draft(self) -> None:
        draft_files = sorted(self.run_dir.glob("draft_prompt_round_*.md"))
        if draft_files:
            latest_draft = draft_files[-1]
            try:
                content = latest_draft.read_text(encoding="utf-8")
                self.query_one("#draft-viewer", TextArea).text = content
                self.query_one("#run-log", Log).write(f"[green]Loaded {latest_draft.name}[/]")
            except Exception as e:
                self.query_one("#run-log", Log).write(f"[red]Error loading draft: {e}[/]")

    def load_judgments(self) -> None:
        judgments_dir = self.run_dir / "judgments"
        if not judgments_dir.exists():
            return

        self.judgments.clear()
        judgment_files = sorted(judgments_dir.glob("round_*_*.md"))

        for jf in judgment_files:
            try:
                content = jf.read_text(encoding="utf-8")
                # Extract judgment from first line
                first_line = content.split("\n")[0] if content else ""
                judgment = "Unknown"

                if "Publish" in first_line:
                    judgment = "Publish"
                elif "Minor Revisions" in first_line:
                    judgment = "Minor Revisions"
                elif "Major Revisions" in first_line:
                    judgment = "Major Revisions"
                elif "Reject" in first_line:
                    judgment = "Reject"

                # Parse filename for round and model
                parts = jf.stem.split("_")
                round_num = parts[1] if len(parts) > 1 else "?"
                model = parts[2] if len(parts) > 2 else "unknown"

                self.judgments.append({
                    "round": round_num,
                    "model": model,
                    "judgment": judgment,
                    "file": jf.name
                })
            except Exception:
                pass

        self.update_judgments_table()

    def setup_judgments_table(self) -> None:
        table = self.query_one("#judgments-table", DataTable)
        table.add_columns("Round", "Model", "Judgment")
        table.cursor_type = "row"

    def update_judgments_table(self) -> None:
        table = self.query_one("#judgments-table", DataTable)
        table.clear()

        for j in self.judgments:
            style = ""
            if j["judgment"] == "Publish":
                style = "judgment-publish"
            elif j["judgment"] == "Minor Revisions":
                style = "judgment-minor"
            elif j["judgment"] == "Major Revisions":
                style = "judgment-major"
            elif j["judgment"] == "Reject":
                style = "judgment-reject"

            table.add_row(j["round"], j["model"], j["judgment"])

    def populate_files_tree(self) -> None:
        tree = self.query_one("#files-tree", Tree)
        tree.clear()

        # Add key directories
        for subdir in ["drafts", "judgments"]:
            dir_path = self.run_dir / subdir
            if dir_path.exists():
                node = tree.root.add(f"📁 {subdir}/")
                for f in sorted(dir_path.glob("*.md")):
                    node.add_leaf(f"📄 {f.name}")

        # Add root level key files
        key_files = ["paper.md", "paper_only.md", "consensus.md", "run.yml", "state.txt"]
        for fname in key_files:
            fpath = self.run_dir / fname
            if fpath.exists():
                tree.root.add_leaf(f"📄 {fname}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "pause-btn":
            self.pause_run()
        elif button_id == "resume-btn":
            self.resume_run()
        elif button_id == "abort-btn":
            self.abort_run()
        elif button_id == "refresh-btn":
            self.refresh()

    def pause_run(self) -> None:
        pause_file = self.run_dir / "control_pause"
        pause_file.touch()
        self.notify("Run paused", severity="warning")
        self.query_one("#run-log", Log).write("[yellow]Run paused[/]")
        self.update_status_display()

    def resume_run(self) -> None:
        pause_file = self.run_dir / "control_pause"
        if pause_file.exists():
            pause_file.unlink()
        self.notify("Run resumed", severity="success")
        self.query_one("#run-log", Log).write("[green]Run resumed[/]")
        self.update_status_display()

    def abort_run(self) -> None:
        # Confirm first
        self.query_one("#run-log", Log).write("[red]Abort requested - create control_abort file to confirm[/]")
        abort_file = self.run_dir / "control_abort"
        abort_file.touch()
        self.notify("Run aborted", severity="error")

    def refresh(self) -> None:
        self.load_run_state()
        self.populate_files_tree()
        self.notify("Refreshed")

    def start_auto_refresh(self) -> None:
        def refresh_loop():
            while not self.stop_refresh.is_set():
                time.sleep(2.0)
                if not self.stop_refresh.is_set():
                    self.call_from_thread(self.load_run_state)

        self.refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self.refresh_thread.start()

    def on_unmount(self) -> None:
        self.stop_refresh.set()
        if self.refresh_thread:
            self.refresh_thread.join(timeout=1.0)

    def action_refresh(self) -> None:
        self.refresh()

    def action_pause_resume(self) -> None:
        pause_file = self.run_dir / "control_pause"
        if pause_file.exists():
            self.resume_run()
        else:
            self.pause_run()


def run_monitor(run_dir: Optional[Path] = None) -> int:
    try:
        app = RunMonitor(run_dir)
        app.run()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1