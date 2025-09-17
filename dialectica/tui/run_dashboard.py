from __future__ import annotations

import os
import json
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from textual.app import App, ComposeResult
from textual.widgets import (
    Static, Header, Footer, Label, Button, ListView, ListItem,
    LoadingIndicator, DataTable, TextArea, TabbedContent, TabPane
)
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container, Center, Middle
from textual.screen import Screen
from textual.reactive import reactive
from textual import events
from textual.message import Message

from ..pipeline import runner
from ..pipeline.artifacts import load_state, save_state


class RunPhase(Enum):
    GENERATING = "generating"
    SCORING = "scoring"
    SELECTING = "selecting"
    DRAFTING = "drafting"
    COMPLETE = "complete"


class GeneratingScreen(Screen):
    """Screen shown while generating ideas"""

    CSS = """
    GeneratingScreen {
        align: center middle;
    }

    #gen-container {
        width: 60;
        height: 20;
        border: thick $accent;
        padding: 2;
        background: $surface;
    }

    LoadingIndicator {
        margin: 2 0;
    }

    .status-text {
        text-align: center;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="gen-container"):
            with Center():
                yield Label("[bold]Generating Ideas[/bold]", classes="status-text")
            with Center():
                yield LoadingIndicator()
            with Center():
                yield Label("GPT-5 is creating 10 novel theory ideas...", classes="status-text")
            with Center():
                yield Label("", id="gen-progress", classes="status-text")


class IdeaDetailScreen(Screen):
    """Screen for viewing idea details"""

    CSS = """
    IdeaDetailScreen {
        background: $surface;
    }

    #idea-container {
        width: 100%;
        height: 100%;
        padding: 2;
    }

    #idea-title {
        text-style: bold;
        margin: 0 0 1 0;
        border-bottom: solid $primary;
        padding: 1;
    }

    #idea-content {
        height: 1fr;
        padding: 1;
        border: solid $secondary;
        background: $panel;
    }

    .section-header {
        text-style: bold;
        color: $primary;
        margin: 1 0;
    }

    .section-content {
        margin: 0 0 2 0;
        padding-left: 2;
    }

    #back-button {
        dock: bottom;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, idea: Dict[str, Any]):
        super().__init__()
        self.idea = idea

    def compose(self) -> ComposeResult:
        with Container(id="idea-container"):
            yield Label(f"Idea #{self.idea.get('index', '?')}: {self.idea.get('title', 'Untitled')}", id="idea-title")

            # Create a scrollable container for the content
            with ScrollableContainer(id="idea-content"):
                # Summary section
                if "summary" in self.idea:
                    yield Label("📝 Summary", classes="section-header")
                    yield Static(self.idea['summary'], classes="section-content")

                # Layperson Explanation section
                if "layperson" in self.idea:
                    yield Label("👤 Layperson Explanation", classes="section-header")
                    yield Static(self.idea['layperson'], classes="section-content")

                # Falsification section
                if "falsification" in self.idea:
                    yield Label("🔬 Falsification", classes="section-header")
                    yield Static(self.idea['falsification'], classes="section-content")

                # IBM Cost Plan section
                if "ibm_cost_plan" in self.idea:
                    yield Label("💰 IBM Cost Plan", classes="section-header")
                    yield Static(self.idea['ibm_cost_plan'], classes="section-content")

                # Novelty section
                if "novelty" in self.idea:
                    yield Label("✨ Novelty", classes="section-header")
                    yield Static(self.idea['novelty'], classes="section-content")

            yield Button("Back to Ideas", id="back-button", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-button":
            self.app.pop_screen()


class ScoringReviewScreen(Screen):
    """Screen for viewing scoring details from a model"""

    CSS = """
    ScoringReviewScreen {
        background: $surface;
    }

    #review-container {
        width: 100%;
        height: 100%;
        padding: 2;
    }

    #review-title {
        text-style: bold;
        margin: 0 0 1 0;
        border-bottom: solid $primary;
        padding: 1;
    }

    DataTable {
        height: 1fr;
    }

    #back-button {
        dock: bottom;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, model: str, scores: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.scores = scores

    def compose(self) -> ComposeResult:
        with Container(id="review-container"):
            yield Label(f"Scoring Review - {self.model}", id="review-title")

            table = DataTable()
            yield table

            yield Button("Back to Ideas", id="back-button", variant="primary")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Idea #", "Title", "Overall", "Novelty", "Feasibility", "Impact")

        # Add scores for each idea
        if "items" in self.scores:
            for item in self.scores["items"]:
                idx = item.get("index", "?")
                criteria = item.get("criteria", {})
                overall = criteria.get("overall", {}).get("score", "-")
                novelty = criteria.get("novelty", {}).get("score", "-") if "novelty" in criteria else "-"
                feasibility = criteria.get("feasibility", {}).get("score", "-") if "feasibility" in criteria else "-"
                impact = criteria.get("impact", {}).get("score", "-") if "impact" in criteria else "-"

                # Get idea title if available
                title = f"Idea {idx}"

                table.add_row(str(idx), title, str(overall), str(novelty), str(feasibility), str(impact))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-button":
            self.app.pop_screen()


class IdeasScoringScreen(Screen):
    """Main screen showing ideas list and scoring progress"""

    CSS = """
    IdeasScoringScreen {
        background: $background;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    #ideas-panel {
        width: 60%;
        border-right: solid $primary;
    }

    #scoring-panel {
        width: 40%;
        padding: 1;
    }

    #ideas-list {
        height: 1fr;
        margin: 1;
        border: solid $secondary;
    }

    .idea-item {
        padding: 1;
        margin: 0;
    }

    .idea-item:hover {
        background: $boost;
    }

    #scoring-status {
        height: 20;
        margin: 1 0;
        padding: 2;
        border: solid $accent;
        background: $panel;
    }

    .model-section {
        margin: 0 0 2 0;
        padding: 1;
        border: tall $secondary;
        background: $boost;
    }

    .model-status {
        height: 2;
        margin: 0 0 1 0;
    }

    .model-label {
        width: 12;
        text-style: bold;
    }

    .spinner {
        width: 20;
        margin: 0 0 0 2;
    }

    .small-button {
        width: auto;
        height: 3;
        margin: 1 0 0 0;
    }

    .scoring-complete {
        color: $success;
    }

    .scoring-progress {
        color: $warning;
    }

    .scoring-pending {
        color: $text-muted;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir
        self.ideas = []
        self.scores = {
            "gpt5": None,
            "gemini": None,
            "grok4": None
        }
        self.scoring_started = {
            "gpt5": False,
            "gemini": False,
            "grok4": False
        }
        self.refresh_thread = None
        self.stop_refresh = threading.Event()
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-container"):
            # Left panel - Ideas list
            with Vertical(id="ideas-panel"):
                yield Label("[bold]Generated Ideas[/bold]", classes="panel-title")
                yield ListView(id="ideas-list")

            # Right panel - Scoring status
            with Vertical(id="scoring-panel"):
                yield Label("[bold]Scoring Progress[/bold]", classes="panel-title")

                with ScrollableContainer(id="scoring-status"):
                    # GPT-5 status
                    with Vertical(classes="model-section"):
                        with Horizontal(classes="model-status"):
                            yield Static("🤖 GPT-5", classes="model-label")
                            yield Static("", id="gpt5-spinner", classes="spinner")
                        yield Static("⏳ Waiting to start...", id="gpt5-status", classes="scoring-pending")
                        yield Button("📊 View Scores", id="gpt5-view", disabled=True, variant="primary", classes="small-button")

                    # Gemini status
                    with Vertical(classes="model-section"):
                        with Horizontal(classes="model-status"):
                            yield Static("✨ Gemini", classes="model-label")
                            yield Static("", id="gemini-spinner", classes="spinner")
                        yield Static("⏳ Waiting to start...", id="gemini-status", classes="scoring-pending")
                        yield Button("📊 View Scores", id="gemini-view", disabled=True, variant="primary", classes="small-button")

                    # Grok status
                    with Vertical(classes="model-section"):
                        with Horizontal(classes="model-status"):
                            yield Static("🔮 Grok-4", classes="model-label")
                            yield Static("", id="grok4-spinner", classes="spinner")
                        yield Static("⏳ Waiting to start...", id="grok4-status", classes="scoring-pending")
                        yield Button("📊 View Scores", id="grok4-view", disabled=True, variant="primary", classes="small-button")

                    yield Static("", id="scoring-summary")

                with Horizontal():
                    yield Button("Auto-Select Best", id="auto-select", disabled=True, variant="success")
                    yield Button("Continue to Draft", id="continue", disabled=True, variant="warning")

    def on_mount(self) -> None:
        self.load_ideas()
        self.start_scoring_monitor()

    def load_ideas(self) -> None:
        """Load generated ideas from file"""
        # Try different possible filenames
        possible_files = [
            self.run_dir / "ideas.json",
            self.run_dir / "ideas_gpt5.md",
            self.run_dir / "ideas_gpt5.json"
        ]

        ideas_file = None
        for f in possible_files:
            if f.exists():
                ideas_file = f
                break

        if ideas_file and ideas_file.exists():
            try:
                content = ideas_file.read_text(encoding="utf-8")
                # Try different parsing approaches
                data = None

                # First try: Look for JSON block in markdown
                if "```json" in content:
                    json_start = content.index("```json") + 7
                    json_end = content.index("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    data = json.loads(json_str)
                # Second try: Look for raw JSON starting with {
                elif content.strip().startswith("{"):
                    data = json.loads(content)
                # Third try: Look for ideas_v1 JSON response
                else:
                    # Search for JSON object in the content
                    import re
                    json_match = re.search(r'\{[\s\S]*"ideas"[\s\S]*\}', content)
                    if json_match:
                        data = json.loads(json_match.group())

                if data:
                    self.ideas = data.get("ideas", [])

                    # Populate list view
                    list_view = self.query_one("#ideas-list", ListView)
                    for idea in self.ideas:
                        idx = idea.get('index', '?')
                        title = idea.get('title', 'Untitled')
                        item = ListItem(
                            Label(f"[bold]{idx}.[/bold] {title}")
                        )
                        list_view.append(item)

                    if self.ideas:
                        self.notify(f"Loaded {len(self.ideas)} ideas", severity="info")
                else:
                    self.notify("Could not parse ideas format", severity="warning")

            except Exception as e:
                self.notify(f"Error loading ideas: {e}", severity="error")
                # Log the error for debugging
                import traceback
                print(f"Error loading ideas: {traceback.format_exc()}")

    def start_scoring_monitor(self) -> None:
        """Start monitoring thread for scoring progress"""
        def monitor_loop():
            while not self.stop_refresh.is_set():
                time.sleep(0.2)  # Faster refresh for smoother animation
                if not self.stop_refresh.is_set():
                    self.call_from_thread(self.check_scoring_status)

        self.refresh_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.refresh_thread.start()

    def check_scoring_status(self) -> None:
        """Check and update scoring status"""
        # Check for rating files
        ratings_dir = self.run_dir

        # Check if scoring has started
        state_file = ratings_dir / "state.txt"
        state = {}
        if state_file.exists():
            try:
                for line in state_file.read_text().splitlines():
                    if ":" in line:
                        key, val = line.split(":", 1)
                        state[key.strip()] = val.strip()
            except Exception:
                pass

        phase = state.get("phase", "ideas")

        # Animate spinner
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        spinner_char = self.spinner_frames[self.spinner_index]

        # Check GPT5 scores
        gpt5_file = ratings_dir / "ratings_gpt5.txt"
        if gpt5_file.exists() and self.scores["gpt5"] is None:
            try:
                content = gpt5_file.read_text(encoding="utf-8")
                self.scores["gpt5"] = json.loads(content)
                self.query_one("#gpt5-spinner", Static).update("")
                self.query_one("#gpt5-status", Static).update("[green]✅ Complete! Score calculated[/]")
                self.query_one("#gpt5-status", Static).add_class("scoring-complete")
                self.query_one("#gpt5-view", Button).disabled = False
            except Exception:
                self.query_one("#gpt5-spinner", Static).update("")
                self.query_one("#gpt5-status", Static).update("[red]❌ Error loading scores[/]")
        elif phase in ["scoring", "scored", "selected", "drafting", "complete"] and not self.scores["gpt5"]:
            self.scoring_started["gpt5"] = True
            self.query_one("#gpt5-spinner", Static).update(f"[cyan]{spinner_char}[/]")
            self.query_one("#gpt5-status", Static).update("[yellow]🔄 Analyzing ideas...[/]")
            self.query_one("#gpt5-status", Static).add_class("scoring-progress")

        # Check Gemini scores
        gemini_file = ratings_dir / "ratings_gemini.txt"
        if gemini_file.exists() and self.scores["gemini"] is None:
            try:
                content = gemini_file.read_text(encoding="utf-8")
                self.scores["gemini"] = json.loads(content)
                self.query_one("#gemini-spinner", Static).update("")
                self.query_one("#gemini-status", Static).update("[green]✅ Complete! Score calculated[/]")
                self.query_one("#gemini-status", Static).add_class("scoring-complete")
                self.query_one("#gemini-view", Button).disabled = False
            except Exception:
                self.query_one("#gemini-spinner", Static).update("")
                self.query_one("#gemini-status", Static).update("[red]❌ Error loading scores[/]")
        elif phase in ["scoring", "scored", "selected", "drafting", "complete"] and not self.scores["gemini"]:
            self.scoring_started["gemini"] = True
            self.query_one("#gemini-spinner", Static).update(f"[magenta]{spinner_char}[/]")
            self.query_one("#gemini-status", Static).update("[yellow]🔄 Evaluating theories...[/]")
            self.query_one("#gemini-status", Static).add_class("scoring-progress")

        # Check Grok scores
        grok_file = ratings_dir / "ratings_grok4.txt"
        if grok_file.exists() and self.scores["grok4"] is None:
            try:
                content = grok_file.read_text(encoding="utf-8")
                self.scores["grok4"] = json.loads(content)
                self.query_one("#grok4-spinner", Static).update("")
                self.query_one("#grok4-status", Static).update("[green]✅ Complete! Score calculated[/]")
                self.query_one("#grok4-status", Static).add_class("scoring-complete")
                self.query_one("#grok4-view", Button).disabled = False
            except Exception:
                self.query_one("#grok4-spinner", Static).update("")
                self.query_one("#grok4-status", Static).update("[red]❌ Error loading scores[/]")
        elif phase in ["scoring", "scored", "selected", "drafting", "complete"] and not self.scores["grok4"]:
            self.scoring_started["grok4"] = True
            self.query_one("#grok4-spinner", Static).update(f"[blue]{spinner_char}[/]")
            self.query_one("#grok4-status", Static).update("[yellow]🔄 Computing ratings...[/]")
            self.query_one("#grok4-status", Static).add_class("scoring-progress")

        # Check if all scoring complete
        if all(s is not None for s in self.scores.values()):
            self.query_one("#scoring-summary").update("[green bold]✅ All scoring complete![/]")
            self.query_one("#auto-select", Button).disabled = False
            self.query_one("#continue", Button).disabled = False
            self.stop_refresh.set()
        elif any(s is not None for s in self.scores.values()):
            completed = sum(1 for s in self.scores.values() if s is not None)
            self.query_one("#scoring-summary").update(f"[cyan]{completed}/3 models scored[/]")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle idea selection"""
        index = self.query_one("#ideas-list", ListView).index
        if 0 <= index < len(self.ideas):
            self.app.push_screen(IdeaDetailScreen(self.ideas[index]))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "gpt5-view" and self.scores["gpt5"]:
            self.app.push_screen(ScoringReviewScreen("GPT-5", self.scores["gpt5"]))
        elif button_id == "gemini-view" and self.scores["gemini"]:
            self.app.push_screen(ScoringReviewScreen("Gemini", self.scores["gemini"]))
        elif button_id == "grok4-view" and self.scores["grok4"]:
            self.app.push_screen(ScoringReviewScreen("Grok-4", self.scores["grok4"]))
        elif button_id == "auto-select":
            self.auto_select_idea()
        elif button_id == "continue":
            self.continue_to_draft()

    def auto_select_idea(self) -> None:
        """Auto-select the best idea based on scores"""
        self.notify("Auto-selecting best idea...", severity="info")
        # This would trigger the auto-selection logic

    def continue_to_draft(self) -> None:
        """Continue to drafting phase"""
        self.notify("Continuing to draft phase...", severity="info")
        # This would move to the drafting phase

    def on_unmount(self) -> None:
        self.stop_refresh.set()
        if self.refresh_thread:
            self.refresh_thread.join(timeout=1.0)


class RunDashboardApp(App):
    """Main application managing the run flow"""

    TITLE = "Dialectica Run Dashboard"
    CSS = """
    Screen {
        background: $background;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "back", "Back"),
    ]

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir
        self.phase = RunPhase.GENERATING

    def on_mount(self) -> None:
        # Start with generating screen
        self.push_screen(GeneratingScreen())
        # Start monitoring thread
        self.start_phase_monitor()

    def start_phase_monitor(self) -> None:
        """Monitor the run progress and switch screens"""
        def monitor():
            # Wait for ideas to be generated
            possible_files = [
                self.run_dir / "ideas.json",
                self.run_dir / "ideas_gpt5.md",
                self.run_dir / "ideas_gpt5.json"
            ]

            found = False
            while not found:
                for f in possible_files:
                    if f.exists():
                        found = True
                        break
                if not found:
                    time.sleep(1.0)

            # Switch to scoring screen
            self.call_from_thread(self.switch_to_scoring)

        threading.Thread(target=monitor, daemon=True).start()

    def switch_to_scoring(self) -> None:
        """Switch from generating to scoring screen"""
        self.pop_screen()  # Remove generating screen
        self.push_screen(IdeasScoringScreen(self.run_dir))
        self.phase = RunPhase.SCORING

    def action_back(self) -> None:
        """Go back to previous screen"""
        if len(self.screen_stack) > 1:
            self.pop_screen()


def run_dashboard(run_dir: Path) -> int:
    """Run the dashboard for a specific run"""
    app = RunDashboardApp(run_dir)
    app.run()
    return 0