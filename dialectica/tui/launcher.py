from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import json
import threading

from textual.app import App, ComposeResult
from textual.widgets import Static, Header, Footer, Label, Button, Input, TextArea, Log
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.screen import Screen

from ..pipeline import runner
from ..pipeline.artifacts import latest_run_dir, load_state, save_state


class LauncherScreen(Screen):
    TITLE = "Dialectica Launcher"
    CSS = """
    Screen {
        align: center middle;
    }

    #launcher-container {
        width: 90;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        padding: 2;
        background: $surface;
    }

    .form-group {
        height: auto;
        margin: 1 0;
    }

    Label {
        margin: 0 0 1 0;
        color: $text;
    }

    Input {
        margin: 0 0 1 0;
    }

    #constraint-inputs {
        height: auto;
        width: 100%;
    }

    #constraint-inputs Input {
        width: 30;
        margin: 0 1 0 0;
    }

    #add-constraint {
        width: 12;
        min-width: 12;
    }

    TextArea {
        height: 8;
        margin: 0 0 1 0;
    }

    #constraints-display {
        height: 10;
        border: solid $primary;
        padding: 1;
        margin: 1 0;
        background: $panel;
    }

    #button-row {
        dock: bottom;
        height: auto;
        margin: 2 0 0 0;
    }

    #button-row Button {
        width: 1fr;
        margin: 0 1;
    }

    #log-panel {
        height: 15;
        border: solid $secondary;
        margin: 2 0 0 0;
        display: none;
    }

    #log-panel.visible {
        display: block;
    }

    Log {
        overflow-x: scroll;
        overflow-y: scroll;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "quit", "Exit"),
        ("ctrl+enter", "submit", "Launch"),
        ("l", "toggle_log", "Toggle Log"),
    ]

    def __init__(self):
        super().__init__()
        # Pre-fill with test constraints
        self.constraints = {
            "testable": "Can validate with code"
        }
        self.running = False

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="launcher-container"):
            yield Label("[bold]Create New Dialectica Run[/bold]")

            with Vertical(classes="form-group"):
                yield Label("Run Name:")
                yield Input(placeholder="e.g., quantum_gravity_theory", id="run-name", value="compsci_auto")

            with Vertical(classes="form-group"):
                yield Label("Field:")
                yield Input(
                    placeholder="e.g., physics, compsci, biology",
                    id="field",
                    value="compsci"
                )

            with Vertical(classes="form-group"):
                yield Label("Domain Pack:")
                yield Input(
                    placeholder="e.g., domain_physics, domain_compsci",
                    id="domain",
                    value="domain_compsci"
                )

            with Vertical(classes="form-group"):
                yield Label("Constraints Overview:")
                yield TextArea(
                    "Research into using small LLMs for agentic coding",
                    id="overview"
                )

            with Vertical(classes="form-group"):
                yield Label("Add Constraint (Key : Value):")
                with Horizontal(id="constraint-inputs"):
                    yield Input(placeholder="constraint key", id="key")
                    yield Input(placeholder="constraint value", id="value")
                    yield Button("Add", id="add-constraint", variant="primary")

            yield Static("No constraints added yet", id="constraints-display")

            with Horizontal(id="button-row"):
                yield Button("Launch Run", id="launch", variant="success")
                yield Button("Clear", id="clear", variant="warning")
                yield Button("Exit", id="exit", variant="error")

            yield Log(id="log-panel", highlight=True)

    def on_mount(self) -> None:
        self.query_one("#run-name").focus()
        self.update_constraints_display()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "add-constraint":
            self.add_constraint()
        elif button_id == "launch":
            self.launch_run()
        elif button_id == "clear":
            self.clear_form()
        elif button_id == "exit":
            self.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields"""
        if event.input.id in ("key", "value"):
            # Add constraint when Enter pressed in constraint fields
            self.add_constraint()
            # Focus back to key field for quick entry
            self.query_one("#key", Input).focus()

    def add_constraint(self) -> None:
        key_input = self.query_one("#key", Input)
        value_input = self.query_one("#value", Input)

        key = key_input.value.strip()
        value = value_input.value.strip()

        if key and value:
            self.constraints[key] = value
            key_input.value = ""
            value_input.value = ""
            self.update_constraints_display()
            self.notify(f"Added constraint: {key}")

    def update_constraints_display(self) -> None:
        display = self.query_one("#constraints-display", Static)
        if self.constraints:
            lines = [f"[bold]{k}:[/bold] {v}" for k, v in self.constraints.items()]
            display.update("\n".join(lines))
        else:
            display.update("[dim]No constraints added yet[/dim]")

    def clear_form(self) -> None:
        self.query_one("#run-name", Input).value = ""
        self.query_one("#overview", TextArea).text = ""
        self.query_one("#key", Input).value = ""
        self.query_one("#value", Input).value = ""
        self.constraints = {}
        self.update_constraints_display()
        self.notify("Form cleared")

    def launch_run(self) -> None:
        if self.running:
            self.notify("A run is already in progress", severity="warning")
            return

        run_name = self.query_one("#run-name", Input).value.strip() or "dialectica_run"
        overview = self.query_one("#overview", TextArea).text.strip()
        field = self.query_one("#field", Input).value.strip() or "compsci"
        domain = self.query_one("#domain", Input).value.strip() or "domain_compsci"

        if not overview and not self.constraints:
            self.notify("Please add an overview or constraints", severity="error")
            return

        # Show log panel
        log = self.query_one("#log-panel", Log)
        log.add_class("visible")

        # Build constraints JSON
        constraints_obj = {
            "overview": overview,
            "constraints": self.constraints
        }

        # Write temporary constraints file
        constraints_dir = Path("constraints")
        constraints_dir.mkdir(exist_ok=True)
        tmp_path = constraints_dir / f"{run_name}_constraints.json"
        tmp_path.write_text(json.dumps(constraints_obj, indent=2), encoding="utf-8")

        try:
            # Kickoff run
            run_dir = runner.kickoff_run(
                [tmp_path],
                name=run_name,
                field=field,
                domain_pack=domain
            )

            # Switch to streaming generation screen
            # Use improved ideas streaming with buffering and list view
            from .ideas_streaming import IdeasStreamingScreen
            from .run_dashboard import IdeasScoringScreen

            # Build the prompt for idea generation
            from ..pipeline.prompts import compose_ideas_prompt
            from ..pipeline.artifacts import read_text

            constraints_text = read_text(run_dir / "constraints.json")
            prompt = compose_ideas_prompt(constraints_text, field=field)

            # Push improved streaming screen with list view
            self.app.push_screen(IdeasStreamingScreen(run_dir, prompt))

            # Start pipeline in background (for scoring and beyond)
            def run_pipeline():
                try:
                    # Wait a bit for ideas to be saved
                    import time
                    time.sleep(5)

                    # Check if ideas were generated
                    ideas_file = run_dir / "ideas.json"
                    while not ideas_file.exists():
                        time.sleep(1)

                    runner.score_ideas(run_dir, [tmp_path])
                    selected = runner.auto_select_by_sum(run_dir)
                    runner.first_draft(run_dir, [tmp_path])

                    state = load_state(run_dir)
                    current_round = int(state.get("round", "1")) + 1
                    next_provider = state.get("next", "gemini")
                    max_cycles = 10

                    for cycle in range(max_cycles):
                        current_round, next_provider = runner.next_round(
                            run_dir, [tmp_path], current_round, next_provider
                        )
                        if runner.check_consensus_and_finalize(run_dir):
                            break
                except Exception as e:
                    self.app.call_from_thread(lambda: self.notify(f"Pipeline error: {e}", severity="error"))

            threading.Thread(target=run_pipeline, daemon=True).start()

        except Exception as e:
            self.notify(f"Run failed: {e}", severity="error")

    def action_toggle_log(self) -> None:
        log = self.query_one("#log-panel", Log)
        if log.has_class("visible"):
            log.remove_class("visible")
        else:
            log.add_class("visible")

    def action_submit(self) -> None:
        if not self.running:
            self.launch_run()


class LauncherApp(App):
    """Main app that manages screens"""
    TITLE = "Dialectica"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(LauncherScreen())

    def action_back(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()


def run_launcher() -> int:
    app = LauncherApp()
    app.run()
    return 0