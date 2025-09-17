#!/usr/bin/env python3
"""Test script for the streaming TUI without requiring OpenAI API"""

import time
import json
import threading
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.widgets import Static, Label, Button
from textual.containers import Vertical, ScrollableContainer, Container
from textual.screen import Screen


class TestStreamingScreen(Screen):
    """Test screen that simulates GPT-5 streaming"""

    CSS = """
    TestStreamingScreen {
        align: center middle;
        background: $background;
    }

    #stream-container {
        width: 90%;
        height: 90%;
        border: thick $accent;
        padding: 2;
        background: $surface;
    }

    #stream-header {
        height: 3;
        border-bottom: solid $primary;
        margin: 0 0 1 0;
        padding: 1;
    }

    #stream-title {
        text-style: bold;
        text-align: center;
    }

    #stream-content {
        height: 1fr;
        padding: 1;
        border: solid $secondary;
        background: $panel;
        overflow-y: scroll;
    }

    #stream-display {
        padding: 1;
        color: $text;
    }

    #stream-footer {
        height: 3;
        margin: 1 0 0 0;
    }

    #stream-status {
        text-align: center;
        margin: 1 0;
    }

    .streaming {
        color: $warning;
    }

    .complete {
        color: $success;
    }
    """

    def __init__(self):
        super().__init__()
        self.content_buffer = []

    def compose(self) -> ComposeResult:
        with Container(id="stream-container"):
            with Vertical(id="stream-header"):
                yield Label("🤖 Test Streaming Display", id="stream-title")

            with ScrollableContainer(id="stream-content"):
                yield Static("", id="stream-display")

            with Vertical(id="stream-footer"):
                yield Static("⏳ Ready to start...", id="stream-status", classes="streaming")
                yield Button("Start Simulation", id="start-btn", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            self.start_simulation()
            event.button.disabled = True

    def start_simulation(self) -> None:
        """Start the simulated streaming"""
        def simulate_stream():
            try:
                # Update status
                self.call_from_thread(self.update_status, "🔄 Starting simulation...")
                self.call_from_thread(self.append_display, "[cyan]Beginning simulated stream...[/cyan]\n\n")
                time.sleep(1)

                # Simulate connecting
                self.call_from_thread(self.update_status, "📡 Simulating connection...")
                self.call_from_thread(self.append_display, "[green]Connected![/green]\n\n")
                time.sleep(0.5)

                # Start "streaming" JSON
                self.call_from_thread(self.update_status, "✍️ Streaming data...")

                # Simulated JSON ideas response
                json_response = {
                    "ideas": [
                        {
                            "index": 1,
                            "title": "Quantum Error Correction via Topological Encoding",
                            "summary": "A novel approach to quantum computing error correction...",
                            "layperson": "Like creating backup copies of quantum information...",
                            "falsification": "Test on IBM quantum computer with noise injection...",
                            "ibm_cost_plan": "6 months, $50k budget, 2 researchers",
                            "novelty": "First to combine topological and surface codes"
                        },
                        {
                            "index": 2,
                            "title": "Neural Network Pruning via Information Bottleneck",
                            "summary": "Optimize neural networks by identifying critical paths...",
                            "layperson": "Making AI smaller and faster by removing unnecessary parts...",
                            "falsification": "Benchmark on ImageNet with 90% pruning rate...",
                            "ibm_cost_plan": "3 months, $20k compute costs",
                            "novelty": "Information theory approach to model compression"
                        }
                    ]
                }

                # Convert to string and stream character by character
                json_str = json.dumps(json_response, indent=2)

                # Stream the JSON character by character
                for i, char in enumerate(json_str):
                    self.content_buffer.append(char)
                    self.call_from_thread(self.append_display, char)

                    # Variable speed to simulate network latency
                    if i % 10 == 0:
                        time.sleep(0.05)
                    elif char in '{[':
                        time.sleep(0.1)
                    else:
                        time.sleep(0.01)

                # Complete
                self.call_from_thread(self.update_status, "✅ Stream complete!", "complete")
                self.call_from_thread(self.append_display, "\n\n[green bold]Simulation finished![/green bold]")

            except Exception as e:
                self.call_from_thread(self.update_status, f"❌ Error: {e}", "failed")
                self.call_from_thread(self.append_display, f"\n\n[red]Error: {e}[/red]")

        # Start in background thread
        threading.Thread(target=simulate_stream, daemon=True).start()

    def update_status(self, message: str, status_class: str = "streaming") -> None:
        """Update the status message"""
        status = self.query_one("#stream-status", Static)

        # Update classes
        status.remove_class("streaming")
        status.remove_class("complete")
        status.remove_class("failed")
        status.add_class(status_class)

        status.update(message)

    def append_display(self, content: str) -> None:
        """Append content to the display"""
        display = self.query_one("#stream-display", Static)
        current = str(display.renderable) if display.renderable else ""
        display.update(current + content)

        # Auto-scroll
        container = self.query_one("#stream-content", ScrollableContainer)
        container.scroll_end(animate=False)


class TestStreamingApp(App):
    """Test app for streaming display"""

    TITLE = "Streaming Test"
    CSS = """
    Screen {
        background: $background;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "quit", "Exit"),
    ]

    def on_mount(self) -> None:
        self.push_screen(TestStreamingScreen())


if __name__ == "__main__":
    app = TestStreamingApp()
    app.run()