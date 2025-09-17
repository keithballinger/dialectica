#!/usr/bin/env python3
"""Test streaming functionality with mock data"""

import json
import time
from textual.app import App, ComposeResult
from textual.widgets import Button, RichLog, Static
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual import work


class MockStreamingScreen(Screen):
    """Mock streaming screen for testing without API"""

    CSS = """
    #container {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #title {
        height: 3;
        padding: 1;
        text-align: center;
        text-style: bold;
    }

    #log {
        height: 1fr;
        border: solid cyan;
        padding: 1;
        background: $panel;
    }

    #status {
        height: 3;
        padding: 1;
        text-align: center;
    }

    #buttons {
        height: 3;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.current_text = ""

    def compose(self) -> ComposeResult:
        with Container(id="container"):
            yield Static("[bold]Mock Streaming Test[/bold]", id="title")
            yield RichLog(id="log", highlight=True, markup=True, wrap=True, auto_scroll=True)
            yield Static("Ready", id="status")
            with Horizontal(id="buttons"):
                yield Button("Test Write", id="test", variant="success")
                yield Button("Normal Stream", id="normal", variant="primary")
                yield Button("Fast Stream", id="fast", variant="primary")
                yield Button("Slow Stream", id="slow", variant="primary")
                yield Button("Clear", id="clear", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        log = self.query_one("#log", RichLog)

        if button_id == "clear":
            self.clear_display()
        elif button_id == "test":
            # Simple synchronous test to verify RichLog works
            log.write("[cyan]Test write 1: Starting test...[/cyan]\n")
            log.write("Test write 2: Plain text\n")
            log.write("[green]Test write 3: Green text[/green]\n")
            log.write("[yellow]Test write 4: Yellow text[/yellow]\n")
            log.write("[bold red]Test write 5: Bold red[/bold red]\n")
            self.query_one("#status").update("[green]Test writes complete![/green]")
        elif button_id in ["normal", "fast", "slow"]:
            # Disable all stream buttons during streaming
            for btn_id in ["test", "normal", "fast", "slow"]:
                self.query_one(f"#{btn_id}", Button).disabled = True
            # Use the worker decorator for proper threading
            self.run_stream_worker(button_id)

    def clear_display(self) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()
        self.current_text = ""
        self.query_one("#status").update("Cleared")
        # Re-enable buttons
        for btn_id in ["test", "normal", "fast", "slow"]:
            self.query_one(f"#{btn_id}", Button).disabled = False

    @work(thread=True)
    def run_stream_worker(self, mode: str) -> None:
        """Run streaming in a worker thread"""
        log = self.query_one("#log", RichLog)
        status = self.query_one("#status", Static)

        # Clear display first
        self.app.call_from_thread(log.clear)
        self.current_text = ""

        try:
            # Write initial status - testing if writes show up
            self.app.call_from_thread(status.update, "[yellow]🔄 Connecting...[/yellow]")
            self.app.call_from_thread(log.write, "[cyan]Starting mock stream...[/cyan]\n")
            time.sleep(0.5)

            # Write more test content to verify it appears
            self.app.call_from_thread(log.write, "[green]Connected! Starting stream...[/green]\n")
            self.app.call_from_thread(log.write, "Preparing to stream JSON...\n\n")
            self.app.call_from_thread(status.update, "[yellow]✍️ Streaming JSON...[/yellow]")

            # Generate mock JSON response (smaller for testing)
            ideas_data = {
                "ideas": [
                    {
                        "index": 1,
                        "title": "Adaptive Neural Architecture Search",
                        "summary": "Evolutionary algorithms for neural network design"
                    },
                    {
                        "index": 2,
                        "title": "Quantum-Inspired Optimization",
                        "summary": "Classical algorithms using quantum principles"
                    },
                    {
                        "index": 3,
                        "title": "Self-Healing Systems via ML",
                        "summary": "ML models for system failure prediction"
                    }
                ]
            }

            json_str = json.dumps(ideas_data, indent=2)

            # Determine streaming speed
            if mode == "fast":
                delay = 0.01
                chunk_size = 20
            elif mode == "slow":
                delay = 0.1
                chunk_size = 3
            else:  # normal
                delay = 0.05
                chunk_size = 5

            # Stream the JSON character by character
            # Write the opening message first
            self.app.call_from_thread(log.write, "[dim]--- Begin JSON Stream ---[/dim]\n")

            for i in range(0, len(json_str), chunk_size):
                chunk = json_str[i:i+chunk_size]
                self.current_text += chunk
                # Write each chunk wrapped in color for visibility
                self.app.call_from_thread(log.write, f"[yellow]{chunk}[/yellow]")
                time.sleep(delay)

            self.app.call_from_thread(log.write, "\n[dim]--- End JSON Stream ---[/dim]")

            # Mark complete
            self.app.call_from_thread(status.update, "[green]✅ Stream complete![/green]")
            self.app.call_from_thread(log.write, "\n\n[green bold]Stream finished![/green bold]")

        except Exception as e:
            self.app.call_from_thread(status.update, f"[red]❌ Error: {str(e)}[/red]")
            self.app.call_from_thread(log.write, f"\n\n[red bold]Error: {str(e)}[/red bold]")

        finally:
            # Re-enable buttons
            self.app.call_from_thread(self.enable_buttons)

    def enable_buttons(self) -> None:
        """Re-enable all buttons"""
        for btn_id in ["test", "normal", "fast", "slow"]:
            self.query_one(f"#{btn_id}", Button).disabled = False


class MockStreamingApp(App):
    """Test app for streaming"""

    TITLE = "Mock Streaming Test"
    CSS = """
    Screen {
        align: center middle;
        background: $background;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(MockStreamingScreen())


if __name__ == "__main__":
    app = MockStreamingApp()
    app.run()