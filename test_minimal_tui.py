#!/usr/bin/env python3
"""Minimal test to debug TUI text display"""

import time
import threading
from textual.app import App, ComposeResult
from textual.widgets import Static, Label, Button, RichLog
from textual.containers import Vertical, Container
from textual.screen import Screen


class MinimalTestScreen(Screen):
    """Minimal test screen"""

    CSS = """
    #container {
        width: 80%;
        height: 80%;
        border: thick $accent;
        padding: 2;
        margin: 2;
        background: $surface;
    }

    #display {
        height: 10;
        border: solid $primary;
        padding: 1;
        background: white;
        color: black;
    }

    #richlog {
        height: 10;
        border: solid $secondary;
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="container"):
            yield Label("Test Display Methods", style="bold")

            # Test 1: Static widget
            yield Label("1. Static Widget:")
            yield Static("Initial static text", id="display")

            # Test 2: RichLog widget (alternative)
            yield Label("\n2. RichLog Widget:")
            yield RichLog(id="richlog")

            # Buttons
            yield Button("Test Static Update", id="test-static")
            yield Button("Test RichLog Write", id="test-richlog")
            yield Button("Test Threaded Update", id="test-thread")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "test-static":
            # Direct update
            display = self.query_one("#display", Static)
            current = str(display.renderable)
            display.update(current + "\nDirect update works!")

        elif event.button.id == "test-richlog":
            # RichLog write
            log = self.query_one("#richlog", RichLog)
            log.write("RichLog write works!")

        elif event.button.id == "test-thread":
            # Threaded update
            def thread_update():
                for i in range(5):
                    time.sleep(0.5)
                    self.call_from_thread(self.update_static, f"\nThread update {i}")
                    self.call_from_thread(self.update_richlog, f"Thread log {i}")

            threading.Thread(target=thread_update, daemon=True).start()

    def update_static(self, text: str) -> None:
        display = self.query_one("#display", Static)
        current = str(display.renderable)
        display.update(current + text)

    def update_richlog(self, text: str) -> None:
        log = self.query_one("#richlog", RichLog)
        log.write(text)


class MinimalApp(App):
    """Minimal test app"""

    def on_mount(self) -> None:
        self.push_screen(MinimalTestScreen())


if __name__ == "__main__":
    print("Starting minimal TUI test...")
    print("This will test if text appears in the UI")
    print("-" * 50)
    app = MinimalApp()
    app.run()