#!/usr/bin/env python3
"""Debug RichLog streaming issue with call_from_thread"""

import time
import threading
from textual.app import App, ComposeResult
from textual.widgets import Static, Label, Button, RichLog
from textual.containers import Vertical, Container
from textual.screen import Screen


class DebugRichLogScreen(Screen):
    """Screen to debug RichLog streaming"""

    CSS = """
    #container {
        width: 90%;
        height: 90%;
        border: thick $accent;
        padding: 2;
        margin: 2;
    }

    .test-section {
        height: 8;
        border: solid $primary;
        padding: 1;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="container"):
            yield Label("RichLog Debugging Tests")

            # Test 1: Direct write (should work)
            yield Label("1. Direct RichLog write:")
            yield RichLog(id="log1", classes="test-section")

            # Test 2: Thread with call_from_thread (problem case)
            yield Label("2. Thread + call_from_thread:")
            yield RichLog(id="log2", classes="test-section")

            # Test 3: Thread with call_from_thread and refresh
            yield Label("3. Thread + call_from_thread + refresh:")
            yield RichLog(id="log3", classes="test-section")

            # Buttons
            yield Button("Test Direct", id="test-direct")
            yield Button("Test Thread", id="test-thread")
            yield Button("Test Thread + Refresh", id="test-thread-refresh")
            yield Button("Clear All", id="clear-all")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "test-direct":
            # Direct write - should work
            log = self.query_one("#log1", RichLog)
            log.write("[green]Direct write works![/green]")
            log.write("Some text", end="")
            log.write(" more text", end="")
            log.write(" final text\n")

        elif event.button.id == "test-thread":
            # Thread with call_from_thread - problematic
            def thread_func():
                for i in range(5):
                    time.sleep(0.2)
                    self.call_from_thread(self.write_to_log2, f"Chunk {i}", end="")
                self.call_from_thread(self.write_to_log2, "\n[green]Done![/green]")

            threading.Thread(target=thread_func, daemon=True).start()

        elif event.button.id == "test-thread-refresh":
            # Thread with call_from_thread and refresh
            def thread_func():
                for i in range(5):
                    time.sleep(0.2)
                    self.call_from_thread(self.write_to_log3_with_refresh, f"Chunk {i}", end="")
                self.call_from_thread(self.write_to_log3_with_refresh, "\n[green]Done with refresh![/green]")

            threading.Thread(target=thread_func, daemon=True).start()

        elif event.button.id == "clear-all":
            self.query_one("#log1", RichLog).clear()
            self.query_one("#log2", RichLog).clear()
            self.query_one("#log3", RichLog).clear()

    def write_to_log2(self, text: str, end: str = "\n") -> None:
        """Write to log2 without refresh"""
        log = self.query_one("#log2", RichLog)
        log.write(text, end=end)

    def write_to_log3_with_refresh(self, text: str, end: str = "\n") -> None:
        """Write to log3 with refresh"""
        log = self.query_one("#log3", RichLog)
        log.write(text, end=end)
        log.refresh()  # Force refresh


class DebugApp(App):
    """Debug app"""

    def on_mount(self) -> None:
        self.push_screen(DebugRichLogScreen())


if __name__ == "__main__":
    print("Starting RichLog debug test...")
    print("This will test different ways of writing to RichLog")
    print("-" * 50)
    app = DebugApp()
    app.run()