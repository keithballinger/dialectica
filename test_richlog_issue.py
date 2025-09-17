#!/usr/bin/env python3
"""Simple test to reproduce RichLog streaming issue"""

import time
import threading
import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Button, RichLog, Static
from textual.containers import Vertical


class TestApp(App):
    CSS = """
    RichLog {
        border: solid white;
        height: 10;
        margin: 1;
    }

    Button {
        margin: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Testing RichLog streaming issue")
        yield RichLog(id="log")
        yield Button("Test Direct Write", id="direct")
        yield Button("Test Thread Write", id="thread")
        yield Button("Test Thread Write + Refresh", id="thread-refresh")
        yield Button("Clear", id="clear")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        log = self.query_one("#log", RichLog)

        if event.button.id == "direct":
            # This should work
            log.write("Direct: Hello")
            log.write(" World!", end="")
            log.write(" More text")

        elif event.button.id == "thread":
            # This might not work
            def thread_func():
                for i in range(3):
                    time.sleep(0.5)
                    self.call_from_thread(lambda i=i: log.write(f"Thread {i}", end=""))
                self.call_from_thread(lambda: log.write(" Done!"))

            threading.Thread(target=thread_func, daemon=True).start()

        elif event.button.id == "thread-refresh":
            # This should work with refresh
            def thread_func():
                for i in range(3):
                    time.sleep(0.5)
                    self.call_from_thread(self.write_and_refresh, f"RefreshThread {i}", end="")
                self.call_from_thread(self.write_and_refresh, " Done!")

            threading.Thread(target=thread_func, daemon=True).start()

        elif event.button.id == "clear":
            log.clear()

    def write_and_refresh(self, text, end="\n"):
        log = self.query_one("#log", RichLog)
        log.write(text, end=end)
        log.refresh()


if __name__ == "__main__":
    TestApp().run()