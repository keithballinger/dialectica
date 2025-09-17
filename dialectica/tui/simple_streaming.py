from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.widgets import Label, Button, RichLog, Header, Footer
from textual.containers import Vertical, Container
from textual.screen import Screen

from openai import OpenAI


class SimpleStreamingScreen(Screen):
    """Simplified streaming screen that actually works"""

    CSS = """
    #container {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #log {
        width: 100%;
        height: 1fr;
        border: solid $primary;
        padding: 1;
        overflow-x: auto;
        overflow-y: scroll;
    }

    #status {
        height: 3;
        padding: 1;
    }
    """

    def __init__(self, run_dir: Path, prompt: str):
        super().__init__()
        self.run_dir = run_dir
        self.prompt = prompt
        self.full_text = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="container"):
            yield Label("[bold]GPT-5 Idea Generation[/bold]")
            yield RichLog(id="log", highlight=True, markup=True, auto_scroll=True, wrap=True, max_lines=10000)
            yield Label("Press 'Start' to begin streaming", id="status")
            yield Button("Start Streaming", id="start", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """Focus on start button when mounted"""
        self.query_one("#start", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            event.button.disabled = True
            self.start_streaming()

    def start_streaming(self) -> None:
        """Start the streaming in a thread"""
        log = self.query_one("#log", RichLog)
        status = self.query_one("#status", Label)

        # Clear log
        log.clear()
        log.write("[cyan]Starting GPT-5 streaming...[/cyan]\n")
        status.update("🔄 Connecting...")

        def stream_thread():
            try:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    self.app.call_from_thread(log.write, "[red]Error: Missing OPENAI_API_KEY[/red]")
                    return

                client = OpenAI(api_key=api_key)

                # Simple direct streaming
                self.app.call_from_thread(log.write, f"[dim]Prompt preview: {self.prompt[:100]}...[/dim]\n\n")
                self.app.call_from_thread(log.write, "[green]Connected! Streaming response:[/green]\n\n")

                with client.responses.stream(
                    model="gpt-5",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": self.prompt}]}],
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            # Write each delta directly
                            delta = event.delta
                            self.full_text.append(delta)
                            self.app.call_from_thread(log.write, delta)

                        elif event.type == "response.completed":
                            self.app.call_from_thread(log.write, "\n\n[green]✅ Stream complete![/green]")
                            self.app.call_from_thread(status.update, "✅ Complete!")

                            # Save the result
                            self.app.call_from_thread(self.save_ideas)

                        elif event.type == "response.error":
                            self.app.call_from_thread(log.write, f"\n[red]Error: {event.error}[/red]")
                            self.app.call_from_thread(status.update, "❌ Error occurred")

            except Exception as e:
                self.app.call_from_thread(log.write, f"\n[red]Exception: {e}[/red]")
                self.app.call_from_thread(status.update, f"❌ {str(e)}")

        # Start thread
        threading.Thread(target=stream_thread, daemon=True).start()

    def save_ideas(self) -> None:
        """Save the generated ideas"""
        full_text = "".join(self.full_text)

        # Try to parse as JSON
        try:
            if full_text.strip():
                ideas_json = json.loads(full_text)

                # Save files
                ideas_file = self.run_dir / "ideas.json"
                ideas_file.write_text(json.dumps(ideas_json, indent=2), encoding="utf-8")

                # Also save as markdown for compatibility
                ideas_md = self.run_dir / "ideas_gpt5.md"
                ideas_md.write_text(full_text, encoding="utf-8")

                log = self.query_one("#log", RichLog)
                log.write(f"\n[green]Saved to {ideas_file}[/green]")
        except Exception as e:
            log = self.query_one("#log", RichLog)
            log.write(f"\n[yellow]Warning: Could not parse JSON: {e}[/yellow]")


class SimpleStreamingApp(App):
    """Simple app wrapper"""

    def __init__(self, run_dir: Path, prompt: str):
        super().__init__()
        self.run_dir = run_dir
        self.prompt = prompt

    def on_mount(self) -> None:
        self.push_screen(SimpleStreamingScreen(self.run_dir, self.prompt))