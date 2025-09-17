from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from textual.app import App, ComposeResult
from textual.widgets import Static, Label, LoadingIndicator, TextArea, Button, RichLog
from textual.containers import Vertical, ScrollableContainer, Container, Center
from textual.screen import Screen
from textual.reactive import reactive

from openai import OpenAI


class StreamingGenerationScreen(Screen):
    """Screen that shows GPT-5 generating ideas in real-time"""

    CSS = """
    StreamingGenerationScreen {
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

    #stream-display {
        height: 1fr;
        padding: 1;
        border: solid $secondary;
        background: $panel;
        color: $text;
    }

    .idea-header {
        color: $primary;
        text-style: bold;
        margin: 1 0;
    }

    .idea-content {
        margin: 0 0 1 2;
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

    .failed {
        color: $error;
    }
    """

    def __init__(self, run_dir: Path, prompt: str):
        super().__init__()
        self.run_dir = run_dir
        self.prompt = prompt
        self.full_text = []
        self.streaming = False
        self.ideas_json = None

    def compose(self) -> ComposeResult:
        with Container(id="stream-container"):
            with Vertical(id="stream-header"):
                yield Label("🤖 GPT-5 Generating Ideas", id="stream-title")

            # Use RichLog instead of Static for better streaming display
            yield RichLog(id="stream-display", highlight=True, markup=True, wrap=True, auto_scroll=True)

            with Vertical(id="stream-footer"):
                yield Static("⏳ Initializing stream...", id="stream-status", classes="streaming")

    def on_mount(self) -> None:
        """Start streaming when the screen mounts"""
        self.start_streaming()

    def start_streaming(self) -> None:
        """Start the GPT-5 streaming thread"""
        def stream_thread():
            try:
                self.app.call_from_thread(self.update_status, "🔄 Connecting to GPT-5...")
                self.app.call_from_thread(self.update_display, "[dim]Initializing API connection...[/dim]\n")

                # Get API key
                import os
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT5_API_KEY")
                if not api_key:
                    error_msg = "❌ Missing API key"
                    self.app.call_from_thread(self.update_status, error_msg, "failed")
                    self.app.call_from_thread(self.update_display, f"\n{error_msg}")
                    return

                client = OpenAI(api_key=api_key)

                # Start streaming
                self.streaming = True
                self.app.call_from_thread(self.update_status, "✨ Sending request to GPT-5...", "streaming")
                self.app.call_from_thread(self.update_display, "[cyan]Sending request to model...[/cyan]\n")

                # Debug: show prompt length
                prompt_preview = self.prompt[:200] + "..." if len(self.prompt) > 200 else self.prompt
                self.app.call_from_thread(self.update_display, f"[dim]Prompt ({len(self.prompt)} chars): {prompt_preview}[/dim]\n\n")

                # Stream using exact API format from working example
                event_count = 0
                with client.responses.stream(
                    model="gpt-5",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": self.prompt}]}],
                ) as stream:
                    self.app.call_from_thread(self.update_status, "📡 Connected, waiting for response...", "streaming")
                    self.app.call_from_thread(self.update_display, "[green]Connected to GPT-5![/green]\n\n")

                    for event in stream:
                        event_count += 1

                        # Show event types for debugging
                        if event_count <= 5:
                            self.app.call_from_thread(self.append_display, f"[dim]Event #{event_count}: {event.type}[/dim]\n")

                        if event.type == "response.output_text.delta":
                            delta = event.delta
                            self.full_text.append(delta)
                            if event_count == 7:  # First text event based on your output
                                self.app.call_from_thread(self.update_status, "✍️ Receiving ideas...", "streaming")
                                self.app.call_from_thread(self.append_display, "\n[cyan]Response:[/cyan]\n")
                            # Stream the delta text
                            self.app.call_from_thread(self.append_display, delta)

                        elif event.type == "response.refusal.delta":
                            # Model refused to generate
                            self.streaming = False
                            self.app.call_from_thread(self.update_status, f"⚠️ Refused: {event.delta}", "failed")

                        elif event.type == "response.error":
                            self.streaming = False
                            self.app.call_from_thread(self.update_status, f"❌ Error: {event.error}", "failed")

                        elif event.type == "response.completed":
                            self.streaming = False
                            # Flush any remaining buffer
                            if hasattr(self, '_line_buffer') and self._line_buffer:
                                self.app.call_from_thread(self.append_display, "\n")
                            self.app.call_from_thread(self.update_status, "✅ Generation complete!", "complete")

                        else:
                            # Unknown / future event types
                            pass

                    # Get final structured object
                    final = stream.get_final_response()
                    # final.output_text is the assembled text convenience accessor
                    # final.output contains the rich, tool-aware structure if needed
                    if final:
                        final_text = final.output_text if hasattr(final, 'output_text') else "".join(self.full_text)
                        self.full_text = [final_text]
                        self.app.call_from_thread(self.finalize_ideas)

            except Exception as e:
                self.streaming = False
                import traceback
                error_details = str(e)

                # Check for specific error types
                if "401" in error_details:
                    error_msg = "❌ Authentication failed - check API key"
                elif "404" in error_details:
                    error_msg = "❌ Model not found - check model name"
                elif "429" in error_details:
                    error_msg = "❌ Rate limit exceeded"
                elif "500" in error_details or "502" in error_details or "503" in error_details:
                    error_msg = f"❌ Server error: {error_details}"
                else:
                    error_msg = f"❌ Error: {error_details}"

                self.app.call_from_thread(self.update_status, error_msg, "failed")
                self.app.call_from_thread(self.update_display, f"\n\n{error_msg}\n\n[dim]Full error:\n{traceback.format_exc()}[/dim]")

        threading.Thread(target=stream_thread, daemon=True).start()

    def update_display(self, new_content: str) -> None:
        """Replace the display with new content (clear and write)"""
        log = self.query_one("#stream-display", RichLog)
        log.clear()
        log.write(new_content)

    def append_display(self, delta_content: str) -> None:
        """Append content to the display"""
        log = self.query_one("#stream-display", RichLog)
        # For character-by-character streaming, accumulate until we have a full line
        # This prevents rendering issues with partial lines
        if not hasattr(self, '_line_buffer'):
            self._line_buffer = ""

        self._line_buffer += delta_content

        # Write complete lines immediately
        while "\n" in self._line_buffer:
            line_end = self._line_buffer.index("\n") + 1
            line = self._line_buffer[:line_end]
            self._line_buffer = self._line_buffer[line_end:]
            log.write(line)

        # For partial lines, update periodically or when buffer gets long
        if len(self._line_buffer) > 80 or (self._line_buffer and delta_content in ".,;:!?"):
            log.write(self._line_buffer, end="")
            self._line_buffer = ""

    def format_streaming_text(self, text: str) -> str:
        """Format the streaming text for display"""
        # Check if we have JSON structure
        if text.strip().startswith("{"):
            try:
                # Try to parse partial JSON for better formatting
                # This is just for display, so we can be lenient
                if '"ideas"' in text and text.count('[') > 0:
                    # We have at least the start of the ideas array
                    lines = []

                    # Extract ideas if possible
                    import re
                    idea_matches = re.findall(r'"index":\s*(\d+)[^}]*"title":\s*"([^"]*)"', text)

                    if idea_matches:
                        lines.append("[bold cyan]💡 Ideas Generated:[/bold cyan]\n")
                        for idx, title in idea_matches:
                            lines.append(f"[green]#{idx}[/green] {title}\n")

                    # Show raw JSON being built
                    lines.append("\n[dim]Raw JSON:[/dim]\n")
                    lines.append(f"[dim]{text[-500:] if len(text) > 500 else text}[/dim]")

                    return "".join(lines)
            except Exception:
                pass

        # Default: show the raw text with some formatting
        return f"[yellow]{text}[/yellow]"

    def update_status(self, message: str, status_class: str = "streaming") -> None:
        """Update the status message"""
        status = self.query_one("#stream-status", Static)

        # Remove old classes
        status.remove_class("streaming")
        status.remove_class("complete")
        status.remove_class("failed")

        # Add new class
        status.add_class(status_class)
        status.update(message)

    def finalize_ideas(self) -> None:
        """Process and save the generated ideas"""
        full_text = "".join(self.full_text)

        # Try to parse as JSON
        try:
            if full_text.strip().startswith("{"):
                self.ideas_json = json.loads(full_text)

                # Save to file
                ideas_file = self.run_dir / "ideas.json"
                ideas_file.write_text(json.dumps(self.ideas_json, indent=2), encoding="utf-8")

                # Also save as markdown for compatibility
                ideas_md_file = self.run_dir / "ideas_gpt5.md"
                ideas_md_file.write_text(full_text, encoding="utf-8")

                self.update_status(f"✅ Saved {len(self.ideas_json.get('ideas', []))} ideas!", "complete")
            else:
                self.update_status("⚠️ Invalid format received", "failed")
        except Exception as e:
            self.update_status(f"⚠️ Error parsing ideas: {str(e)}", "failed")

        # Add continue button
        self.mount(Button("Continue to Scoring", id="continue-btn", variant="success"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue-btn":
            # Switch to scoring screen
            from .run_dashboard import IdeasScoringScreen
            self.app.pop_screen()
            self.app.push_screen(IdeasScoringScreen(self.run_dir))