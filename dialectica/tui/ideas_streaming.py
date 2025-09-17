#!/usr/bin/env python3
"""Improved streaming screen that buffers and displays ideas in a list"""

import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from textual.app import App, ComposeResult
from textual.widgets import Label, Button, ListView, ListItem, Static, Header, Footer, ProgressBar, LoadingIndicator
from textual.containers import Vertical, Container, Horizontal, ScrollableContainer, Center
from textual.screen import Screen

from openai import OpenAI


class IdeasStreamingScreen(Screen):
    """Streaming screen that displays ideas as they complete"""

    CSS = """
    #container {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #header-section {
        height: 8;
        padding: 1;
        border-bottom: solid $primary;
    }

    #spinner-container {
        height: 3;
        align: center middle;
    }

    #ideas-container {
        height: 1fr;
        padding: 1;
    }

    #ideas-list {
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    .idea-item {
        padding: 1;
        margin: 0 0 1 0;
        border: tall $secondary;
        background: $boost;
    }

    ListItem {
        padding: 1;
        margin: 0 0 1 0;
        background: $boost;
        height: auto;
    }

    .idea-title {
        text-style: bold;
        color: $primary;
    }

    .idea-summary {
        margin: 1 0 0 2;
        color: $text-muted;
    }

    #status-bar {
        height: 3;
        padding: 1;
        border-top: solid $primary;
        text-align: center;
    }

    #footer-section {
        height: 4;
        padding: 1;
    }

    .status-connecting {
        color: $warning;
    }

    .status-streaming {
        color: $primary;
    }

    .status-complete {
        color: $success;
    }

    .status-error {
        color: $error;
    }
    """

    def __init__(self, run_dir: Path, prompt: str):
        super().__init__()
        self.run_dir = run_dir
        self.prompt = prompt
        self.ideas: List[Dict[str, Any]] = []
        self.buffer = ""
        self.streaming = False
        self.full_text = []
        self.debug_file = run_dir / "streaming_debug.txt"

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="container"):
            # Header section
            with Vertical(id="header-section"):
                yield Static("[bold]🤖 GPT-5 Idea Generation[/bold]", id="title")
                yield Static("Target: 10 ideas", id="subtitle")
                yield ProgressBar(total=10, show_eta=False, show_percentage=True, id="progress")
                with Center(id="spinner-container"):
                    yield LoadingIndicator(id="loading-spinner")

            # Ideas display area
            with ScrollableContainer(id="ideas-container"):
                yield ListView(id="ideas-list")

            # Status bar
            yield Static("🔄 Ready to stream...", id="status-bar", classes="status-connecting")

            # Footer with button
            with Horizontal(id="footer-section"):
                # Start button is hidden since we auto-start
                yield Button("Save & Continue", id="continue", variant="success", disabled=True)

        yield Footer()

    def on_mount(self) -> None:
        """Auto-start streaming when screen mounts"""
        # Start streaming automatically
        self.start_streaming()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            event.button.disabled = True
            self.start_streaming()
        elif event.button.id == "continue":
            # Save the ideas and continue to next phase
            self.save_ideas_json()

    def start_streaming(self) -> None:
        """Start the streaming in a thread"""
        def stream_thread():
            try:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    self.app.call_from_thread(self.update_status, "❌ Missing OPENAI_API_KEY", "error")
                    return

                client = OpenAI(api_key=api_key)

                # Show spinner
                self.app.call_from_thread(self.show_spinner)

                # Update status
                self.app.call_from_thread(self.update_status, "🔄 Connecting to GPT-5...", "connecting")

                # Debug: Write prompt to file
                with open(self.debug_file, "w") as f:
                    f.write(f"=== PROMPT ===\n{self.prompt}\n\n=== STREAMING ===\n")

                # Stream the response
                with client.responses.stream(
                    model="gpt-5",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": self.prompt}]}],
                ) as stream:
                    self.app.call_from_thread(self.update_status, "✍️ Receiving ideas...", "streaming")

                    for event in stream:
                        if event.type == "response.output_text.delta":
                            delta = event.delta
                            self.buffer += delta
                            self.full_text.append(delta)

                            # Debug: Write to file
                            with open(self.debug_file, "a") as f:
                                f.write(delta)

                            # Skip any "Major Revisions" or critique text at the beginning
                            # Look for the first numbered idea to start parsing
                            if "1)" in self.buffer and not self.ideas:
                                # Clean buffer to start from first idea
                                idx = self.buffer.find("1)")
                                if idx > 0:
                                    self.buffer = self.buffer[idx:]

                            # Hide spinner after first content
                            if len(self.full_text) == 1:
                                self.app.call_from_thread(self.hide_spinner)

                            # Try to parse complete ideas from buffer
                            self.app.call_from_thread(self.parse_and_display_ideas)

                        elif event.type == "response.completed":
                            # Parse the complete text to get full details
                            self.app.call_from_thread(self.parse_full_ideas)

                            # If we still have no ideas, the response might be plain text
                            if len(self.ideas) == 0:
                                # Try to parse the complete text as plain response
                                self.app.call_from_thread(self.parse_plain_text_response)

                            self.app.call_from_thread(self.update_status,
                                f"✅ Complete! Generated {len(self.ideas)} ideas", "complete")

                            # Show completion message
                            self.app.call_from_thread(self.add_completion_message)

                            # Enable continue button only if we have ideas
                            if len(self.ideas) > 0:
                                self.app.call_from_thread(self.enable_continue_button)

                        elif event.type == "response.error":
                            self.app.call_from_thread(self.update_status,
                                f"❌ Error: {event.error}", "error")

            except Exception as e:
                self.app.call_from_thread(self.update_status, f"❌ Error: {str(e)}", "error")

        threading.Thread(target=stream_thread, daemon=True).start()

    def parse_and_display_ideas(self) -> None:
        """Parse the buffer for complete ideas and display them"""
        # First, always try to extract partial ideas
        self.try_extract_partial_ideas()

        # Also try to parse as complete JSON if we have what looks like a complete object
        if self.buffer.strip().startswith("{") and self.buffer.count("}") >= self.buffer.count("{"):
            try:
                # Attempt to parse the full buffer
                data = json.loads(self.buffer)
                if "ideas" in data:
                    # Update with any missing ideas
                    for idea in data["ideas"]:
                        # Limit to 10 ideas only
                        if not any(i.get("index") == idea.get("index") for i in self.ideas) and len(self.ideas) < 10:
                            self.ideas.append(idea)
                            self.update_progress()
                    self.update_ideas_display()
            except json.JSONDecodeError:
                pass  # Continue with partial parsing

    def parse_full_ideas(self) -> None:
        """Parse the complete buffer to extract full idea details"""
        import re

        full_text = "".join(self.full_text)

        # Debug: Write what we're parsing
        with open(self.debug_file, "a") as f:
            f.write(f"\n=== PARSING FULL IDEAS ===\n")
            f.write(f"Total text length: {len(full_text)}\n")

        # Pattern to match each complete idea block
        # This pattern captures everything between idea numbers
        idea_blocks_pattern = r'(\d+)\)\s+([^\n]+)\n((?:(?!\d+\)).*\n)*)'

        for match in re.finditer(idea_blocks_pattern, full_text, re.MULTILINE):
            index = int(match.group(1))
            title = match.group(2).strip()
            block = match.group(3)

            # Extract components from the block
            summary = ""
            layperson = ""

            # Find Summary line
            summary_match = re.search(r'Summary:\s*([^\n]+)', block)
            if summary_match:
                summary = summary_match.group(1).strip()

            # Find For a smart layperson (might be multiple lines)
            layperson_match = re.search(r'For a smart layperson:\s*([^\n]+(?:\n(?!(?:Falsification:|Novelty:))[^\n]+)*)', block)
            if layperson_match:
                layperson = layperson_match.group(1).strip().replace('\n', ' ')

            # Update existing idea with full details
            for idea in self.ideas:
                if idea['index'] == index:
                    idea['title'] = title  # Update with final title
                    idea['summary'] = summary
                    idea['layperson'] = layperson
                    idea['is_placeholder'] = False  # Clear placeholder flag

                    # Debug
                    with open(self.debug_file, "a") as f:
                        f.write(f"Updated idea {index} with details\n")
                        f.write(f"  Title: {title[:50]}...\n")
                        f.write(f"  Summary: {summary[:100]}...\n")
                        f.write(f"  Layperson: {layperson[:100]}...\n")
                    break

        # Refresh the display with full details
        self.update_ideas_display()

    def try_extract_partial_ideas(self) -> None:
        """Try to extract individual ideas using a two-phase approach"""
        import re

        # Phase 1: Just detect idea numbers to show placeholders
        idea_number_pattern = r'(\d+)\)\s+'

        for match in re.finditer(idea_number_pattern, self.buffer):
            index = int(match.group(1))

            # Only add ideas 1-10 that we haven't seen yet
            if (not any(i.get("index") == index for i in self.ideas) and
                index <= 10 and
                len(self.ideas) < 10):

                # Add placeholder idea
                idea = {
                    "index": index,
                    "title": "Loading...",  # Placeholder
                    "summary": "",
                    "layperson": "",
                    "is_placeholder": True
                }
                self.ideas.append(idea)
                self.update_progress()

                # Debug
                with open(self.debug_file, "a") as f:
                    f.write(f"\nDetected idea {index} - showing placeholder\n")

                # Update display immediately
                self.update_ideas_display()

        # Phase 2: When we have Summary lines, extract complete titles
        complete_title_pattern = r'(\d+)\)\s+([^\n]+)\nSummary:'

        for match in re.finditer(complete_title_pattern, self.buffer, re.MULTILINE):
            index = int(match.group(1))
            title = match.group(2).strip()

            # Find and update the existing placeholder
            for idea in self.ideas:
                if idea["index"] == index and idea.get("is_placeholder"):
                    idea["title"] = title
                    idea["is_placeholder"] = False

                    # Debug
                    with open(self.debug_file, "a") as f:
                        f.write(f"\nUpdated idea {index} with title: {title[:50]}...\n")

                    # Update display with real title
                    self.update_ideas_display()
                    break

    def update_ideas_display(self) -> None:
        """Update the list view with current ideas"""
        list_view = self.query_one("#ideas-list", ListView)
        list_view.clear()

        for idea in sorted(self.ideas, key=lambda x: x.get("index", 0)):
            # Create a container for each idea with better formatting
            index = idea.get('index', '?')
            title = idea.get('title', 'Untitled')
            summary = idea.get('summary', '')
            layperson = idea.get('layperson', '')
            is_placeholder = idea.get('is_placeholder', False)

            # Build the display text with proper formatting
            if is_placeholder:
                # Show loading state for placeholders
                idea_text = f"[bold cyan]Idea #{index}:[/bold cyan] [dim italic]⏳ Loading...[/dim italic]"
            else:
                # Show actual title
                idea_text = f"[bold cyan]Idea #{index}:[/bold cyan] {title}"

                # Only add details if we have them (after streaming is complete)
                if summary:
                    idea_text += f"\n[dim]Summary:[/dim] {summary}"

                if layperson:
                    idea_text += f"\n[dim italic]💡 {layperson}[/dim italic]"

            idea_widget = ListItem(Static(idea_text))
            list_view.append(idea_widget)

        # Debug: Log what we're displaying
        with open(self.debug_file, "a") as f:
            f.write(f"\n=== DISPLAY UPDATE ===\n")
            for idea in self.ideas:
                f.write(f"Idea {idea.get('index')}: {idea.get('title', 'No title')[:50]}\n")
                f.write(f"  Summary: {idea.get('summary', 'No summary')[:100]}\n")

        # Also update progress when we update display
        self.update_progress()

    def update_progress(self) -> None:
        """Update the progress bar"""
        try:
            progress = self.query_one("#progress", ProgressBar)
            # Update both progress value and advance to show movement
            current_count = min(len(self.ideas), 10)
            progress.update(progress=current_count)
        except Exception as e:
            pass  # Ignore errors updating progress

    def update_status(self, message: str, status_class: str) -> None:
        """Update the status bar"""
        status = self.query_one("#status-bar", Static)

        # Remove old classes
        status.remove_class("status-connecting")
        status.remove_class("status-streaming")
        status.remove_class("status-complete")
        status.remove_class("status-error")

        # Add new class
        status.add_class(f"status-{status_class}")
        status.update(message)

    def enable_continue_button(self) -> None:
        """Enable the continue button"""
        self.query_one("#continue", Button).disabled = False

    def save_ideas_json(self) -> None:
        """Save the ideas to a JSON file"""
        if self.ideas:
            ideas_data = {"ideas": self.ideas}
            ideas_file = self.run_dir / "ideas.json"
            ideas_file.write_text(json.dumps(ideas_data, indent=2), encoding="utf-8")

            # Also save the full text for compatibility
            full_text = "".join(self.full_text)
            if full_text:
                ideas_md = self.run_dir / "ideas_gpt5.md"
                ideas_md.write_text(full_text, encoding="utf-8")

            self.update_status(f"✅ Saved {len(self.ideas)} ideas!", "complete")

            # Continue to scoring screen
            from .run_dashboard import IdeasScoringScreen
            self.app.pop_screen()
            self.app.push_screen(IdeasScoringScreen(self.run_dir))

    def set_progress_complete(self) -> None:
        """Set progress bar to complete state"""
        try:
            progress = self.query_one("#progress", ProgressBar)
            # Set to maximum to show 100%
            progress.update(progress=10)
        except Exception:
            pass

    def add_completion_message(self) -> None:
        """Add a completion message to the list"""
        list_view = self.query_one("#ideas-list", ListView)
        completion_widget = ListItem(
            Static(
                f"\n[bold green]━━━ Generation Complete ━━━[/bold green]\n"
                f"[dim]Total ideas generated: {len(self.ideas)}[/dim]"
            )
        )
        list_view.append(completion_widget)

    def parse_plain_text_response(self) -> None:
        """Parse plain text response if JSON parsing failed"""
        full_text = "".join(self.full_text)
        if not full_text:
            return

        # Split by lines and look for numbered items
        lines = full_text.split('\n')
        current_index = 1

        for i, line in enumerate(lines):
            line = line.strip()
            # Look for lines that start with a number or contain theory/idea keywords
            if (line and
                (line[0].isdigit() or
                 any(keyword in line.lower() for keyword in ['theory', 'idea', 'hypothesis', 'model']))):

                # Extract title (first meaningful line)
                title = line.lstrip('0123456789. \t').strip()
                if title and len(title) > 10:  # Skip very short lines
                    # Look for summary in next lines
                    summary = ""
                    for j in range(i+1, min(i+3, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not next_line[0].isdigit():
                            summary += next_line + " "

                    idea = {
                        "index": current_index,
                        "title": title[:100],  # Limit title length
                        "summary": summary[:200] if summary else "No summary available"
                    }

                    if not any(existing["title"] == idea["title"] for existing in self.ideas):
                        self.ideas.append(idea)
                        self.update_progress()
                        current_index += 1

        if self.ideas:
            self.update_ideas_display()

    def show_spinner(self) -> None:
        """Show the loading spinner"""
        try:
            spinner = self.query_one("#loading-spinner", LoadingIndicator)
            spinner.display = True
        except Exception:
            pass

    def hide_spinner(self) -> None:
        """Hide the loading spinner"""
        try:
            spinner = self.query_one("#loading-spinner", LoadingIndicator)
            spinner.display = False
        except Exception:
            pass


class IdeasStreamingApp(App):
    """Test app wrapper"""

    def __init__(self, run_dir: Path, prompt: str):
        super().__init__()
        self.run_dir = run_dir
        self.prompt = prompt

    def on_mount(self) -> None:
        self.push_screen(IdeasStreamingScreen(self.run_dir, self.prompt))