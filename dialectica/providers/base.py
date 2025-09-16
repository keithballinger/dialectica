from __future__ import annotations

from typing import Protocol


class Provider(Protocol):
    name: str
    model: str

    def complete(self, prompt: str) -> str:
        """Return a Markdown string result for the given prompt."""
        ...


class DryRunProvider:
    """Deterministic stub provider for offline/dry-run testing."""

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    def complete(self, prompt: str) -> str:
        header = f"Dry-Run Response — {self.name} ({self.model})\n\n"
        if "propose 10 ideas" in prompt.lower() or "10 ideas" in prompt.lower():
            ideas = "\n".join(
                [f"{i}. Novel theory idea #{i}: placeholder description." for i in range(1, 11)]
            )
            return header + ideas + "\n"
        if "rate each idea" in prompt.lower() or "rate on a scale" in prompt.lower():
            ratings = "\n".join(
                [f"{i}. Score: {5 + (i % 5)}/10 — rationale placeholder." for i in range(1, 11)]
            )
            return header + ratings + "\n"
        if "critique" in prompt.lower() and "draft" in prompt.lower():
            # Alternate judgments deterministically
            return (
                header
                + "Minor Revisions\n\n"
                + "Critique: placeholder critique focusing on constraints compliance.\n\n"
                + "Revised Draft\n\n"
                + "# Title\n\nAbstract...\n\nMethod...\n\nExperiments...\n\nConclusion...\n"
            )
        if "first draft" in prompt.lower() or "initial draft" in prompt.lower():
            return (
                header
                + "# Title\n\nAbstract...\n\nIntroduction...\n\nMethod...\n\nExperiments...\n\nConclusion...\n"
            )
        # Default fallback
        return header + "Placeholder response for prompt.\n"
