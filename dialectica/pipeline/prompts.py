from __future__ import annotations

from textwrap import dedent


SYSTEM_GUIDANCE = dedent(
    """
    You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
    If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.
    """
)


def compose_kickoff_prompt(constraints_text: str) -> str:
    return dedent(
        f"""
        Mission: Invent a new scientific theory from scratch.

        Constraints of Paper:
        {constraints_text}

        Process Overview:
        1) GPT5 proposes 10 ideas.
        2) Grok rates each idea 1–10 with short rationale.
        3) Gemini rates each idea 1–10 with short rationale.
        4) Human selects one idea to pursue.
        5) GPT5 drafts once; then iterative critique+rewrite: Gemini → Grok → GPT5 → ...
        6) Finish when all models agree: publishable with minor revisions.

        {SYSTEM_GUIDANCE}
        """
    ).strip()


def compose_ideas_prompt(constraints_text: str) -> str:
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Propose 10 ideas for novel, falsifiable scientific theories that satisfy the constraints.

        Constraints of Paper:
        {constraints_text}

        Output format: a numbered list from 1 to 10; one line per idea.
        """
    ).strip()


def compose_scoring_prompt(constraints_text: str, ideas_text: str) -> str:
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Rate each idea on a scale of 1–10 for novelty, falsifiability, and feasibility under the constraints. Provide a one-sentence rationale per idea.

        Constraints of Paper:
        {constraints_text}

        Ideas:
        {ideas_text}

        Output format: For each idea number, output: `<n>. Score: <x>/10 — <short rationale>`
        """
    ).strip()


def compose_first_draft_prompt(constraints_text: str, selected_idea: str) -> str:
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Write the first draft of a short scientific paper in Markdown based on the selected idea.

        Constraints of Paper:
        {constraints_text}

        Selected Idea:
        {selected_idea}

        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
        """
    ).strip()


def compose_critique_rewrite_prompt(constraints_text: str, latest_draft: str) -> str:
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        {constraints_text}

        Draft:
        {latest_draft}

        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — OMIT this section entirely if your judgment is Publish
        """
    ).strip()
