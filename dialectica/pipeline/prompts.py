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


def compose_ideas_prompt(constraints_text: str, field: str = "general") -> str:
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Propose 10 ideas for novel, falsifiable scientific theories that satisfy the constraints.
        Field: {field}

        Constraints of Paper:
        {constraints_text}

        Output format: EXACTLY 10 blocks in this format, no extra commentary:
        
        1) <Concise Title>
        Summary: <one concise sentence>
        For a smart layperson: <2–3 sentences explaining the idea accessibly>
        Falsification: <1–3 sentences with concrete steps>
        Novelty: <one sentence on why this is new>

        2) ... (repeat for 2 through 10)
        
        Use plain Markdown lines exactly as shown above.
        """
    ).strip()


def compose_scoring_prompt(constraints_text: str, ideas_text: str, rubric_criteria: list[str]) -> str:
    criteria_lines = "\n".join(f"- {c}" for c in rubric_criteria)
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Rate each idea strictly in JSON using the ratings_v1 schema. For each idea index (1–10), provide scores (1–10) for these criteria (overall is required):
        {criteria_lines}

        Constraints of Paper:
        {constraints_text}

        Ideas:
        {ideas_text}

        Output: Return valid JSON only, no extra text, matching ratings_v1:
        {{
          "items": [
            {{ "index": 1, "criteria": {{ "overall": {{"score": 7, "rationale": "..."}}, "novelty": {{"score": 8}}, "experimental_feasibility": {{"score": 6}} }} }},
            {{ "index": 2, "criteria": {{ "overall": {{"score": 8}} }} }}
          ]
        }}
        """
    ).strip()


def compose_first_draft_prompt(constraints_text: str, selected_idea: str, field: str = "general") -> str:
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Write the first draft of a short scientific paper in Markdown based on the selected idea.
        Field: {field}

        Constraints of Paper:
        {constraints_text}

        Selected Idea:
        {selected_idea}

        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
        """
    ).strip()


def compose_critique_rewrite_prompt(constraints_text: str, latest_draft: str, field: str = "general") -> str:
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.
        Field: {field}

        Constraints of Paper:
        {constraints_text}

        Draft:
        {latest_draft}

        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
        """
    ).strip()


def compose_annotation_prompt(paper_markdown: str) -> str:
    return dedent(
        f"""
        {SYSTEM_GUIDANCE}

        Task: Produce an annotated version of the following scientific paper for a smart layperson.
        Keep the original Markdown structure and content, and insert clear annotations that explain:
        - All mathematical terms, symbols, variables, and equations (define each in plain language)
        - Key ideas in accessible terms using concrete analogies when helpful
        - Why each experiment or method step matters

        Annotation format requirements:
        - After each paragraph or equation block, add a blockquote line starting with 'Note:' that explains it simply.
        - For equations, define each symbol used (e.g., 'α', 'β', 'p(x)', 'HOP').
        - Do not remove or alter the original text; only add explanatory notes.
        - Keep annotations concise and factual; avoid hype.

        Paper (Markdown):
        {paper_markdown}
        """
    ).strip()
