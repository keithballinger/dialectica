"""Literature review module for searching and summarizing academic papers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TypedDict
from urllib import request, parse, error
import json

from .log import info, step, done


class Paper(TypedDict):
    """Metadata for an academic paper."""

    title: str
    authors: list[str]
    year: int | None
    abstract: str
    citations: int
    url: str
    venue: str
    paper_id: str


def search_semantic_scholar(
    query: str, max_results: int = 10, fields: list[str] | None = None
) -> list[Paper]:
    """Search Semantic Scholar for papers matching the query.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        fields: List of fields to retrieve (default: title, abstract, authors, year, etc.)

    Returns:
        List of Paper dictionaries with metadata
    """
    if fields is None:
        fields = [
            "title",
            "abstract",
            "authors",
            "year",
            "citationCount",
            "url",
            "venue",
            "paperId",
        ]

    # Build API request
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),  # API limit per request
        "fields": ",".join(fields),
    }

    url = f"{base_url}?{parse.urlencode(params)}"

    # Add API key if available
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        req = request.Request(url, headers=headers)
        with request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        papers: list[Paper] = []
        for item in data.get("data", []):
            # Extract author names
            author_list = item.get("authors", [])
            author_names = [a.get("name", "Unknown") for a in author_list]

            paper: Paper = {
                "title": item.get("title", "Untitled"),
                "authors": author_names,
                "year": item.get("year"),
                "abstract": item.get("abstract", ""),
                "citations": item.get("citationCount", 0),
                "url": item.get("url", ""),
                "venue": item.get("venue", ""),
                "paper_id": item.get("paperId", ""),
            }
            papers.append(paper)

        return papers

    except error.HTTPError as e:
        info(f"HTTP error searching Semantic Scholar: {e.code} {e.reason}")
        return []
    except error.URLError as e:
        info(f"URL error searching Semantic Scholar: {e.reason}")
        return []
    except Exception as e:
        info(f"Error searching Semantic Scholar: {e}")
        return []


def search_arxiv(query: str, max_results: int = 10) -> list[Paper]:
    """Search arXiv for papers matching the query.

    Requires: pip install arxiv

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of Paper dictionaries with metadata
    """
    try:
        import arxiv
    except ImportError:
        info("arxiv package not installed. Install with: pip install arxiv")
        return []

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[Paper] = []
        for result in client.results(search):
            paper: Paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "year": result.published.year if result.published else None,
                "abstract": result.summary,
                "citations": 0,  # arXiv doesn't provide citation counts
                "url": result.entry_id,
                "venue": "arXiv",
                "paper_id": result.entry_id.split("/")[-1],
            }
            papers.append(paper)

        return papers

    except Exception as e:
        info(f"Error searching arXiv: {e}")
        return []


def search_literature(
    query: str, max_results: int = 20, sources: list[str] | None = None
) -> list[Paper]:
    """Search multiple literature sources and combine results.

    Args:
        query: Search query string
        max_results: Maximum total results to return
        sources: List of sources to search (default: ["semantic_scholar", "arxiv"])

    Returns:
        Combined list of papers from all sources, deduplicated by title
    """
    if sources is None:
        sources = ["semantic_scholar", "arxiv"]

    all_papers: list[Paper] = []

    if "semantic_scholar" in sources:
        step(f"Searching Semantic Scholar: '{query}'")
        papers = search_semantic_scholar(query, max_results=max_results)
        all_papers.extend(papers)
        info(f"Found {len(papers)} papers from Semantic Scholar")
        time.sleep(0.5)  # Rate limiting

    if "arxiv" in sources:
        step(f"Searching arXiv: '{query}'")
        papers = search_arxiv(query, max_results=max_results // 2)
        all_papers.extend(papers)
        info(f"Found {len(papers)} papers from arXiv")

    # Deduplicate by normalized title
    seen_titles: set[str] = set()
    unique_papers: list[Paper] = []
    for paper in all_papers:
        normalized = paper["title"].lower().strip()
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique_papers.append(paper)

    # Sort by citation count (descending)
    unique_papers.sort(key=lambda p: p["citations"], reverse=True)

    return unique_papers[:max_results]


def format_paper_summary(paper: Paper) -> str:
    """Format a paper as a markdown citation with summary."""
    authors_str = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors_str += " et al."

    year_str = f"({paper['year']})" if paper["year"] else "(n.d.)"
    venue_str = f" *{paper['venue']}*" if paper["venue"] else ""
    citations_str = f" [{paper['citations']} citations]" if paper["citations"] > 0 else ""

    summary = f"**{paper['title']}**\n"
    summary += f"{authors_str} {year_str}{venue_str}{citations_str}\n"
    if paper["abstract"]:
        # Truncate abstract to ~200 chars
        abstract = paper["abstract"][:200].strip()
        if len(paper["abstract"]) > 200:
            abstract += "..."
        summary += f"> {abstract}\n"
    if paper["url"]:
        summary += f"[Link]({paper['url']})\n"

    return summary


def create_literature_review(
    query: str, max_results: int = 10, sources: list[str] | None = None
) -> str:
    """Create a formatted literature review markdown document.

    Args:
        query: Search query
        max_results: Maximum number of papers to include
        sources: List of sources to search

    Returns:
        Markdown formatted literature review
    """
    step(f"Creating literature review for: '{query}'")

    papers = search_literature(query, max_results=max_results, sources=sources)

    if not papers:
        return f"# Literature Review\n\nQuery: {query}\n\nNo papers found.\n"

    review = f"# Literature Review\n\n"
    review += f"**Query:** {query}\n\n"
    review += f"**Found {len(papers)} relevant papers**\n\n"
    review += "---\n\n"

    for i, paper in enumerate(papers, 1):
        review += f"## {i}. {paper['title']}\n\n"
        review += format_paper_summary(paper)
        review += "\n---\n\n"

    done(f"Literature review complete: {len(papers)} papers")
    return review


def extract_key_concepts(papers: list[Paper], top_n: int = 10) -> list[str]:
    """Extract key concepts/keywords from a set of papers.

    Simple implementation using title word frequency.
    Could be enhanced with NLP or LLM summarization.

    Args:
        papers: List of papers to analyze
        top_n: Number of top concepts to return

    Returns:
        List of key concepts/keywords
    """
    # Simple word frequency from titles and abstracts
    word_counts: dict[str, int] = {}
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those", "we",
        "our", "using", "based", "via", "new", "novel", "approach", "method"
    }

    for paper in papers:
        text = (paper["title"] + " " + paper["abstract"]).lower()
        words = text.split()
        for word in words:
            # Simple cleanup
            word = word.strip(".,;:()[]{}\"'")
            if len(word) > 3 and word not in stopwords and word.isalpha():
                word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def check_novelty(idea_text: str, related_papers: list[Paper]) -> str:
    """Generate a novelty assessment by comparing idea to existing papers.

    Args:
        idea_text: The proposed idea/theory text
        related_papers: List of related papers found in literature

    Returns:
        Markdown formatted novelty assessment
    """
    if not related_papers:
        return "**Novelty Assessment:** No closely related papers found. Idea may be highly novel.\n"

    assessment = "**Novelty Assessment:**\n\n"
    assessment += f"Found {len(related_papers)} related papers. "

    # Count highly cited papers (potential sign of well-established area)
    high_citations = [p for p in related_papers if p["citations"] > 100]
    if high_citations:
        assessment += f"{len(high_citations)} are highly cited (>100 citations), "
        assessment += "suggesting this is an established research area. "

    # Check recent papers (published in last 2 years)
    import datetime
    current_year = datetime.datetime.now().year
    recent_papers = [
        p for p in related_papers if p["year"] and p["year"] >= current_year - 2
    ]
    if recent_papers:
        assessment += f"{len(recent_papers)} papers published in last 2 years, indicating active research. "
    else:
        assessment += "No recent papers found, suggesting potential opportunity. "

    assessment += "\n\n**Recommendation:** Review the related papers below to identify gaps and differentiate your approach.\n"

    return assessment
