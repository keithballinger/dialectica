#!/usr/bin/env python3
"""Test script for literature search functionality."""

from dialectica.literature import (
    search_semantic_scholar,
    search_literature,
    create_literature_review,
    extract_key_concepts,
    check_novelty,
)


def test_semantic_scholar():
    print("Testing Semantic Scholar search...")
    papers = search_semantic_scholar("large language model inference", max_results=5)
    print(f"Found {len(papers)} papers from Semantic Scholar")
    if papers:
        print(f"First paper: {papers[0]['title']}")
        print(f"  Authors: {', '.join(papers[0]['authors'][:3])}")
        print(f"  Year: {papers[0]['year']}")
        print(f"  Citations: {papers[0]['citations']}")
    print()


def test_literature_search():
    print("Testing combined literature search...")
    papers = search_literature("quantum computing optimization", max_results=10)
    print(f"Found {len(papers)} papers total")
    print()


def test_literature_review():
    print("Testing literature review generation...")
    review = create_literature_review(
        "machine learning optimization algorithms", max_results=5
    )
    print(f"Generated review of {len(review)} characters")
    print("\nFirst 500 chars of review:")
    print(review[:500])
    print()


def test_key_concepts():
    print("Testing key concept extraction...")
    papers = search_literature("neural network compression", max_results=5)
    concepts = extract_key_concepts(papers, top_n=10)
    print(f"Key concepts: {', '.join(concepts[:10])}")
    print()


def test_novelty_check():
    print("Testing novelty assessment...")
    idea = "A new method for optimizing large language model inference using dynamic pruning"
    papers = search_literature("language model inference optimization", max_results=5)
    assessment = check_novelty(idea, papers)
    print(assessment)
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Literature Search Module Tests")
    print("=" * 60)
    print()

    try:
        test_semantic_scholar()
        test_literature_search()
        test_literature_review()
        test_key_concepts()
        test_novelty_check()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
