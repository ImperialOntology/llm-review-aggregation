import pytest
import os
import json

from src.llm_judge.run import run_ontology_judging
from src.llm_judge.llm_manager import Gemini

gemini_api_key = input("Enter your Gemini API key: ").strip()


@pytest.mark.integtest
def test_llm_judge_ontology():
    # Initialize LLM
    llm = Gemini(gemini_api_key=gemini_api_key)

    # Test parameters
    category_name = "Electronics"

    # Load sample ontology tree from JSON file
    with open("tests/data/sample_ontology.json", "r") as f:
        ontology_tree = json.load(f)

    # Run ontology judging
    relation_scores = run_ontology_judging(
        llm=llm,
        category_name=category_name,
        ontology_tree=ontology_tree,
        save_to_db=False  # Don't save to DB for testing
    )

    # Assertions
    # Calculate expected number of relations
    expected_relations = 0

    def count_relations(tree, parent=None):
        nonlocal expected_relations
        if isinstance(tree, dict):
            for child, subtree in tree.items():
                if parent is not None:
                    expected_relations += 1
                count_relations(subtree, child)

    count_relations(ontology_tree)

    # Check that all relations were scored
    assert len(
        relation_scores) == expected_relations, f"Expected {expected_relations} relations, got {len(relation_scores)}"

    # Check that all scores are within expected range (1-10)
    for relation, score in relation_scores.items():
        assert 1 <= score <= 10, f"Score for relation '{relation}' is out of range: {score}"

    # Print results for inspection
    print(f"Category: {category_name}")
    print("---" * 10)
    print("Relation Scores:")
    for relation, score in sorted(relation_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {relation}: {score}")
    print("===" * 10)


if __name__ == "__main__":
    test_llm_judge_ontology()
