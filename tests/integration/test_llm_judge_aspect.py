import pytest
import json

from src.llm_judge.run import run_aspect_judging
from src.llm_judge.llm_manager import Gemini

gemini_api_key = input("Enter your Gemini API key: ").strip()


@pytest.mark.integtest
def test_llm_judge_aspect():
    # Initialize LLM
    llm = Gemini(gemini_api_key=gemini_api_key)

    # Test parameters
    category_name = "Electronics"

    # Load sample aspect frequency from JSON file
    with open("tests/data/sample_aspects.json", "r") as f:
        aspect_frequency = json.load(f)

    # Run aspect judging
    aspect_scores = run_aspect_judging(
        llm=llm,
        category_name=category_name,
        aspect_frequency=aspect_frequency,
        save_to_db=False  # Don't save to DB for testing
    )

    # Assertions
    assert len(aspect_scores) == len(aspect_frequency)

    # Check that all scores are within expected range (1-10)
    for aspect, score in aspect_scores.items():
        assert 1 <= score <= 5, f"Score for aspect '{aspect}' is out of range: {score}"

    # Print results for inspection
    print(f"Category: {category_name}")
    print("---" * 10)
    print("Aspect Scores:")
    for aspect, score in sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {aspect}: {score}")
    print("===" * 10)


if __name__ == "__main__":
    test_llm_judge_aspect()
