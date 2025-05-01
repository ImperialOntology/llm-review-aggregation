import pytest
from unittest.mock import mock_open, patch


@pytest.fixture
def mock_json_grammer():
    """
    Json grammer for the language model
    """
    return "Mock json grammer"


@pytest.fixture
def mock_aspect_prompt_manager(mock_json_grammer, mocker, monkeypatch):
    """
    Build the aspect prompt manager
    Prevent the prompt manager from actually open the file
    """
    with patch("builtins.open", mock_open(read_data=mock_json_grammer)):
        from src.ontology.ontology_llm.aspects.prompt_manager import SentimentAspectPromptManager
        return SentimentAspectPromptManager()


def test_get_prompt(mock_aspect_prompt_manager):
    """Test that the generated prompt contains the expected structure."""
    review_text = "The camera quality is excellent."
    prompt = mock_aspect_prompt_manager.get_prompt(review_text)

    assert review_text in prompt  # check the review is in prompt
    assert mock_aspect_prompt_manager.instruction in prompt  # check instruction in prompt
    assert mock_aspect_prompt_manager.primer in prompt  # check primer in prompt

    """Test DT post_tag will not be put into aspects"""
    review_text = "The camera quality is excellent."

    generated_text = '{"aspect": "The", "polarity":"positive"}'

    aspects = mock_aspect_prompt_manager.process_response(review_text, generated_text)

    # The aspect "The" should not be included because it's a determiner (DT), not a noun
    assert "The" not in aspects


def test_not_match_response(mock_aspect_prompt_manager):
    """Test aspect no in review will not be put into aspects"""
    review_text = "The camera quality is excellent."
    generated_text = '{"aspect": "lens", "polarity":"positive"}'

    aspects = mock_aspect_prompt_manager.process_response(review_text, generated_text)
    assert "lens" not in aspects
