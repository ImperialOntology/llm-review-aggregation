import pytest
from unittest.mock import mock_open, patch
from src.ontology.ontology_llm.concepts.prompt_manager import ConceptPromptManager


@pytest.fixture
def mock_json_grammer():
    """
    Json grammer for the language model
    """
    return "Mock json grammer"


@pytest.fixture
def concept_prompt_manager(mock_json_grammer):
    """
    Prevent the prompt manager from actually open the file
    """
    with patch("builtins.open", mock_open(read_data=mock_json_grammer)):
        return ConceptPromptManager()


def test_get_prompt(concept_prompt_manager):
    product = "Laptop"
    aspect = "Battery Life"

    prompt = concept_prompt_manager.get_prompt(product, aspect)

    assert "Laptop" in prompt  # Check product in prompt
    assert "Battery Life" in prompt  # Check candidate aspect in prompt
    assert concept_prompt_manager.instruction in prompt  # check instruction in prompt
    assert concept_prompt_manager.primer in prompt  # check primer in prompt


def test_process_response_normal_true_case(concept_prompt_manager):
    generated_text = '"answer": "yes"'
    expected_result = True
    assert concept_prompt_manager.process_response(generated_text) == expected_result


def test_process_response_normal_false_case(concept_prompt_manager):
    generated_text = '"answer": "AnyOtherTexts"'
    expected_result = False
    assert concept_prompt_manager.process_response(generated_text) == expected_result


def test_process_response_no_answer_case(concept_prompt_manager):
    generated_text = '" ": "yes"'
    expected_result = False
    assert concept_prompt_manager.process_response(generated_text) == expected_result


def test_process_response_with_noise(concept_prompt_manager):
    generated_text = 'A "answer": "yes" A'
    expected_result = True
    assert concept_prompt_manager.process_response(generated_text) == expected_result
