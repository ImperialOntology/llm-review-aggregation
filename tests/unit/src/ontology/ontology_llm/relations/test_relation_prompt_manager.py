import pytest
from unittest.mock import mock_open, patch
from src.ontology.ontology_llm.relations.prompt_manager import RelationPromptManager


@pytest.fixture
def mock_json_grammer():
    """
    Json grammer for the language model
    """
    return """ASPECT1_PLACEHOLDER
              ASPECT2_PLACEHOLDER"""


@pytest.fixture
def relation_prompt_managger(mock_json_grammer):
    """
    Prevent the prompt manager from actually open the file
    """
    with patch("builtins.open", mock_open(read_data=mock_json_grammer)):
        return RelationPromptManager()


def test_get_prompt(relation_prompt_managger):
    sentence = "This is a testing sentence with aspect 1 and aspect 2"
    aspect1 = "aspect 1"
    aspect2 = "aspect 2"

    prompt, json_grammer = relation_prompt_managger.get_prompt(sentence, aspect1, aspect2)

    assert sentence in prompt
    assert aspect1 in prompt
    assert aspect2 in prompt

    assert aspect1 in json_grammer
    assert aspect2 in json_grammer


def test_process_response_ture_relation(relation_prompt_managger):
    generated_text = '"part": "lens", "whole":"camera"'
    aspect1 = "lens"
    aspect2 = "camera"

    is_first_aspect_child, number = relation_prompt_managger.process_response(generated_text, aspect1, aspect2)
    assert is_first_aspect_child == True
    assert number == float(1)


def test_process_response_false_relation(relation_prompt_managger):
    generated_text = '"part": "lens", "whole":"camera"'
    aspect1 = "camera"
    aspect2 = "lens"

    is_first_aspect_child, number = relation_prompt_managger.process_response(generated_text, aspect1, aspect2)
    assert is_first_aspect_child == False
    assert number == float(1)


def test_process_response_None_result(relation_prompt_managger):
    generated_text = '"part": "lens", "whole":"camera"'
    aspect1 = "phone"
    aspect2 = "lens"

    result = relation_prompt_managger.process_response(generated_text, aspect1, aspect2)
    assert result == None
