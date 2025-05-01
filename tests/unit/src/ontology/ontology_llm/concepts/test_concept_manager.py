import pytest
from unittest.mock import MagicMock
from src.ontology.ontology_llm.concepts.manager import PromptedConceptExtractionManager
from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.ontology.ontology_llm.concepts.prompt_manager import ConceptPromptManager
from src.ontology.ontology_llm.concepts.fasttext_wrapper import FastTextWrapper
from src.ontology.synset_extractor import SynsetExtractor


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    yield


@pytest.fixture
def mock_dependencies():
    mock_llm = MagicMock(spec=HuggingFaceLLMWrapper)
    mock_prompt_manager = MagicMock(spec=ConceptPromptManager)
    mock_vectorizer = MagicMock(spec=FastTextWrapper)
    mock_synset_extractor = MagicMock(spec=SynsetExtractor)

    mock_prompt_manager.get_prompt.return_value = "mocked prompt"
    mock_prompt_manager.json_grammar = "mocked grammar"
    mock_prompt_manager.process_response = MagicMock(side_effect=lambda x: x)
    mock_llm.generate = MagicMock("mocked response")

    def cluster_aspects_side_effect(root_name, aspect_counts, vectorizer):
        # we assume each aspect do not have any synsets
        synset_counts = {k: v for k, v in aspect_counts.items()}
        synsets = {k: [k] for k in aspect_counts.keys()}
        return synset_counts, synsets
    mock_synset_extractor.cluster_aspects.side_effect = cluster_aspects_side_effect

    return {
        "llm": mock_llm,
        "prompt_manager": mock_prompt_manager,
        "vectorizer": mock_vectorizer,
        "synset_extractor": mock_synset_extractor
    }


@pytest.fixture
def manager(mock_dependencies):
    return PromptedConceptExtractionManager(
        root_name="product",
        llm=mock_dependencies["llm"],
        prompt_manager=mock_dependencies["prompt_manager"],
        vectorizer=mock_dependencies["vectorizer"],
        synset_extractor=mock_dependencies["synset_extractor"],
        top_k_aspects_to_keep=3  # Small value for testing
    )


def test_get_synsets(manager, mock_dependencies):
    aspect_counts = {"display": 10, "battery": 8, "camera": 5}

    synset_counts, synsets = manager._get_synsets(aspect_counts)

    # Verify the synset extractor was called
    mock_dependencies["synset_extractor"].cluster_aspects.assert_called_once_with(
        "product", aspect_counts, mock_dependencies["vectorizer"]
    )

    # Verify the results match what our mock returns
    assert synset_counts == aspect_counts
    assert synsets == {"display": ["display"], "battery": ["battery"], "camera": ["camera"]}


def test_extract_concepts(manager, mock_dependencies):
    """Test the extract_concepts method."""
    # Setup test data
    # Note: product as the root must have top-k frequencies
    aspect_frequency = {"product": 16, "display": 15, "battery": 12, "camera": 8, "price": 5, "size": 3}
    manager._batch_generate = MagicMock(side_effect=['product', 'display', 'battery'])
    mock_get_synsets = MagicMock()

    # Then set the side_effect to return a tuple of your desired values
    mock_get_synsets.side_effect = lambda aspect_counts: (
        {"product": 16, "display": 15, "battery": 12},
        {"product": ["product"], "display": ["display"], "battery": ["battery"]}
    )

    # Assign the mock to the method
    manager._get_synsets = mock_get_synsets
    manager.prompt_manager.process_response = MagicMock(side_effect=lambda x: x)
    synset_counts, synsets = manager.extract_concepts(aspect_frequency)

    assert len(synset_counts.keys()) == 3
    assert synset_counts['product'] == 16
    assert synset_counts['display'] == 15
    assert synset_counts['battery'] == 12
    assert synsets['product'] == ['product']
    assert synsets['display'] == ['display']
    assert synsets['battery'] == ['battery']
