import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.ontology.ontology_llm.relations.prompt_manager import RelationPromptManager
from src.ontology.ontology_llm.relations.manager import PromptedRelationExtractionManager


@pytest.fixture
def relation_manager():
    mock_llm = MagicMock(spec=HuggingFaceLLMWrapper)
    mock_prompt_manager = MagicMock(spec=RelationPromptManager)

    mock_prompt_manager.json_grammar = "mocked grammar"
    mock_prompt_manager.get_prompt.return_value = ("mocked prompt", "mocked json grammar")
    # Set up manager with mocked dependencies

    root_name = "product"
    manager = PromptedRelationExtractionManager(
        root_name=root_name,
        llm=mock_llm,
        prompt_manager=mock_prompt_manager,
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.0
    )

    return manager, mock_llm, mock_prompt_manager, root_name


@pytest.fixture
def test_data():
    concepts = ["product", "display", "camera", "battery"]
    synsets = {
        "product": ["product", "device", "phone"],
        "display": ["display", "screen", "resolution"],
        "camera": ["camera", "lens", "photo quality"],
        "battery": ["battery", "battery life", "charging"]
    }
    reviews = [
        "The phone has a great display with high resolution.",
        "I love the camera on this product, but the battery life is poor.",
        "The screen resolution is amazing, and the photo quality is superb.",
        "The charging is fast, but the lens could be better.",
        "This device has the best battery life I've seen."
    ]
    concept_counts = {
        "product": 10,
        "display": 8,
        "camera": 6,
        "battery": 5
    }

    return concepts, synsets, reviews, concept_counts


@patch('src.ontology.ontology_llm.relations.manager.word_tokenize')
def test_get_relation_extraction_texts(word_tokenize_mock, relation_manager, test_data):
    manager, _, _, _ = relation_manager
    concepts, synsets, _, _ = test_data
    word_tokenize_mock.side_effect = lambda text: text.split()

    sentence = "The product has a clear display"
    sentence, aspect1, synset_idx1, aspect2, synset_idx2 = manager._get_relation_extraction_texts(
        concepts, synsets, sentence)

    assert aspect1 in synsets[concepts[synset_idx1]]
    assert aspect2 in synsets[concepts[synset_idx2]]
    assert synset_idx1 != synset_idx2


@patch('src.ontology.ontology_llm.relations.manager.word_tokenize')
def test_aspects_from_same_synset(word_tokenize_mock, relation_manager, test_data):
    manager, _, _, _ = relation_manager
    concepts, synsets, _, _ = test_data
    word_tokenize_mock.side_effect = lambda text: text.split()

    sentence = "The display has excellent screen resolution."
    result = manager._get_relation_extraction_texts(concepts, synsets, sentence)
    assert result is None


@patch('src.ontology.ontology_llm.relations.manager.word_tokenize')
def test_one_aspect(word_tokenize_mock, relation_manager, test_data):
    manager, _, _, _ = relation_manager
    concepts, synsets, _, _ = test_data
    word_tokenize_mock.side_effect = lambda text: text.split()

    sentence = "The battery life is excellent."
    result = manager._get_relation_extraction_texts(concepts, synsets, sentence)
    assert result is None


@patch('src.ontology.ontology_llm.relations.manager.Pool')
@patch('src.ontology.ontology_llm.relations.manager.sent_tokenize')
@patch('src.ontology.ontology_llm.relations.manager.word_tokenize')
def test_extract_relations(word_tokenize_mock, sent_tokenize_mock, pool_mock, relation_manager, test_data):
    manager, _, prompt_manager, _ = relation_manager
    _, synsets, reviews, _ = test_data

    # Set up the mock Pool context manager
    pool_instance = MagicMock()
    pool_mock.return_value.__enter__.return_value = pool_instance

    # Mock the pool's map functions to return predetermined sentences and relations
    sentences = ["The display is part of the product.", "The camera lens is amazing."]
    pool_instance.map.side_effect = [
        [sentences],  # result of sent_tokenize
        [sentences]   # result of str.splitlines
    ]

    relation_texts = [
        # In the orde of sentence, aspect1, synset_idx1, aspect2, synset_idx2
        #  from _get_relation_extraction_texts
        ("The display is part of the product.", "display", 1, "product", 0),
        (None)  # This one should be filtered out
    ]
    pool_instance.starmap.return_value = relation_texts

    # Mock the _get_relation method, first aspect is the child
    manager._batch_generate = MagicMock(return_value=["Mock1"])
    manager.prompt_manager.process_response = MagicMock(return_value=(True, 1.0))

    result_concepts, result_matrix = manager.extract_relations(synsets, reviews)

    assert result_concepts == list(synsets.keys())
    assert result_matrix.shape == (len(result_concepts), len(result_concepts))

    # Check that the matrix has been updated correctly
    # In this case, the relation "display is part of product" means
    # meronym_matrix[1][0] should be 1.0
    assert result_matrix[1][0] == 1.0


def test_get_relatedness(relation_manager, test_data):
    manager, _, _, root_name = relation_manager
    concepts, _, _, concept_counts = test_data

    # Create a sample meronym_matrix
    meronym_matrix = np.zeros((len(concepts), len(concepts)))
    meronym_matrix[1][0] = 2.0  # parent: product, child: display it happened twice
    meronym_matrix[2][0] = 1.0  # parent: product, child: camera
    meronym_matrix[3][0] = 1.0  # parent: product, child battery
    meronym_matrix[2][3] = 1.0  # For testing purpose parent: battery, child:camera

    # Call the method
    filtered_norm_matrix, filtered_concepts = manager.tree_builder.get_relatedness(
        concepts, concept_counts, meronym_matrix
    )

    # Verify that normalization occurred
    assert filtered_norm_matrix.shape[0] == len(filtered_concepts)

    # Check that all concepts are preserved (since none have zero relations)
    assert set(filtered_concepts) == set(concepts)

    display_idx = filtered_concepts.index("display")
    product_idx = filtered_concepts.index("product")
    assert filtered_norm_matrix[display_idx][product_idx] == 0.2    # 2/10, 2 is that it appear twice,
    # Cuase its parent is product so it divided by 10

    camera_idx = filtered_concepts.index("camera")
    battery_idx = filtered_concepts.index("battery")
    assert filtered_norm_matrix[camera_idx][battery_idx] == 0.2     # 1/5

    root_idx = filtered_concepts.index(root_name)
    assert filtered_norm_matrix[root_idx][root_idx] == 1.0          # Ensure root does not fileter out


def test_get_parent(relation_manager):
    manager, _, _, _ = relation_manager

    # Create a sample relatedness matrix
    matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],  # product row
        [0.2, 0.0, 0.0, 0.0],  # display row
        [0.1, 0.0, 0.0, 0.2],  # camera row
        [0.1, 0.0, 0.0, 0.0]    # battery row
    ])

    # Test for display (idx=1)
    parent_idx, relation_score = manager.tree_builder._get_parent(1, matrix)
    assert parent_idx == 0  # product is the parent
    assert relation_score == 0.2

    # Test for camera (idx=2)
    parent_idx, relation_score = manager.tree_builder._get_parent(2, matrix)
    assert parent_idx == 3              # in this case, camera has two parents,
    assert relation_score == 0.20       # product and battery, howoever the value of battery is higher


def test_build_tree(relation_manager):
    manager, _, _, _ = relation_manager

    # Create sample data
    filtered_concepts = ["product", "display", "camera", "battery"]
    filtered_norm_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],  # product row
        [0.2, 0.0, 0.0, 0.0],  # display row
        [0.1, 0.0, 0.0, 0.2],  # camera row
        [0.1, 0.0, 0.0, 0.0]    # battery row
    ])

    # Call the method
    root = manager.tree_builder.build_tree(filtered_norm_matrix, filtered_concepts)

    # Verify the tree structure
    assert root.name == "product"
    assert len(root.children) == 2

    # All others should be direct children of product
    child_names = [child.name for child in root.children]
    assert "display" in child_names
    assert "camera" not in child_names
    assert "battery" in child_names

    battery_node = None
    for child in root.children:
        if child.name == "battery":
            battery_node = child

    assert battery_node is not None
    assert len(battery_node.children) == 1
    assert battery_node.children[0].name == "camera"


def test_extract(relation_manager, test_data):
    manager, _, _, _ = relation_manager
    concepts, synsets, reviews, concept_counts = test_data

    # Patch the component methods
    with patch.object(manager, 'extract_relations') as extract_relations_mock, \
            patch.object(manager.tree_builder, 'get_relatedness') as get_relatedness_mock:

        # Set up mocks
        extract_relations_mock.return_value = (concepts, np.zeros((4, 4)))
        # Create sample data
        filtered_concepts = ["product", "display", "camera", "battery"]
        filtered_norm_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],  # product row
            [0.2, 0.0, 0.0, 0.0],  # display row
            [0.1, 0.0, 0.0, 0.2],  # camera row
            [0.1, 0.0, 0.0, 0.0]    # battery row
        ])
        get_relatedness_mock.return_value = (filtered_norm_matrix, filtered_concepts)

        # Call the method
        result = manager.extract(concept_counts, synsets, reviews)

        # Verify the mocks were called correctly
        extract_relations_mock.assert_called_once_with(synsets, reviews)
        get_relatedness_mock.assert_called_once()

        # Check the result
        assert result.name == "product"
        assert manager.tree_builder.tree == result
