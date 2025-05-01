import pytest
import numpy as np
from src.ontology.synset_extractor import SynsetExtractor
from unittest.mock import MagicMock


@pytest.fixture
def synset_extractor():
    return SynsetExtractor()


@pytest.fixture
def mock_wv_model():
    mock_model = MagicMock()

    # Configure the similarity method first (since are_syns will use it)
    def similarity_side_effect(a1, a2):
        if a1 == a2:
            return 1.0

        # Define similarity values for specific pairs
        sim_pairs = {
            ("happy", "glad"): 0.8,
            ("glad", "happy"): 0.8,
            ("buy", "purchase"): 0.9,
            ("purchase", "buy"): 0.9,
            ("car", "auto"): 0.7,
            ("auto", "car"): 0.7,
            ("happy", "buy"): 0.1,
            ("happy", "car"): 0.05,
            ("glad", "purchase"): 0.15,
            ("glad", "auto"): 0.1
        }

        # Use get() to handle missing pairs with a default of 0.0
        return sim_pairs.get((a1, a2), 0.0)

    mock_model.similarity.side_effect = similarity_side_effect

    # Configure the are_syns method to return true when similarity > threshold
    def are_syns_side_effect(a1, a2, threshold):
        # Always return True for identical terms
        if a1 == a2:
            return True

        # For different terms, check similarity against threshold
        return mock_model.similarity(a1, a2) >= threshold

    mock_model.are_syns.side_effect = are_syns_side_effect

    return mock_model


def test_connected(synset_extractor):
    m = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ])

    assert synset_extractor._connected(0, 0, m, 0) is False  # k == 0
    assert synset_extractor._connected(1, 1, m, 1) is True  # m[idx1][idx2] != 0
    assert synset_extractor._connected(0, 0, m, 1) is False  # m[idx1][idx2] == 0
    assert synset_extractor._connected(2, 1, m, 2) is True  # went to the next loop and True


def test_clique_similarity_single_node():
    m = np.array([[0]])
    c = {0}
    assert SynsetExtractor._clique_similarity(c, m) == 1


def test_clique_similarity(synset_extractor):
    m = np.array([
        [0.8, 0.2],
        [1, 0]
    ])

    c = [0, 1]

    similarity_1 = synset_extractor._clique_similarity(c, m)
    assert similarity_1 == 0.2 * 1


def test_cluster_aspects_identical_aspects(synset_extractor, mock_wv_model):
    root_aspect = "car"
    counts = {"car": 10}

    synset_counts, synsets = synset_extractor.cluster_aspects(root_aspect, counts, mock_wv_model)

    assert synset_counts == {"car": 10}
    assert synsets == {"car": ["car"]}


def test_cluster_aspects(synset_extractor, mock_wv_model):
    root_aspect = "car"
    counts = {
        "car": 10, "auto": 8,
        "happy": 5, "glad": 3,
        "purchase": 7, "buy": 4
    }

    synset_counts, synsets = synset_extractor.cluster_aspects(root_aspect, counts, mock_wv_model)

    # Only three synsets should be included (car, auto), (happy, glad), (purchase, buy)
    assert len(synset_counts) == 3
    assert len(synsets) == 3

    # Three most frequent word represent each category
    assert "car" in synsets
    assert "happy" in synsets
    assert "purchase" in synsets

    # Check synsets and synset counts meet with our expectation
    assert set(synsets["car"]) == {"car", "auto"}
    assert synset_counts["car"] == 18
    assert set(synsets["happy"]) == {"happy", "glad"}
    assert synset_counts["happy"] == 8
    assert set(synsets["purchase"]) == {"purchase", "buy"}
    assert synset_counts["purchase"] == 11


def test_cluster_aspects_root_not_representative(synset_extractor, mock_wv_model):
    root_aspect = "auto"  # Less frequent than "car"
    counts = {"car": 10, "auto": 8}

    synset_counts, synsets = synset_extractor.cluster_aspects(root_aspect, counts, mock_wv_model)

    # The root aspect should be the representative even though it's less frequent
    assert "auto" in synsets
    assert set(synsets["auto"]) == {"car", "auto"}
    assert synset_counts["auto"] == 18
    assert "car" not in synsets


def test_cluster_aspects_root_not_in_synsets(synset_extractor, mock_wv_model):
    root_aspect = "vehicle"  # root not appear
    counts = {"car": 10, "auto": 8}

    synset_counts, synsets = synset_extractor.cluster_aspects(root_aspect, counts, mock_wv_model)

    # if root_aspect neither in key or value of synsets, remain unchagne
    assert "car" in synsets
    assert set(synsets["car"]) == {"car", "auto"}
    assert synset_counts["car"] == 18
    assert "vehicle" not in synsets
