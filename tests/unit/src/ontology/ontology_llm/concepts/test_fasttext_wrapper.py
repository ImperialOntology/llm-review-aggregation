import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.ontology.ontology_llm.concepts.fasttext_wrapper import FastTextWrapper


@pytest.fixture
def mock_fasttext_model():
    mock_model = MagicMock()
    mock_model.wv = MagicMock()

    mock_vocab = {"word1": np.random.randn(10),
                  "word2": np.random.randn(10)}

    def get_vector(word):
        return mock_vocab[word] if word in mock_vocab else np.random.randn(10)

    def similarity(word1, word2):
        return 0.8 if (word1 == "word1" and word2 == "word2") else 0.0

    mock_model.wv.__contains__.side_effect = lambda word: word in mock_vocab
    mock_model.wv.__getitem__.side_effect = get_vector
    mock_model.wv.similarity.side_effect = similarity
    return mock_model


@pytest.fixture
@patch("src.ontology.ontology_llm.concepts.fasttext_wrapper.FastText")
def mock_fasttext_wrapper(mock_fasttext, mock_fasttext_model):
    wrapper = FastTextWrapper(load_model=False)
    mock_fasttext.return_value = mock_fasttext_model
    wrapper.is_fit = False
    wrapper.word_vectors = mock_fasttext_model.wv
    wrapper.wnl = MagicMock()
    return wrapper


@patch('src.ontology.ontology_llm.concepts.fasttext_wrapper.word_tokenize')
def test_learn_embeddings(mock_fasttext_wrapper):
    """
    Test that learn_embeddings correctly updates the wrapper's attributes
    """
    sample_sentences = ["This is a test.", "Machine learning is cool."]

    mock_fasttext_wrapper.learn_embeddings(sample_sentences, save_model=False)


def test_similarity_same_lemma(mock_fasttext_wrapper):
    mock_fasttext_wrapper.wnl.lemmatize.side_effect = lambda word: "run" if word in ["running", "runs"] else word
    result = mock_fasttext_wrapper.similarity("running", "runs")

    assert result == 1
    assert mock_fasttext_wrapper.wnl.lemmatize.call_count == 2
    mock_fasttext_wrapper.word_vectors.relative_cosine_similarity.assert_not_called()


def test_similarity_different_words(mock_fasttext_wrapper):
    mock_fasttext_wrapper.wnl.lemmatize.side_effect = lambda word: word
    mock_fasttext_wrapper.word_vectors.relative_cosine_similarity = MagicMock()
    mock_fasttext_wrapper.word_vectors.relative_cosine_similarity.side_effect = [0.3, 0.4]

    result = mock_fasttext_wrapper.similarity("cat", "dog")

    assert result == 0.7  # 0.3 + 0.4
    assert mock_fasttext_wrapper.wnl.lemmatize.call_count == 2  # lemmatize should be called twice
    assert mock_fasttext_wrapper.word_vectors.relative_cosine_similarity.call_count == 2  # cosin similarity should be called twice
    mock_fasttext_wrapper.word_vectors.relative_cosine_similarity.assert_any_call(
        "cat", "dog")  # either order should each be called once
    mock_fasttext_wrapper.word_vectors.relative_cosine_similarity.assert_any_call("dog", "cat")


def test_are_syns_identical_words(mock_fasttext_wrapper):

    result = mock_fasttext_wrapper.are_syns("word1", "word1", 0.2)

    assert result is True  # same word are expecting a True result


def test_are_syns_same_lemma(mock_fasttext_wrapper):
    mock_fasttext_wrapper.wnl.lemmatize.side_effect = lambda word: "cat" if word in ["cat", "cats"] else word
    result = mock_fasttext_wrapper.are_syns("cat", "cats", 0.2)  # different word with same lemmatize

    assert result is True
    assert mock_fasttext_wrapper.wnl.lemmatize.call_count == 2


def test_are_syns_above_threshold(mock_fasttext_wrapper):
    mock_fasttext_wrapper.wnl.lemmatize.side_effect = lambda word: word
    mock_fasttext_wrapper.similarity = MagicMock(return_value=0.9)

    result = mock_fasttext_wrapper.are_syns("happy", "glad", 0.2)  # words with similarity higher than threshold

    assert result is True
    mock_fasttext_wrapper.similarity.assert_called_once_with("glad", "happy")


def test_are_syns_below_threshold(mock_fasttext_wrapper):
    mock_fasttext_wrapper.wnl.lemmatize.side_effect = lambda word: word
    mock_fasttext_wrapper.similarity = MagicMock(return_value=0.1)

    result = mock_fasttext_wrapper.are_syns("devil", "angel", 0.2)  # words with similarity lower than threshold

    assert result is False
    mock_fasttext_wrapper.similarity.assert_called_once_with("angel", "devil")
