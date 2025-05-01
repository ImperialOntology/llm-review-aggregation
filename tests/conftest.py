import pytest
import sys
from unittest.mock import MagicMock


def pytest_collection_modifyitems(items):
    """Modifies test items in place to ensure test modules run in a given order."""
    MODULE_ORDER = ["ontology", 
                    "data",
                    "llm_judge",
                    "argumentation",]
    module_mapping = {item: item.module.__name__ for item in items}

    sorted_items = items.copy()
    # Iteratively move tests of each module to the end of the test queue
    for module in MODULE_ORDER:
        sorted_items = [it for it in sorted_items if module not in module_mapping[it]] + [
            it for it in sorted_items if module in module_mapping[it]
        ]
    items[:] = sorted_items

@pytest.fixture(autouse=True)
def patch_import(monkeypatch):
    """
    Patch the import of the nltk library to prevent it from being loaded
    and slowing down the tests.
    """

    # Create a mock for nltk and its submodules
    mock_nltk = MagicMock()
    # create a mock for transformers
    mock_transformers = MagicMock()
    mock_transformers_cfg = MagicMock()
    mock_gensim = MagicMock()
    mock_torch = MagicMock()

    # Patch sys.modules to replace 'nltk' with the mock
    monkeypatch.setitem(sys.modules, "nltk", mock_nltk)
    monkeypatch.setitem(sys.modules, "nltk.tokenize", mock_nltk.tokenize)
    monkeypatch.setitem(sys.modules, "nltk.stem", mock_nltk.stem)
    monkeypatch.setitem(sys.modules, "nltk.corpus", mock_nltk.corpus)

    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)
    monkeypatch.setitem(sys.modules, "transformers.generation", mock_transformers.generation)

    # transformers_cfg.grammar_utils
    monkeypatch.setitem(sys.modules, "transformers_cfg", mock_transformers_cfg)
    monkeypatch.setitem(sys.modules, "transformers_cfg.grammar_utils", mock_transformers_cfg.grammar_utils)
    monkeypatch.setitem(sys.modules, "transformers_cfg.generation", mock_transformers_cfg.generation)
    monkeypatch.setitem(sys.modules, "transformers_cfg.generation.logits_process",
                        mock_transformers_cfg.generation.logits_process)
    
    # Patch sys.modules to replace 'gensim' with the mock
    monkeypatch.setitem(sys.modules, "gensim", mock_gensim)
    monkeypatch.setitem(sys.modules, "gensim.models", mock_gensim.models)
    monkeypatch.setitem(sys.modules, "gensim.models.phrases", mock_gensim.models.phrases)

    # Patch sys.modules to replace 'torch' with the mock
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", mock_torch.nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", mock_torch.nn.functional)
    monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)

    yield
