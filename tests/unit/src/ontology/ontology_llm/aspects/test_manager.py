from unittest.mock import MagicMock, patch, mock_open
import json
from pathlib import Path

# Create proper mock classes

MOCK_CACHE_DIR = '/tmp/mock_cache_dir'


class MockHuggingFaceLLMWrapper:
    def __init__(self):
        pass


class MockExtractionManager:
    def __init__(self, llm, prompt_manager, batch_size, max_new_tokens, temperature, top_p, repetition_penalty):
        # Store these attributes so they can be accessed by the subclass
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def _batch_generate(self, messages, grammar):
        # This will be mocked in the test
        pass


class MockSentimentAspectPromptManager:
    def __init__(self):
        self.json_grammar = "test_grammar"

    def get_prompt(self, review):
        pass

    def process_response(self, review, response):
        pass


class TestPromptedAspectExtractionManager:
    """Test cases for PromptedAspectExtractionManager."""

    def setup_method(self):
        from src.ontology.ontology_llm.aspects.manager import PromptedAspectExtractionManager

        """Set up test fixtures before each test method is executed."""
        # Create mock instances
        self.mock_llm = MagicMock()
        self.mock_prompt_manager = MagicMock()
        self.mock_prompt_manager.json_grammar = "test_grammar"

        # Create an instance of the class to test
        self.manager = PromptedAspectExtractionManager(
            llm=self.mock_llm,
            prompt_manager=self.mock_prompt_manager,
            batch_size=2,  # Small batch size for testing
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1
        )

        # Mock the _batch_generate method
        self.manager._batch_generate = MagicMock()

    def test_extract_aspects_from_reviews(self):
        # Setup
        reviews = ["This is a good product", "This is a bad product"]
        expected_prompts = ["prompt1", "prompt2"]
        expected_responses = ["response1", "response2"]
        expected_aspects = [["aspect1"], ["aspect2"]]

        # Configure mocks
        self.mock_prompt_manager.get_prompt.side_effect = expected_prompts
        self.manager._batch_generate.return_value = expected_responses
        self.mock_prompt_manager.process_response.side_effect = expected_aspects

        # Mock the file operations
        mock_file = mock_open()

        # Test with default output path
        with patch('builtins.open', mock_file):
            result = self.manager.extract_aspects_from_reviews(reviews, output_file=Path(MOCK_CACHE_DIR)/"aspects.csv")

        # Verify the results
        assert result == expected_aspects

        # Verify the mocks were called correctly
        self.mock_prompt_manager.get_prompt.assert_any_call(reviews[0])
        self.mock_prompt_manager.get_prompt.assert_any_call(reviews[1])
        self.manager._batch_generate.assert_called_once_with(expected_prompts, "test_grammar")
        self.mock_prompt_manager.process_response.assert_any_call(reviews[0], expected_responses[0])
        self.mock_prompt_manager.process_response.assert_any_call(reviews[1], expected_responses[1])

        # Verify file operations
        mock_file.assert_called_once_with(Path(MOCK_CACHE_DIR)/"aspects.csv", "w")
        mock_file().write.assert_any_call(f"{json.dumps(expected_aspects[0])}\n")
        mock_file().write.assert_any_call(f"{json.dumps(expected_aspects[1])}\n")

    def test_extract_aspects_with_custom_output_path(self):
        # Setup
        reviews = ["Test review"]
        custom_path = Path("/custom/path/output.csv")

        # Configure mocks
        self.mock_prompt_manager.get_prompt.return_value = "prompt"
        self.manager._batch_generate.return_value = ["response"]
        self.mock_prompt_manager.process_response.return_value = ["test_aspect"]

        # Mock the file operations
        mock_file = mock_open()

        # Test with custom output path
        with patch('builtins.open', mock_file):
            result = self.manager.extract_aspects_from_reviews(reviews, output_file=custom_path)

        # Verify the results
        assert result == [["test_aspect"]]

        # Verify file operations with custom path
        mock_file.assert_called_once_with(custom_path, "w")
        mock_file().write.assert_called_once_with(f"{json.dumps(['test_aspect'])}\n")
