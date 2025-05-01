import pytest
from unittest.mock import MagicMock, patch
import sys

# Create mock modules and exceptions
class MockResourceExhausted(Exception):
    pass

class MockInvalidArgument(Exception):
    pass

class MockGoogleAPICallError(Exception):
    pass

# Create proper nested structure for the google modules
mock_exceptions = MagicMock()
mock_exceptions.ResourceExhausted = MockResourceExhausted
mock_exceptions.InvalidArgument = MockInvalidArgument
mock_exceptions.GoogleAPICallError = MockGoogleAPICallError

mock_api_core = MagicMock()
mock_api_core.exceptions = mock_exceptions

mock_genai = MagicMock()
mock_genai.types = MagicMock()

# Create a mock Google module with the proper structure
mock_google = MagicMock()
mock_google.api_core = mock_api_core
mock_google.generativeai = mock_genai

# Setup the module mocks
sys.modules['google'] = mock_google
sys.modules['google.api_core'] = mock_api_core
sys.modules['google.api_core.exceptions'] = mock_exceptions
sys.modules['google.generativeai'] = mock_genai

# Now import the actual classes to test
with patch.dict('sys.modules', {
    'google.generativeai': mock_genai,
    'google.api_core.exceptions': mock_exceptions
}):
    from src.base.llm_wrapper import BaseLLMWrapper
    # Import the actual Gemini class
    from src.llm_judge.llm_manager import Gemini


class TestGemini:
    def setup_method(self):
        """Setup for each test method"""
        # Create mock for BaseLLMWrapper - the parent class
        self.base_llm_wrapper_mock = MagicMock(spec=BaseLLMWrapper)
        
        # We'll use the already created mock modules
        self.genai_mock = mock_genai
        self.exceptions_mock = mock_exceptions
        
        # Configure the Gemini API key
        self.api_key = "test_api_key"
        
        # Setup model mock
        self.model_mock = MagicMock()
        self.genai_mock.GenerativeModel.return_value = self.model_mock
        
        # Create an instance of Gemini with the mocked dependencies
        self.gemini = Gemini(self.api_key)
    
    def test_initialization(self):
        # Verify API was configured with the correct key
        self.genai_mock.configure.assert_called_once_with(api_key=self.api_key)
        
        # Verify the model was created with the default name
        self.genai_mock.GenerativeModel.assert_called_once_with("gemini-1.5-flash")
        
        # Verify that the model attribute was set correctly
        assert self.gemini.model == self.model_mock

    def test_initialization_with_custom_model(self):
        custom_model = "gemini-1.5-pro"
        with patch('time.sleep'):
            gemini = Gemini(self.api_key, model_name=custom_model)
        
        # Verify the model was created with the custom name
        self.genai_mock.GenerativeModel.assert_called_with(custom_model)

    def test_generate_simple_case(self):
        # Setup chat mock
        chat_mock = MagicMock()
        self.model_mock.start_chat.return_value = chat_mock
        
        # Setup response mock
        response_mock = MagicMock()
        response_mock.text = "This is a test response"
        chat_mock.send_message.return_value = response_mock
        
        # Setup chat history
        chat_mock._history = []
        
        # Test with a simple message
        messages = ["Hello, how are you?"]
        result = self.gemini.generate(messages)
        
        # Verify the chat was started
        self.model_mock.start_chat.assert_called_once_with(history=[])
        
        # Verify the message was sent with the correct configuration
        chat_mock.send_message.assert_called_with(
            "Hello, how are you?", 
            generation_config=self.genai_mock.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=128,
                temperature=0.3,
                top_p=0.95,
                top_k=40
            )
        )
        
        # Verify the result is correct
        assert result == "This is a test response"

    def test_generate_with_conversation_history(self):
        # Setup chat mock
        chat_mock = MagicMock()
        self.model_mock.start_chat.return_value = chat_mock
        
        # Setup response mock
        response_mock = MagicMock()
        response_mock.text = "I can help with that"
        chat_mock.send_message.return_value = response_mock
        
        # Setup chat history
        chat_mock._history = []
        
        # Test with a conversation history
        messages = ["Hello", "Hi there", "Can you help me?"]
        result = self.gemini.generate(messages)
        
        # Verify chat was started
        self.model_mock.start_chat.assert_called_once()
        
        # Verify the first user message was sent
        chat_mock.send_message.assert_any_call("Hello")
        
        # Verify the assistant's response was added to history
        assert chat_mock._history == [{"role": "model", "parts": [{"text": "Hi there"}]}]
        
        # Verify the final message was sent with the config
        chat_mock.send_message.assert_called_with(
            "Can you help me?",
            generation_config=self.genai_mock.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=128,
                temperature=0.3,
                top_p=0.95,
                top_k=40
            )
        )
        
        # Verify the result is correct
        assert result == "I can help with that"

    def test_generate_with_resource_exhausted_then_success(self):
        # Setup chat mock
        with patch('time.sleep') as sleep_mock:
            gemini = Gemini(self.api_key)
            chat_mock = MagicMock()
            self.model_mock.start_chat.return_value = chat_mock

            # Setup chat history
            chat_mock._history = []

            # Make send_message raise ResourceExhausted once then succeed
            response_mock = MagicMock()
            response_mock.text = "Success after retry"

            # First call raises exception, second returns mock response
            chat_mock.send_message.side_effect = [
                self.exceptions_mock.ResourceExhausted("Rate limit exceeded"), 
                response_mock
            ]

            # Test with a simple message
            messages = ["Test message"]
            result = gemini.generate(messages)

            # Verify sleep was called (for the retry)
            sleep_mock.assert_called_with(10)
        
            # Verify the result is correct
            assert result == "Success after retry"

    def test_generate_with_invalid_argument_error(self):
        # Setup chat mock
        chat_mock = MagicMock()
        self.model_mock.start_chat.return_value = chat_mock
        
        # Setup chat history
        chat_mock._history = []
        
        # Make send_message raise InvalidArgument
        invalid_arg_error = self.exceptions_mock.InvalidArgument("Invalid input")
        chat_mock.send_message.side_effect = invalid_arg_error
        
        # Test with a simple message and expect an exception
        messages = ["Test message"]
        with pytest.raises(self.exceptions_mock.InvalidArgument):
            self.gemini.generate(messages)

    def test_generate_with_api_call_error_then_success(self):
        with patch('time.sleep') as sleep_mock:
            # Setup chat mock
            chat_mock = MagicMock()
            self.model_mock.start_chat.return_value = chat_mock

            # Setup chat history
            chat_mock._history = []

            # Make send_message raise GoogleAPICallError once then succeed
            response_mock = MagicMock()
            response_mock.text = "Success after API error"

            # First call raises exception, second returns mock response
            chat_mock.send_message.side_effect = [
                self.exceptions_mock.GoogleAPICallError("API error"), 
                response_mock
            ]

            # Test with a simple message
            messages = ["Test message"]
            result = self.gemini.generate(messages)

            # Verify sleep was called (for the retry)
            sleep_mock.assert_called_with(5)  # 5 seconds for API call error

            # Verify the result is correct
            assert result == "Success after API error"

    def test_generate_with_unexpected_error_then_success(self):
        with patch('time.sleep') as sleep_mock:
            # Setup chat mock
            chat_mock = MagicMock()
            self.model_mock.start_chat.return_value = chat_mock

            # Setup chat history
            chat_mock._history = []

            # Make send_message raise a generic exception once then succeed
            response_mock = MagicMock()
            response_mock.text = "Success after unexpected error"

            # First call raises exception, second returns mock response
            chat_mock.send_message.side_effect = [
                Exception("Unexpected error"), 
                response_mock
            ]

            # Test with a simple message
            messages = ["Test message"]
            result = self.gemini.generate(messages)

            # Verify sleep was called (for the retry)
            sleep_mock.assert_called_with(10)  # 10 seconds for unexpected error

            # Verify the result is correct
            assert result == "Success after unexpected error"

    def test_print_result(self):
        # Setup chat mock
        chat_mock = MagicMock()
        self.model_mock.start_chat.return_value = chat_mock
        
        # Setup response mock
        response_mock = MagicMock()
        response_mock.text = "Printed response"
        chat_mock.send_message.return_value = response_mock
        
        # Setup chat history
        chat_mock._history = []
        
        # Test with print_result=True
        messages = ["Hello"]
        
        # Test that output is printed
        with patch('builtins.print') as print_mock:
            self.gemini.generate(messages, print_result=True)
            print_mock.assert_called_with("Printed response", flush=True)

    def test_generate_with_empty_messages(self):
        # Setup chat mock
        chat_mock = MagicMock()
        self.model_mock.start_chat.return_value = chat_mock
        
        # Setup response mock
        response_mock = MagicMock()
        response_mock.text = "Default response"
        chat_mock.send_message.return_value = response_mock
        
        # Setup chat history
        chat_mock._history = []
        
        # Test with empty messages
        messages = []
        result = self.gemini.generate(messages)
        
        # Verify the default message was sent
        chat_mock.send_message.assert_called_with(
            "Hello, can you help me?",
            generation_config=self.genai_mock.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=128,
                temperature=0.3,
                top_p=0.95,
                top_k=40
            )
        )
        
        # Verify the result is correct
        assert result == "Default response"

