from abc import ABC
from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.base.prompt_manager import BasePromptManager
from tqdm import tqdm


class ExtractionManager(ABC):
    """
    A base class for managing the extraction process using a language model and prompt manager.
    """

    def __init__(self,
                 llm: HuggingFaceLLMWrapper,
                 prompt_manager: BasePromptManager,
                 batch_size=32,
                 max_new_tokens=200,
                 temperature=0.3,
                 top_p=0.95,
                 repetition_penalty=1.0,
                 ) -> None:
        """
        Initialize the ExtractionManager.

        Args:
            llm (HuggingFaceLLMWrapper): The language model wrapper for generating responses.
            prompt_manager (BasePromptManager): The prompt manager for handling prompts.
            batch_size (int): The number of messages to process in a single batch (default: 32).
            max_new_tokens (int): The maximum number of tokens to generate for each response (default: 200).
            temperature (float): The sampling temperature for controlling randomness in generation (default: 0.3).
            top_p (float): The nucleus sampling probability for controlling token selection (default: 0.95).
            repetition_penalty (float): The penalty for repeated tokens during generation (default: 1.0).
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.batch_size = batch_size

    def _batch_generate(self, messages: list, json_grammar):
        """
        Generate responses for a batch of messages using the language model.

        Args:
            messages (list): A list of input messages to process.
            json_grammar: The grammar constraints to apply during generation.

        Returns:
            list: A list of generated responses corresponding to the input messages.
        """
        all_responses = []
        for i in tqdm(range(0, len(messages), self.batch_size)):
            batch_messages = messages[i:i + self.batch_size]
            response = self.llm.generate(messages=batch_messages,
                                         grammar=json_grammar,
                                         max_new_tokens=self.max_new_tokens,
                                         temperature=self.temperature,
                                         top_p=self.top_p,
                                         repetition_penalty=self.repetition_penalty,
                                         )
            all_responses.extend(response)
        return all_responses
