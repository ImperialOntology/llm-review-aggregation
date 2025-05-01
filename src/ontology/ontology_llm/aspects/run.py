from collections import Counter

from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.constants import PROMPT_BASED_LLM_MODEL_NAME

from src.ontology.ontology_llm.aspects.manager import PromptedAspectExtractionManager
from src.ontology.ontology_llm.aspects.prompt_manager import SentimentAspectPromptManager


def run_aspect_extraction(reviews_list, output_file, cache_dir, batch_size=8, model_name=PROMPT_BASED_LLM_MODEL_NAME):
    """
    Run aspect extraction on a list of reviews using a language model.

    Args:
        reviews_list (list): A list of review texts to extract aspects from.
        output_file (str): The path to the file where the extracted aspects will be saved.
        cache_dir (str): The directory to cache the language model and tokenizer.
        batch_size (int): The batch size for processing reviews (default: 8).
        model_name (str): The name of the language model to use (default: PROMPT_BASED_LLM_MODEL_NAME).

    Returns:
        Counter: A counter object mapping aspects to their frequencies across all reviews.
    """
    llm = HuggingFaceLLMWrapper(model_name, cache_dir=cache_dir)
    prompt_manager = SentimentAspectPromptManager()
    aspect_manager = PromptedAspectExtractionManager(llm, prompt_manager, batch_size=batch_size)
    aspect_lists = aspect_manager.extract_aspects_from_reviews(
        reviews_list, output_file=output_file)
    aspect_counter = Counter([aspect for aspects in aspect_lists for aspect in aspects])
    return aspect_counter
