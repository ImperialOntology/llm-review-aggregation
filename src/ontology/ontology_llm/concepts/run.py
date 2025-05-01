from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.constants import PROMPT_BASED_LLM_MODEL_NAME

from src.ontology.ontology_llm.concepts.prompt_manager import ConceptPromptManager
from src.ontology.ontology_llm.concepts.manager import PromptedConceptExtractionManager
from src.ontology.synset_extractor import SynsetExtractor
from src.ontology.ontology_llm.concepts.fasttext_wrapper import FastTextWrapper


def run_concept_extraction(root_name, reviews_list, aspect_frequency, cache_dir, top_k_aspects_to_keep=50, model_name=PROMPT_BASED_LLM_MODEL_NAME, batch_size=8):
    """
    Run concept extraction from a list of reviews using a language model, vectorizer, and synset extractor.

    Args:
        root_name (str): The root concept name to prioritize in the extraction process.
        reviews_list (list): A list of review texts to extract concepts from.
        aspect_frequency (dict): A dictionary mapping aspects to their frequencies.
        cache_dir (str): The directory to cache the language model, tokenizer, and vectorizer.
        top_k_aspects_to_keep (int): The maximum number of top aspects to retain (default: 50).
        model_name (str): The name of the language model to use (default: PROMPT_BASED_LLM_MODEL_NAME).
        batch_size (int): The batch size for processing reviews (default: 8).

    Returns:
        tuple: A tuple containing:
            - synset_counts (dict): A dictionary mapping aspects to their total counts.
            - synsets (dict): A dictionary mapping aspects to their synonym groups.
    """
    llm = HuggingFaceLLMWrapper(model_name=model_name, cache_dir=cache_dir)
    prompt_manager = ConceptPromptManager()
    vectorizer = FastTextWrapper(model_path=f'{cache_dir}/fasttext.model')
    vectorizer.learn_embeddings(reviews_list, save_model=False)
    synset_extractor = SynsetExtractor()
    concept_manager = PromptedConceptExtractionManager(
        root_name,
        llm,
        prompt_manager,
        vectorizer,
        synset_extractor,
        top_k_aspects_to_keep=top_k_aspects_to_keep,
        batch_size=batch_size,
    )

    synset_counts, synsets = concept_manager.extract_concepts(aspect_frequency)
    return synset_counts, synsets
