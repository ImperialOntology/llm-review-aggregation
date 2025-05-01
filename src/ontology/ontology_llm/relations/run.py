from anytree import RenderTree

from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.constants import PROMPT_BASED_LLM_MODEL_NAME

from src.ontology.ontology_llm.relations.prompt_manager import RelationPromptManager
from src.ontology.ontology_llm.relations.manager import PromptedRelationExtractionManager
from logger import logger


def run_relation_extraction(reviews_list, root_name,
                            synsets, synset_counts,
                            cache_dir, model_name=PROMPT_BASED_LLM_MODEL_NAME,
                            batch_size=8):
    """
    Run relation extraction to construct an ontology tree from synsets and reviews.

    Args:
        reviews_list (list): A list of review texts to extract relations from.
        root_name (str): The root concept name for the ontology tree.
        synsets (dict): A dictionary mapping aspects to their synonym groups.
        synset_counts (dict): A dictionary mapping aspects to their total counts.
        cache_dir (str): The directory to cache the language model and tokenizer.
        model_name (str): The name of the language model to use (default: PROMPT_BASED_LLM_MODEL_NAME).
        batch_size (int): The batch size for processing reviews (default: 8).

    Returns:
        anytree.Node: The root node of the constructed ontology tree.
    """
    llm = HuggingFaceLLMWrapper(model_name=model_name, cache_dir=cache_dir)
    prompt_manager = RelationPromptManager()
    relation_manager = PromptedRelationExtractionManager(
        root_name, llm, prompt_manager, batch_size=batch_size)

    # extract relations and construct ontology tree
    tree = relation_manager.extract(synset_counts, synsets, reviews_list)
    logger.info(RenderTree(tree))

    return tree
