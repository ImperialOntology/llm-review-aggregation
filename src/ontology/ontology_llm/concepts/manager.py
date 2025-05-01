from collections import Counter
from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.ontology.ontology_llm.base.manager_base import ExtractionManager
from src.ontology.ontology_llm.concepts.prompt_manager import ConceptPromptManager
from src.ontology.ontology_llm.concepts.fasttext_wrapper import FastTextWrapper
from src.ontology.synset_extractor import SynsetExtractor
from logger import logger


class PromptedConceptExtractionManager(ExtractionManager):
    """
    This class is responsible for extracting concepts from a list of aspects using a language model
    and a prompt manager. It utilizes a vectorizer to cluster aspects into synsets and filters out
    non-conceptual aspects based on the model's responses.
    """

    def __init__(self,
                 root_name,
                 llm: HuggingFaceLLMWrapper,
                 prompt_manager: ConceptPromptManager,
                 vectorizer: FastTextWrapper,
                 synset_extractor: SynsetExtractor,
                 top_k_aspects_to_keep=50,  # number of top frequent aspects to consider
                 batch_size=32,
                 max_new_tokens=200,
                 temperature=0.3,
                 top_p=0.95,
                 repetition_penalty=1.0,
                 ) -> None:
        """
        Initialize the PromptedConceptExtractionManager.
        Args:

            root_name (str): The root name for the concept extraction process.
            llm (HuggingFaceLLMWrapper): The language model wrapper for generating responses.
            prompt_manager (ConceptPromptManager): The prompt manager for handling prompts.
            vectorizer (FastTextWrapper): The vectorizer for clustering aspects into synsets.
            synset_extractor (SynsetExtractor): The extractor for obtaining synsets from aspects.
            top_k_aspects_to_keep (int): The number of top frequent aspects to consider (default: 50).
            batch_size (int): The number of messages to process in a single batch (default: 32).
            max_new_tokens (int): The maximum number of tokens to generate for each response (default: 200).
            temperature (float): The sampling temperature for controlling randomness in generation (default: 0.3).
            top_p (float): The nucleus sampling probability for controlling token selection (default: 0.95).
            repetition_penalty (float): The penalty for repeated tokens during generation (default: 1.0).

        """
        super().__init__(
            llm=llm,
            prompt_manager=prompt_manager,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        self.root_name = root_name
        self.vectorizer = vectorizer
        self.synset_extractor = synset_extractor
        self.top_k_aspects_to_keep = top_k_aspects_to_keep

    def _get_synsets(self, aspect_counts):
        """
        Extract synsets of concepts from the given aspect counts

        Parameters: 
            aspect_counts: Dict[str, int]
                Counts of each aspect subject to clustering {key: aspect, value: count}

        Returns:
            synset_counts: Dict[str, int]
                Counts of each synset;  {key: concept, value: count}
            synsets: Dict[str, Set[str]]
                Synonyms grouped into synsets; {key: concept, value: list of synonym aspects}
        """
        synset_counts, synsets = self.synset_extractor.cluster_aspects(
            self.root_name, aspect_counts, self.vectorizer)

        return synset_counts, synsets

    def extract_concepts(self, aspect_frequency):
        """
        Extract concepts from the given list of lists of aspects

        Parameters:
            aspect_frequency: Dict[str, int]
                Counts of each aspect; {key: aspect, value: count}

        Returns:
            synset_counts: Dict[str, int]
                Counts of each synset;  {key: concept, value: count}
            synsets: Dict[str, Set[str]]
                Synonyms grouped into synsets; {key: concept, value: list of synonym aspects}
        """
        # get top aspects
        aspect_counter = Counter(aspect_frequency)
        top_aspect_counts = {aspect: count for aspect, count
                             in aspect_counter.most_common(self.top_k_aspects_to_keep)}
        # get synsets
        synset_counts, synsets = self._get_synsets(top_aspect_counts)

        # filter out non-concepts
        candidate_concepts = [concept for concept in synsets]

        logger.info('Extracting concepts...')
        messages = [self.prompt_manager.get_prompt(self.root_name, candidate_concept)
                    for candidate_concept in candidate_concepts]
        responses = self._batch_generate(messages, self.prompt_manager.json_grammar)
        is_concept_list = [self.prompt_manager.process_response(response) for response in responses]

        concepts = [concept
                    for concept, is_concept in zip(candidate_concepts, is_concept_list)
                    if is_concept]
        if self.root_name not in concepts:
            concepts.append(self.root_name)
        synset_counts = {concept: synset_counts[concept]
                         for concept in concepts}
        synsets = {concept: synsets[concept] for concept in concepts}

        return synset_counts, synsets
