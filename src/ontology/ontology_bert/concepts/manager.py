from collections import Counter

from src.ontology.synset_extractor import SynsetExtractor
from src.ontology.ontology_bert.concepts.word2vec_wrapper import Word2VecWrapper
from src.ontology.ontology_bert.base.manager_base import BERTExtractionManager
from src.constants import N_ASPECTS


class BERTConceptExtractionManager(BERTExtractionManager):
    """Manage concept extraction by using the BERT approach.
    """

    def __init__(self,
                 root_name,
                 vectorizer: Word2VecWrapper,
                 synset_extractor: SynsetExtractor,
                 top_k_aspects_to_keep=N_ASPECTS) -> None:
        self.root_name = root_name
        self.vectorizer = vectorizer
        self.synset_extractor = synset_extractor
        self.top_k_aspects_to_keep = top_k_aspects_to_keep

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
        synset_counts, synsets = self.synset_extractor.cluster_aspects(
            self.root_name, top_aspect_counts, self.vectorizer)

        return synset_counts, synsets
