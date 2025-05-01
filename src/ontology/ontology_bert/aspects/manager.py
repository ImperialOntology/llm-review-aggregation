import pandas as pd
import itertools
from multiprocessing import Pool
from collections import Counter


from transformers import BertTokenizer

from src.ontology.ontology_bert.aspects.bert_entity_extractor import BertEntityExtractor
from src.ontology.ontology_bert.aspects.entity_dataset import EntityDataset
from src.ontology.ontology_bert.helpers import entity_instances_for_text, get_nouns
from src.ontology.ontology_bert.base.manager_base import BERTExtractionManager
from src.ontology.ontology_bert.phrase_tokenizer import PhraseTokenizer
from src.constants import (
    TRAINED_WEIGHTS,
    ENTITY_PROB_THRESHOLD,
    N_ASPECTS)
from logger import logger


tokenizer = BertTokenizer.from_pretrained(TRAINED_WEIGHTS)


class BERTAspectExtractionManager(BERTExtractionManager):
    """Manage aspect extraction by using the BERT approach.
    """

    def __init__(self,
                 bert_entity_extractor: BertEntityExtractor,
                 njobs: int = 4) -> None:
        self.bert_entity_extractor = bert_entity_extractor
        self.njobs = njobs

    def _count_nouns(self, pool, phrases, ngram_phrases):
        nouns = itertools.chain.from_iterable(pool.starmap(get_nouns, zip(phrases, ngram_phrases)))
        return Counter(nouns)

    def get_aspects(self, pool, counter, sentences):
        # take N_ASPECTS most common terms
        entity_counts = counter.most_common()[:N_ASPECTS]
        entities = [entity for entity, count in entity_counts]

        logger.info('  preparing entity texts for BERT...')
        instances = filter(lambda i: i is not None, pool.starmap(entity_instances_for_text,
                                                                 zip(itertools.repeat(tokenizer, len(sentences)),
                                                                     itertools.repeat(entities, len(sentences)),
                                                                     sentences)))
        df = pd.DataFrame(instances, columns=['tokens', 'entity_idx', 'entity'])

        logger.info('  extracting entities with BERT...')
        dataset = EntityDataset.for_extraction(df)
        probs = self.bert_entity_extractor.extract_aspect_probabilities(entities, dataset)

        aspect_counts = {entity: count for entity, count in entity_counts
                         if probs[entity] is not None
                         and probs[entity] >= ENTITY_PROB_THRESHOLD}
        return Counter(aspect_counts)

    def extract_aspects_from_reviews(self, reviews: list):
        """Extract aspects from a list of reviews.

        Returns: List of lists of aspects extracted from the reviews.

        The aspects are also written to a file in the output_file path.
        """
        reviews = [rev for rev in reviews if not pd.isnull(rev)]  # remove NaN reviews
        with Pool(self.njobs) as pool:
            sentences, phrases = PhraseTokenizer.extract_sentences_and_phrases(pool, reviews)
            ngram_phrases = PhraseTokenizer.extract_ngrams(pool, phrases)
            noun_counter = self._count_nouns(pool, phrases, ngram_phrases)
            aspect_counts = self.get_aspects(pool, noun_counter, sentences)

        return aspect_counts
