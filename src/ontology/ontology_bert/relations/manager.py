import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool

from transformers import BertTokenizer

from src.ontology.ontology_bert.relations.bert_rel_extractor import BertRelExtractor
from src.ontology.ontology_bert.relations.relation_dataset import RelationDataset
from src.ontology.tree_builder import TreeBuilder
from src.ontology.ontology_bert.helpers import relation_instances_for_text
from src.ontology.ontology_bert.phrase_tokenizer import PhraseTokenizer
from src.constants import TRAINED_WEIGHTS
from logger import logger

tokenizer = BertTokenizer.from_pretrained(TRAINED_WEIGHTS)


class BERTRelationExtractionManager:
    """Manage relation extraction by BERT extractor.
    """

    def __init__(self,
                 root_name: str,
                 bert_rel_extractor: BertRelExtractor,
                 njobs: int = 4,
                 ) -> None:
        self.bert_rel_extractor = bert_rel_extractor
        self.njobs = njobs
        self.tree_builder = TreeBuilder(root_name)

    def get_relations(self, reviews_list, synsets):
        logger.info('  select phrases for relation extraction...')
        concepts = list(synsets.keys())
        with Pool(self.njobs) as pool:
            sentences, _ = PhraseTokenizer.extract_sentences_and_phrases(pool, reviews_list)
            instances = filter(lambda i: i is not None, pool.starmap(relation_instances_for_text,
                                                                     zip(itertools.repeat(tokenizer, len(sentences)),
                                                                         itertools.repeat(concepts, len(sentences)),
                                                                         itertools.repeat(synsets, len(sentences)),
                                                                         sentences)))
        df = pd.DataFrame(instances, columns=['tokens', 'entity_indices', 'entity_labels'])

        logger.info('  extracting relations with BERT...')
        dataset = RelationDataset.for_extraction(df)
        prob_matrix, _ = self.bert_rel_extractor.extract_relations(len(concepts), data=dataset)
        return concepts, np.array(prob_matrix)

    def extract(self, concept_counts, synsets, reviews_list):
        """
        Run the full pipeline for extracting relations between concepts and building the ontology tree.

        Returns the ontology tree as anytree Node object."""

        reviews_list = [rev for rev in reviews_list if not pd.isnull(rev)]  # remove NaN reviews
        concepts, meronym_matrix = self.get_relations(reviews_list, synsets)

        filtered_norm_matrix, filtered_concepts = self.tree_builder.get_relatedness(
            concepts, concept_counts, meronym_matrix)

        self.tree_builder.build_tree(filtered_norm_matrix, filtered_concepts)

        return self.tree_builder.tree
