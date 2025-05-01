from nltk.tokenize import sent_tokenize, word_tokenize
import itertools
from multiprocessing import Pool
from tqdm import tqdm
from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.ontology.ontology_llm.base.manager_base import ExtractionManager
from src.ontology.ontology_llm.relations.prompt_manager import RelationPromptManager
import numpy as np
from src.ontology.tree_builder import TreeBuilder
from logger import logger


class PromptedRelationExtractionManager(ExtractionManager):
    """Extracts relations between provided concepts according to review textx
    and thus constructs ontology trees.
    The relation extraction is done by prompting the LLM with a relation extraction prompt.
    """

    def __init__(self,
                 root_name,
                 llm: HuggingFaceLLMWrapper,
                 prompt_manager: RelationPromptManager,
                 batch_size=32,
                 max_new_tokens=100,
                 temperature=0.3,
                 top_p=0.95,
                 repetition_penalty=1.0,
                 ) -> None:
        super().__init__(
            llm=llm,
            prompt_manager=prompt_manager,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        self.tree_builder = TreeBuilder(root_name)

    @staticmethod
    def _get_relation_extraction_texts(aspects, synsets, sentence):
        """
        Checks if sentence contains exactly 2 aspects from different synsets.
        If yes, returns the sentence and the aspects.
        Otherwise, returns None.

        parameters:
        aspects: the list of main concepts correponding to the synsets
        synsets: the synsets of the main concepts, {key: concept, value: syn list}
        sentence: the sentence to check for aspects
        """
        words = word_tokenize(sentence)
        found_aspects = []
        for synset_idx, synset in enumerate(aspects):
            for aspect in synsets[synset]:
                aspect_words = word_tokenize(aspect)
                if (len(aspect_words) == 1 and aspect in words) or (len(aspect_words) > 1 and aspect in sentence):
                    found_aspects.append((synset_idx, aspect))

        # Check if exactly 2 aspects are found
        if len(found_aspects) == 2:
            # Check if aspects are not from the same synset
            found_synsets = set()
            for synset_idx, _ in found_aspects:
                found_synsets.add(aspects[synset_idx])

            if len(found_synsets) > 1:
                synset_idx1, aspect1 = found_aspects[0]
                synset_idx2, aspect2 = found_aspects[1]
                return sentence, aspect1, synset_idx1, aspect2, synset_idx2

        return None

    def extract_relations(self, synsets, reviews_list):
        """
        Extracts relations between provided concepts according to review texts.
        """
        logger.info('    Selecting review sentences for relation extraction...')
        concepts = list(synsets.keys())
        with Pool(4) as pool:
            sentences = list(itertools.chain.from_iterable(
                pool.map(sent_tokenize, reviews_list)))
            sentences = list(itertools.chain.from_iterable(
                pool.map(str.splitlines, sentences)))
            relation_texts = filter(lambda i: i is not None, pool.starmap(
                self._get_relation_extraction_texts,
                zip(itertools.repeat(concepts, len(sentences)),
                    itertools.repeat(synsets, len(sentences)),
                    sentences)))

        meronym_matrix = np.zeros((len(concepts), len(concepts)))

        logger.info('    Grouping relation texts to process...')
        # first, let's group by aspect1 and aspect2 to be able to process in batch
        batched_relation_texts = dict()
        for sentence, aspect1, synset_idx1, aspect2, synset_idx2 in tqdm(relation_texts):
            if (aspect1, synset_idx1, aspect2, synset_idx2) not in batched_relation_texts:
                batched_relation_texts[(aspect1, synset_idx1, aspect2, synset_idx2)] = []
            batched_relation_texts[(aspect1, synset_idx1, aspect2, synset_idx2)].append(sentence)

        logger.info('    Extracting relations...')

        for (aspect1, synset_idx1, aspect2, synset_idx2), sentences in tqdm(batched_relation_texts.items()):
            dummy_sentence = sentences[0]
            _, json_grammar = self.prompt_manager.get_prompt(dummy_sentence, aspect1, aspect2)

            messages = []
            for sentence in sentences:
                message, _ = self.prompt_manager.get_prompt(sentence, aspect1, aspect2)
                messages.append(message)
            responses = self._batch_generate(messages, json_grammar)

            for response in responses:
                formatted_relations = self.prompt_manager.process_response(response, aspect1, aspect2)
                if formatted_relations:
                    is_first_aspect_child, score = formatted_relations
                    if is_first_aspect_child:
                        meronym_matrix[synset_idx1][synset_idx2] += score
                    else:
                        meronym_matrix[synset_idx2][synset_idx1] += score

        return concepts, np.array(meronym_matrix)

    def extract(self, concept_counts, synsets, reviews_list):
        """
        Run the full pipeline for extracting relations between concepts and building the ontology tree.

        Returns the ontology tree as as anytree Node object.
        """
        concepts, meronym_matrix = self.extract_relations(
            synsets, reviews_list)

        filtered_norm_matrix, filtered_concepts = self.tree_builder.get_relatedness(
            concepts, concept_counts, meronym_matrix)

        self.tree_builder.build_tree(filtered_norm_matrix, filtered_concepts)

        return self.tree_builder.tree
