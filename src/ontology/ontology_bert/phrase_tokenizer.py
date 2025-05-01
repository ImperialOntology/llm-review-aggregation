from src.ontology.ontology_bert.helpers import ngrams
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import word_tokenize, sent_tokenize
import itertools
from src.constants import PHRASE_THRESHOLD


class PhraseTokenizer:
    """ 
    Tokenize text into phrases using the Phrases model from Gensim.
    """
    @staticmethod
    def filter_underscore(text):
        return text.replace('_', ' ').lower()

    @staticmethod
    def extract_sentences_and_phrases(pool, reviews: list):
        """Extract sentences from a list of reviews.

        Returns: List of lists of sentences extracted from the reviews.
        """
        sentences = list(itertools.chain.from_iterable(pool.map(sent_tokenize, reviews)))
        sentences = list(itertools.chain.from_iterable(pool.map(str.splitlines, sentences)))
        sentences = pool.map(PhraseTokenizer.filter_underscore, sentences)
        phrases = pool.map(word_tokenize, sentences)

        return sentences, phrases

    @staticmethod
    def extract_ngrams(pool, phrases, min_count=1):
        bigram = Phrases(phrases, min_count=min_count, threshold=PHRASE_THRESHOLD)
        trigram = Phrases(bigram[phrases], min_count=min_count, threshold=PHRASE_THRESHOLD)
        phraser = Phraser(trigram)
        ngram_phrases = pool.starmap(ngrams, zip(phrases, itertools.repeat(phraser, len(phrases))))
        return ngram_phrases
