from gensim.models import FastText
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from time import time
from src.constants import CACHE_DIR
from src.ontology.word_vectoriser_base import WordVectoriserBase
from logger import logger


class FastTextWrapper(WordVectoriserBase):
    def __init__(
        self,
        phrase_threshold: int = 1,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        seed: int = 42,
        load_model: bool = False,
        model_path: str = f'{CACHE_DIR}/fasttext.model'
    ):
        super().__init__(
            phrase_threshold=phrase_threshold,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            seed=seed,
            load_model=load_model,
            model_path=model_path
        )

    def learn_embeddings(self, reviews: list, save_model: bool = True) -> None:
        start = time()
        logger.info('Learning embeddings...')
        corpus = ' '.join(reviews)
        sentences = sent_tokenize(corpus)
        tok_sentences = [word_tokenize(sentence.lower())
                         for sentence in sentences]

        bigram_model = Phrases(
            tok_sentences, min_count=self.min_count, threshold=self.phrase_threshold)
        trigram_model = Phrases(
            bigram_model[tok_sentences], min_count=self.min_count, threshold=self.phrase_threshold)
        trigram_phraser = Phraser(trigram_model)

        model = FastText(
            sentences=trigram_phraser[tok_sentences],
            sg=1,  # skipgram: 1; CBOW: 0
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            word_ngrams=1,  # trigrams
            workers=self.workers,
            seed=self.seed
        )

        self.word_vectors = model.wv
        if save_model:
            with open(self.model_path, "wb") as file:
                pickle.dump(model, file)
        del model

        end = time()
        self.is_fit = True
        logger.info('Learning embeddings took {} seconds'.format(end - start))
