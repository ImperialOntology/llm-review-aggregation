import pickle
from multiprocessing import Pool
from time import time
from gensim.models import Word2Vec
from src.constants import CACHE_DIR
from src.ontology.word_vectoriser_base import WordVectoriserBase
from src.ontology.ontology_bert.phrase_tokenizer import PhraseTokenizer
from logger import logger


class Word2VecWrapper(WordVectoriserBase):
    def __init__(
        self,
        phrase_threshold: int = 4,
        vector_size: int = 300,
        window: int = 4,
        min_count: int = 1,
        workers: int = 4,
        seed: int = 42,
        load_model: bool = False,
        model_path: str = f'{CACHE_DIR}/word2vec.model'
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

        with Pool(self.workers) as pool:
            _, phrases = PhraseTokenizer.extract_sentences_and_phrases(pool, reviews)
            ngram_phrases = PhraseTokenizer.extract_ngrams(pool, phrases, min_count=self.min_count)

        model = Word2Vec(ngram_phrases,
                         vector_size=self.vector_size,
                         window=self.window,
                         min_count=self.min_count,
                         epochs=20)

        self.word_vectors = model.wv
        if save_model:
            with open(self.model_path, "wb") as file:
                pickle.dump(model, file)
        del model

        end = time()
        self.is_fit = True
        logger.info('Learning embeddings took {} seconds'.format(end - start))
