from abc import ABC
import pickle
from nltk.stem import WordNetLemmatizer
from src.constants import CACHE_DIR


class WordVectoriserBase(ABC):
    """
    A base class for word vectorization, providing methods for learning embeddings,
    calculating similarity, and checking synonymy between terms.

    Attributes:
        phrase_threshold (int): The threshold for phrase detection.
        vector_size (int): The size of the word vectors.
        window (int): The maximum distance between the current and predicted word.
        min_count (int): Words with lower frequency than this are ignored.
        workers (int): The number of worker threads to use for training.
        seed (int): The random seed for reproducibility.
        load_model (bool): Whether to load a pre-trained model.
        model_path (str): The path to the pre-trained model file.
        word_vectors: The word vectors generated or loaded.
        wnl (WordNetLemmatizer): A lemmatizer for processing words.
        is_fit (bool): Whether the model has been trained or not.
    """

    def __init__(
        self,
        phrase_threshold: int = 1,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        seed: int = 42,
        load_model: bool = False,
        model_path: str = f'{CACHE_DIR}/word_vectoriser.model'
    ):
        """
        Initialize the WordVectoriserBase.

        Args:
            phrase_threshold (int): The threshold for phrase detection (default: 1).
            vector_size (int): The size of the word vectors (default: 100).
            window (int): The maximum distance between the current and predicted word (default: 5).
            min_count (int): Words with lower frequency than this are ignored. (default: 1).
            workers (int): The number of worker threads to use for training (default: 4).
            seed (int): The random seed for reproducibility (default: 42).
            load_model (bool): Whether to load a pre-trained model (default: False).
            model_path (str): The path to the pre-trained model file (default: CACHE_DIR/word_vectoriser.model).
        """
        self.word_vectors = None
        self.phrase_threshold = phrase_threshold
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.seed = seed
        self.wnl = WordNetLemmatizer()
        self.is_fit = False
        self.model_path = model_path

        if load_model:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                self.word_vectors = model.wv
                del model
            self.is_fit = True

    def learn_embeddings(self) -> None:
        """
        Abstract method for learning word embeddings.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
        """
        raise NotImplementedError('learn_embeddings method must be implemented in the subclass')

    def similarity(self, t1, t2):
        """
        Calculate the similarity between two terms.

        Args:
            t1 (str): The first term.
            t2 (str): The second term.

        Returns:
            float: The similarity score between the two terms. Returns 1 if the terms are identical
            or their lemmatized forms are identical. Otherwise, calculates the relative cosine similarity.
        """
        if t1 == t2 or self.wnl.lemmatize(t1) == self.wnl.lemmatize(t2):
            return 1
        return self.word_vectors.relative_cosine_similarity(t1, t2) + self.word_vectors.relative_cosine_similarity(t2, t1)

    def are_syns(self, t1, t2, threshold):
        """
        Check if two terms are synonyms based on their similarity.

        Args:
            t1 (str): The first term.
            t2 (str): The second term.
            threshold (float): The similarity threshold to consider the terms as synonyms.

        Returns:
            bool: True if the terms are synonyms (based on the threshold or lemmatized equality), False otherwise.
        """
        if t1 == t2 or self.wnl.lemmatize(t1) == self.wnl.lemmatize(t2):
            return True
        else:
            return self.similarity(t2, t1) >= threshold
