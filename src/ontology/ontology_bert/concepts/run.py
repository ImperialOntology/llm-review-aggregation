from src.ontology.ontology_bert.concepts.word2vec_wrapper import Word2VecWrapper
from src.ontology.synset_extractor import SynsetExtractor
from src.ontology.ontology_bert.concepts.manager import BERTConceptExtractionManager


def run_concept_extraction(root_name,
                           reviews_list,
                           aspect_frequency,
                           cache_dir,
                           top_k_aspects_to_keep=100,
                           njobs=4):
    """
    Run concept extraction from provided aspects and a list of reviews using a Word2Vec vectorizer, 
    a synset extractor, and a BERT-based concept extraction manager.

    Args:
        root_name (str): The root concept name to prioritize in the extraction process.
        reviews_list (list): A list of review texts to extract concepts from.
        aspect_frequency (dict): A dictionary mapping aspects to their frequencies.
        cache_dir (str): The directory to cache the Word2Vec model.
        top_k_aspects_to_keep (int): The maximum number of top aspects to retain (default: 100).
        njobs (int): The number of parallel jobs to use for vectorization (default: 4).

    Returns:
        tuple: A tuple containing:
            - synset_counts (dict): A dictionary mapping aspects to their total counts.
            - synsets (dict): A dictionary mapping aspects to their synonym groups.
    """
    vectorizer = Word2VecWrapper(
        model_path=f'{cache_dir}/word2vec.model',
        workers=njobs,
    )
    vectorizer.learn_embeddings(reviews_list, save_model=True)
    synset_extractor = SynsetExtractor(
        similarity_threshold=0.20,
        num_clustering_levels=3,  # Set to same value as in legacy code
    )
    concept_manager = BERTConceptExtractionManager(
        root_name=root_name,
        vectorizer=vectorizer,
        synset_extractor=synset_extractor,
        top_k_aspects_to_keep=top_k_aspects_to_keep,
    )
    synset_counts, synsets = concept_manager.extract_concepts(aspect_frequency)
    return synset_counts, synsets
