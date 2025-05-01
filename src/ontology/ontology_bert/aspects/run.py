from typing import List

from src.ontology.ontology_bert.aspects.bert_entity_extractor import BertEntityExtractor
from src.ontology.ontology_bert.aspects.entity_dataset import EntityDataset
from src.ontology.ontology_bert.aspects.manager import BERTAspectExtractionManager
from logger import logger


def train_term_extractor(size,
                         valid_frac=0.05,
                         batch_size=32,
                         bert_entity_extractor_path='bert_entity_extractor.pt',
                         categories: List = ['camera', 'backpack', 'cardigan', 'guitar', 'laptop'],
                         path='bert-train-data/term_extraction_datasets',
                         random_state=42):
    """
    Train a term extractor BERT model using the specified configuration.

    Args:
        size (int): The maximum size of the dataset to use for training.
        valid_frac (float): The fraction of the dataset to use for validation (default: 0.05).
        batch_size (int): The batch size for training (default: 32).
        bert_entity_extractor_path (str): The file path to save the trained BERT entity extractor (default: 'bert_entity_extractor.pt').
        categories (List): A list of categories to include in the training dataset (default: ['camera', 'backpack', 'cardigan', 'guitar', 'laptop']).
        path (str): The path to the directory containing the training datasets (default: 'bert-train-data/term_extraction_datasets').
        random_state (int): The random seed for reproducibility (default: 42).
    """
    train_data, val_data = EntityDataset.from_files(
        categories=categories,
        valid_frac=valid_frac,
        size=size,
        path=path,
        random_state=random_state,
    )
    logger.info('Obtained dataset of size', len(train_data))
    BertEntityExtractor.train_and_validate(
        train_data=train_data,
        valid_data=val_data,
        batch_size=batch_size,
        save_file=bert_entity_extractor_path,
    )


def run_aspect_extraction(reviews_list: List[str],
                          batch_size=32,
                          bert_entity_extractor_path='bert_entity_extractor.pt',
                          njobs=4):
    """
    Run the aspect extraction process using a pre-trained BERT entity extractor.

    Args:
        reviews_list (List[str]): A list of review texts to extract aspects from.
        batch_size (int): The batch size for processing reviews (default: 32).
        bert_entity_extractor_path (str): The file path to the pre-trained BERT entity extractor (default: 'bert_entity_extractor.pt').
        njobs (int): The number of parallel jobs to use for aspect extraction (default: 4).

    Returns:
        Counter: A counter object mapping aspects to their frequencies across all reviews.
    """
    extractor = BertEntityExtractor.load_saved(
        path=bert_entity_extractor_path,
        batch_size=batch_size,
    )
    extraction_manager = BERTAspectExtractionManager(
        bert_entity_extractor=extractor,
        njobs=njobs,
    )
    aspect_counts = extraction_manager.extract_aspects_from_reviews(reviews_list)

    return aspect_counts
