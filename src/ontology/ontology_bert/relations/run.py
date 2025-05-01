from typing import List
from anytree import RenderTree
from src.ontology.ontology_bert.relations.relation_dataset import RelationDataset
from src.ontology.ontology_bert.relations.bert_rel_extractor import BertRelExtractor
from src.ontology.ontology_bert.relations.manager import BERTRelationExtractionManager
from logger import logger


def train_relation_extractor(size,
                             valid_frac=0.05,
                             batch_size=32,
                             bert_relation_extractor_path='bert_relation_extractor.pt',
                             categories: List = ['camera', 'backpack', 'cardigan', 'guitar', 'laptop'],
                             path='bert-train-data/relation_extraction_datasets',
                             random_state=42):
    """
    Train a relation extractor BERT model using the specified configuration.

    Args:
        size (int): The maximum size of the dataset to use for training.
        valid_frac (float): The fraction of the dataset to use for validation (default: 0.05).
        batch_size (int): The batch size for training (default: 32).
        bert_relation_extractor_path (str): The file path to save the trained BERT relation extractor (default: 'bert_relation_extractor.pt').
        categories (List): A list of categories to include in the training dataset (default: ['camera', 'backpack', 'cardigan', 'guitar', 'laptop']).
        path (str): The path to the directory containing the training datasets (default: 'bert-train-data/relation_extraction_datasets').
        random_state (int): The random seed for reproducibility (default: 42).
    """
    train_data, val_data = RelationDataset.from_files(
        categories=categories,
        valid_frac=valid_frac,
        size=size,
        path=path,
        random_state=random_state,
    )
    logger.info('Obtained dataset of size', len(train_data))
    BertRelExtractor.train_and_validate(
        train_data=train_data,
        valid_data=val_data,
        batch_size=batch_size,
        save_file=bert_relation_extractor_path,
    )


def run_relation_extraction(reviews_list,
                            root_name,
                            synsets,
                            synset_counts,
                            batch_size=32,
                            njobs=4,
                            bert_relation_extractor_path='bert_relation_extractor.pt'):
    """
    Run the relation extraction process using a pre-trained BERT relation extractor
    given reveiws list and previously extracted synsets.

    Args:
        reviews_list (list): A list of review texts to extract relations from.
        root_name (str): The root concept name for the ontology tree.
        synsets (dict): A dictionary mapping aspects to their synonym groups.
        synset_counts (dict): A dictionary mapping aspects to their total counts.
        batch_size (int): The batch size for processing reviews (default: 32).
        njobs (int): The number of parallel jobs to use for relation extraction (default: 4).
        bert_relation_extractor_path (str): The file path to the pre-trained BERT relation extractor (default: 'bert_relation_extractor.pt').

    Returns:
        anytree.Node: The root node of the constructed ontology tree.
    """
    extractor = BertRelExtractor.load_saved(
        path=bert_relation_extractor_path,
        batch_size=batch_size,
    )
    manager = BERTRelationExtractionManager(
        root_name=root_name,
        bert_rel_extractor=extractor,
        njobs=njobs,
    )

    tree = manager.extract(synset_counts, synsets, reviews_list)
    logger.info(RenderTree(tree))
    return tree
