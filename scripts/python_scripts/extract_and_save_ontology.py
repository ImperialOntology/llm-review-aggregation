"""
This script queries the database for reviews and aspects, performs concept and relation extraction,
and then saves the results back to the database.
"""

import os
import time
from argparse import ArgumentParser

from db.manager import DatabaseManager
import pandas as pd

from src.ontology.ontology_llm.concepts.run import (
    run_concept_extraction as run_concept_extraction_llm)
from src.ontology.ontology_llm.relations.run import (
    run_relation_extraction as run_relation_extraction_llm)
from src.ontology.ontology_bert.concepts.run import (
    run_concept_extraction as run_concept_extraction_bert)
from src.ontology.ontology_bert.relations.run import (
    run_relation_extraction as run_relation_extraction_bert)
from src.constants import CATEGORY_MAPPING
from src.ontology.tree_builder import TreeBuilder
from logger import logger


DBNAME = os.getenv("DBNAME")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-source-name", type=str, required=True)
    parser.add_argument('--category-name', type=str, default='Games', choices=list(CATEGORY_MAPPING.keys()),
                        help='Category name which is mapped to the corresponding '
                             'super-category and category in the datasets used (except for the Disneyland). \n'
                             'Here is the said mapping for your reference:\n'
                             f'{CATEGORY_MAPPING}')
    parser.add_argument("--bert-or-llm", type=str, required=True,
                        choices=["bert", "llm"],
                        help="Whether to use BERT or LLM for concept and relation extraction.")
    parser.add_argument("--bert-model-path", type=str, required=False,
                        default="path/to/bert/model",
                        help="Path to the BERT model to use for relation extraction.")
    parser.add_argument("--aspect-extraction-name", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--top-k-aspects-to-keep", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--njobs", type=int, required=False, default=4,
                        help="Number of parallel jobs to run for tokenisation in BERT aspect extraction.")
    return parser.parse_args()


def main():
    # query reviews from database
    logger.info("Querying reviews from database...")
    args = parse_args()
    db_manager = DatabaseManager(DBNAME, USER, PASSWORD, HOST, PORT)
    category_id, reviews = db_manager.query_all_reviews(args.data_source_name, args.category_name)
    reviews_list = [review[2] for review in reviews]

    # query aspects from database
    logger.info("Querying aspects from database...")
    aspect_extraction_id, aspect_frequency = db_manager.query_aspects_by_aspect_extraction(
        category_id, args.aspect_extraction_name)
    if not aspect_frequency:
        raise ValueError(f"No aspects found for aspect extraction {args.aspect_extraction_name}")

    # extract concepts
    category_metadata = CATEGORY_MAPPING[args.category_name]
    root_name = category_metadata["ontology_root"]

    if args.bert_or_llm == "llm":
        start = time.time()
        logger.info("Extracting concepts from reviews...")
        synset_counts, synsets = run_concept_extraction_llm(
            root_name,
            reviews_list,
            aspect_frequency,
            cache_dir=args.cache_dir,
            top_k_aspects_to_keep=args.top_k_aspects_to_keep,
            batch_size=args.batch_size,
        )
        end = time.time()
        logger.info(f"Concept extraction took {end - start:.2f} seconds.")
        logger.info(synset_counts, synsets)
        logger.info("Extracting relations from reviews...")
        start = end
        tree = run_relation_extraction_llm(
            reviews_list,
            root_name=root_name,
            synsets=synsets,
            synset_counts=synset_counts,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
        )
        end = time.time()
        logger.info(f"Relation extraction took {end - start:.2f} seconds.")
    elif args.bert_or_llm == "bert":
        logger.info("Extracting concepts from reviews...")
        start = time.time()
        synset_counts, synsets = run_concept_extraction_bert(
            root_name,
            reviews_list=reviews_list,
            aspect_frequency=aspect_frequency,
            cache_dir=args.cache_dir,
            njobs=args.njobs,
            top_k_aspects_to_keep=args.top_k_aspects_to_keep,
        )
        end = time.time()
        logger.info(f"Concept extraction took {end - start:.2f} seconds.")
        logger.info(synset_counts, synsets)
        logger.info("Extracting relations from reviews...")
        start = end
        tree = run_relation_extraction_bert(
            reviews_list,
            root_name=root_name,
            synsets=synsets,
            synset_counts=synset_counts,
            batch_size=args.batch_size,
            njobs=args.njobs,
            bert_relation_extractor_path=args.bert_model_path,
        )
        end = time.time()
        logger.info(f"Relation extraction took {end - start:.2f} seconds.")
    else:
        raise ValueError("bert-or-llm argument should be either 'bert' or 'llm'.")

    tree_dict = TreeBuilder.convert_tree_to_dict(tree)

    # save ontology tree to database
    logger.info("Saving ontology tree to database...")
    db_manager.insert_ontology_extraction(
        aspect_extraction_id=aspect_extraction_id,
        ontology_tree=tree_dict,
        synsets=synsets,
        name=args.name,
        method=args.method,
        description=args.description)


if __name__ == "__main__":
    main()
