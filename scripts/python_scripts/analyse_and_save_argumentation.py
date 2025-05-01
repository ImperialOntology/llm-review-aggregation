"""
This script extracts argumentative analysis from reviews and saves it to the database.
"""

import os
import time
from argparse import ArgumentParser

from db.manager import DatabaseManager
import pandas as pd

from src.argumentation.arg_framework.run import (
    get_arguments,
    run_argumentative_analysis,
    convert_dict_to_tree,
)
from src.constants import CATEGORY_MAPPING
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
    parser.add_argument("--aspect-extraction-name", type=str, required=True)
    parser.add_argument("--ontology-extraction-name", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--ba-model-path", type=str, required=True)
    return parser.parse_args()


def main():
    # query reviews from database
    logger.info("Querying reviews from database...")
    args = parse_args()
    db_manager = DatabaseManager(DBNAME, USER, PASSWORD, HOST, PORT)
    category_id, reviews = db_manager.query_all_reviews(args.data_source_name, args.category_name)
    reviews_df = pd.DataFrame(reviews, columns=["id", "product_id", "content", "rating"])

    ontology_extraction_id, tree_dict, synsets = db_manager.query_ontology_by_ontology_extraction(
        category_id,
        args.aspect_extraction_name,
        args.ontology_extraction_name,
    )

    tree = convert_dict_to_tree(tree_dict)

    # extract argumentation frameworks
    logger.info("Extracting argumentation frameworks...")
    start = time.time()
    frameworks = run_argumentative_analysis(
        reviews_df=reviews_df,
        tree=tree,
        synsets=synsets,
        model_path=args.ba_model_path
    )
    end = time.time()
    logger.info(f'Argumentation frameworks extracted in {end - start:.2f} seconds')

    arguments_df = get_arguments(frameworks, tree)

    # save argumentative analysis to database
    logger.info("Saving argumentative analysis to database...")
    db_manager.insert_argumentative_analysis(
        ontology_extraction_id=ontology_extraction_id,
        argumentative_analysis_df=arguments_df,
        name=args.name,
        method=args.method,
        description=args.description)


if __name__ == "__main__":
    main()
