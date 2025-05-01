"""
This script queries the database for ontology trees, runs the LLM to judge them, 
and updates the database with the scores.
"""

import os
import time
from argparse import ArgumentParser

from db.manager import DatabaseManager
from src.llm_judge.run import run_ontology_judging
from src.llm_judge.llm_manager import Gemini
from src.constants import CATEGORY_MAPPING
from logger import logger

DBNAME = os.getenv("DBNAME")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-source-name", type=str, required=True, help="Name of the data source")
    parser.add_argument('--category-name', type=str, default='Games', choices=list(CATEGORY_MAPPING.keys()),
                        help='Category name which is mapped to the corresponding '
                             'super-category and category in the datasets used (except for the Disneyland). \n'
                             'Here is the said mapping for your reference:\n'
                             f'{CATEGORY_MAPPING}')
    parser.add_argument("--aspect-extraction-name", type=str, required=True, help="Name of the aspect extraction")
    parser.add_argument("--ontology-extraction-name", type=str, required=True,
                        help="Name of the ontology extraction to judge")
    parser.add_argument("--save-to-db", type=str, required=True, choices=["true", "false"],
                        help="Whether to save results to database")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize database manager
    db_manager = DatabaseManager(DBNAME, USER, PASSWORD, HOST, PORT)

    # Initialize LLM
    llm = Gemini(gemini_api_key=GEMINI_API_KEY)

    # Get category ID from data source name and category name
    data_source_id = db_manager.check_if_data_source_exists(args.data_source_name)
    if data_source_id is None:
        raise ValueError(f"No data source found for '{args.data_source_name}'")
    category_id = db_manager.check_if_category_exists(args.category_name, data_source_id)
    if category_id is None:
        raise ValueError(f"No category found for data source '{args.data_source_name}' "
                         f"data source ID '{data_source_id}' "
                         f"and category '{args.category_name}'")

    logger.info(f"Judging ontology for category '{args.category_name}' (ID: {category_id}) "
                f"and aspect extraction '{args.aspect_extraction_name}' "
                f"and ontology extraction '{args.ontology_extraction_name}'.")

    # Get ontology extraction ID and tree from database
    ontology_extraction_id, ontology_tree, synsets = db_manager.query_ontology_by_ontology_extraction(
        category_id,
        args.aspect_extraction_name,
        args.ontology_extraction_name
    )

    # Define callback function for database updates
    def db_update_callback(ontology_extraction_id, score):
        db_manager.update_ontology_llm_judge_score(
            ontology_extraction_id=ontology_extraction_id,
            score=score
        )

    save_to_db = args.save_to_db.lower() == "true"
    # Run ontology judging
    start = time.time()
    relation_scores = run_ontology_judging(
        llm=llm,
        category_name=args.category_name,
        ontology_tree=ontology_tree,
        save_to_db=save_to_db,
        ontology_extraction_id=ontology_extraction_id,
        db_update_callback=db_update_callback if save_to_db else None
    )
    end = time.time()
    logger.info(f'Ontology judging took {end - start:.2f} seconds')

    logger.info(f"Successfully judged {len(relation_scores)} relations")
    logger.info("Top 10 relation scores:")
    for relation, score in sorted(relation_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {relation}: {score}")

    if not save_to_db:
        logger.warning("Results were not saved to database (--save-to-db flag was set to 'false').")


if __name__ == "__main__":
    main()
