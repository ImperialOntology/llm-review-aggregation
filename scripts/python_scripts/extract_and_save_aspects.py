"""
This script queries the database for reviews, performs aspect extraction using the specified method (BERT or LLM),
and then saves the extracted aspects back to the database.
"""

import os
import time
from argparse import ArgumentParser
from pathlib import Path

from db.manager import DatabaseManager

from src.ontology.ontology_llm.aspects.run import (
    run_aspect_extraction as run_aspect_extraction_llm)
from src.ontology.ontology_bert.aspects.run import (
    run_aspect_extraction as run_aspect_extraction_bert)
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
    parser.add_argument("--bert-or-llm", type=str, required=True,
                        choices=["bert", "llm"],
                        help="Whether to use BERT or LLM for aspect extraction.")
    parser.add_argument("--bert-model-path", type=str, required=False,
                        default="path/to/bert/model",
                        help="Path to the BERT model to use for aspect extraction.")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=False, default=8)
    parser.add_argument("--njobs", type=int, required=False, default=4,
                        help="Number of parallel jobs to run for tokenisation in BERT aspect extraction.")
    return parser.parse_args()


def main():
    # query data from database
    args = parse_args()
    db_manager = DatabaseManager(DBNAME, USER, PASSWORD, HOST, PORT)
    category_id, reviews = db_manager.query_all_reviews(args.data_source_name, args.category_name)
    reviews_list = [review[2] for review in reviews]

    logger.info("Extracting aspects from reviews...")
    start = time.time()
    if args.bert_or_llm == "llm":
        output_file = f"{args.cache_dir}/aspects.csv"
        aspect_counter = run_aspect_extraction_llm(reviews_list,
                                                   output_file=output_file,
                                                   cache_dir=args.cache_dir,
                                                   batch_size=args.batch_size)
    elif args.bert_or_llm == "bert":
        assert Path(args.bert_model_path).exists(), f"Path {args.bert_model_path} does not exist."
        aspect_counter = run_aspect_extraction_bert(
            reviews_list=reviews_list,
            batch_size=args.batch_size,
            bert_entity_extractor_path=args.bert_model_path,
            njobs=args.njobs
        )

    else:
        raise ValueError("bert-or-llm argument should be either 'bert' or 'llm'.")
    end = time.time()
    logger.info(f"Aspect extraction took {end - start:.2f} seconds.")
    logger.info(f"Extracted {len(aspect_counter)} aspects.")

    logger.info("Saving aspects to database...")
    db_manager.insert_aspect_extraction(
        category_id=category_id,
        aspect_counter=aspect_counter,
        name=args.name,
        method=args.method,
        description=args.description,
    )


if __name__ == "__main__":
    main()
