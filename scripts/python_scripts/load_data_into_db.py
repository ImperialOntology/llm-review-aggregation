"""
This script loads data from a given data source into a database.
"""

import os
import time
import argparse
from pathlib import Path
from src.data.amazon_load_preprocess import AmazonDataManager
from src.data.disney_data_preprocess import DisneylandDataManager
from db.manager import DatabaseManager
from src.constants import CATEGORY_MAPPING
from logger import logger


DBNAME = os.getenv("DBNAME")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

DATA_SOURCE_MAPPING = {
    'amazon': AmazonDataManager,
    'disney': DisneylandDataManager
}


def parse_args():
    # argument for cache directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, required=True)
    parser.add_argument('--data-source', type=str, required=True, choices=list(DATA_SOURCE_MAPPING.keys()),
                        help='Data source to load. Currently only amazon and disney are supported')
    parser.add_argument('--data-source-name', type=str,
                        default='amazon',
                        help='Name of the data source')
    parser.add_argument('--data-source-url', type=str,
                        default='https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023',
                        help='URL of the data source')
    parser.add_argument('--category-name', type=str, default='Games', choices=list(CATEGORY_MAPPING.keys()),
                        help='Category name which is mapped to the corresponding '
                             'super-category and category in the datasets used (except for the Disneyland). \n'
                             'Here is the said mapping for your reference:\n'
                             f'{CATEGORY_MAPPING}')
    parser.add_argument('--reviews', type=int, default=100000,
                        help='Number of reviews. NOTE: this is not used when test-mode=true')
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.cache_dir) / 'data'
    data_dir.mkdir(exist_ok=True)

    # insert into database
    # insert data source if doesn't exist
    db_manager = DatabaseManager(DBNAME, USER, PASSWORD, HOST, PORT)

    logger.info('Checking if data source exists')
    data_source_id = db_manager.check_if_data_source_exists(args.data_source_name)

    if data_source_id is None:
        logger.info('Data source does not exist')
        logger.info('Inserting data source')
        data_source_id = db_manager.insert_data_source(args.data_source_name, args.data_source_url)
    else:
        logger.info('Data source already exists')

    logger.info('Checking if category exists')
    category_id = db_manager.check_if_category_exists(args.category_name, data_source_id)
    if category_id:
        logger.info('Category already exists. Stopping insertion')
        return

    # Otherwise, insert data for the category
    logger.info('Category does not exist. Inserting category')

    data_manager_cls = DATA_SOURCE_MAPPING.get(args.data_source)
    assert data_manager_cls is not None, (f'category {args.category_name} is invalid for data source '
                                          f'{args.data_source}, please try other categories.')

    data_manager = data_manager_cls(category_name=args.category_name,
                                    num_reviews=args.reviews,
                                    output_dir=data_dir)
    start = time.time()
    data_manager.load_data()
    products_dataset, reviews_dataset = data_manager.process_data()
    data_manager.save_data(products_dataset, reviews_dataset)
    end = time.time()
    logger.info(f'Data loaded and processed in {end - start:.2f} seconds')

    # Serve data
    datasets = data_manager.serve_data()

    logger.info('Inserting data into database')
    reviews_df, products_df = datasets["reviews_dataset"], datasets["products_dataset"]

    logger.info('Inserting category')
    category_id = db_manager.insert_category(args.category_name, data_source_id)
    logger.info('Inserting products')
    product_id_mapping = db_manager.bulk_insert_products(products_df, category_id)
    logger.info('Inserting reviews')
    db_manager.bulk_insert_reviews(reviews_df, product_id_mapping)
    logger.info('Data inserted successfully')


if __name__ == '__main__':
    main()
