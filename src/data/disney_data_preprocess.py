import pandas as pd
import kagglehub
from datetime import datetime
from kagglehub import KaggleDatasetAdapter
from src.data.base_load_preprocess import DataProcessor, DataManager
from logger import logger


class DisneylandDataProcessor(DataProcessor):
    '''Processes Disneyland Reviews dataset'''

    def process_metadata(self, raw_reviews):
        '''Processes metadata to extract unique branch IDs and names.

        Args: 
            raw_reviews (DataFrame): Pandas DataFrame containing the reviews

        Returns: 
            DataFrame: A DataFrame with 'id' (generated), 'name' (Branch),
                       'description' (empty), 'metadata' (empty), 
                       'created_at', and 'updated_at' columns.
        '''
        logger.info("PROCESSING Disneyland metadata...")

        # Extract unique branches
        branches = raw_reviews['Branch'].unique()

        # Generate metadata DataFrame
        created_at = datetime.now().isoformat()
        metadata = pd.DataFrame({
            'id': range(1, len(branches) + 1),
            'name': branches,
            'description': '',
            'metadata': '',
            'created_at': created_at,
            'updated_at': created_at
        })

        return metadata

    def process_reviews(self, raw_reviews, metadata):
        '''Processes reviews efficiently by mapping branch to product_id.

        Args: 
            raw_reviews (DataFrame): Pandas DataFrame containing the reviews
            metadata (DataFrame): Processed Products DataFrame with branch ids

        Returns: 
            DataFrame: A DataFrame with 'id' (Review_ID), 'product_id' (branch id),
                       'content', and 'rating' columns.
        '''
        logger.info("PROCESSING Disneyland reviews...")

        # Map 'Branch' to 'product_id'
        branch_to_id = dict(zip(metadata['name'], metadata['id']))
        raw_reviews['product_id'] = raw_reviews['Branch'].map(branch_to_id)

        # Extract Reviews
        processed_reviews = pd.DataFrame({
            'id': range(1, len(raw_reviews) + 1),
            'product_id': raw_reviews['product_id'],
            'content': raw_reviews['Review_Text'].apply(self._clean_text),
            'rating': (raw_reviews['Rating'] - 1) / 4
        })

        return processed_reviews


class DisneylandDataManager(DataManager):
    '''Manages loading, processing, and saving of Disneyland Reviews dataset'''

    def __init__(self, output_dir, **kwargs):
        self.output_dir = output_dir
        self.review_data = None
        self.processor = DisneylandDataProcessor()

    def load_data(self):
        '''Loads Disneyland reviews data from a CSV file.'''
        logger.info(f"LOADING Disneyland reviews from {self.output_dir}...")

        # Load dataset using KaggleHub with Pandas
        self.review_data = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "arushchillar/disneyland-reviews",
            path="DisneylandReviews.csv",
            pandas_kwargs={"encoding": "latin1"}
        )

    def process_data(self):
        '''Processes raw reviews into products and reviews datasets.'''
        products_dataset = self.processor.process_metadata(self.review_data)
        reviews_dataset = self.processor.process_reviews(self.review_data, products_dataset)

        return products_dataset, reviews_dataset

    def save_data(self, products_dataset, reviews_dataset):
        '''Saves the processed products and reviews datasets to csv.'''
        logger.info(f"SAVING processed data to {self.output_dir}...")

        products_dataset.to_csv(f"{self.output_dir}/Disneyland_Products.csv", index=False)
        reviews_dataset.to_csv(f"{self.output_dir}/Disneyland_Reviews.csv", index=False)

    def serve_data(self):
        '''Serves the processed products and reviews datasets.'''
        products_dataset = pd.read_csv(f"{self.output_dir}/Disneyland_Products.csv")
        reviews_dataset = pd.read_csv(f"{self.output_dir}/Disneyland_Reviews.csv")

        return {"products_dataset": products_dataset,
                "reviews_dataset": reviews_dataset}
