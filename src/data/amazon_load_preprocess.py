from datasets import load_dataset
from datetime import datetime
import pandas as pd
from src.constants import CATEGORY_MAPPING
from src.data.base_load_preprocess import DataProcessor, DataManager
from logger import logger


class AmazonDataProcessor(DataProcessor):
    ''' Data Processor for Amazon dataset.'''

    def process_metadata(self, raw_metadata):
        ''' Processes metadata to extract each product's id and name along with
        category ids and names

        Args: 
            raw_metadata (Dataset): Hugging Face dataset containing the metadata

        Returns: 
            Dataset: A dataset with product 'id' (from 'parent_asin'), 
                    'name' (from 'title'), 'description' (from 'description'), 
                    'created_at' (time), and 'updated_at' as columns.
        '''
        logger.info("PROCESSING metadata...")

        # Map metadata to include category_id
        extracted_metadata = raw_metadata.map(
            lambda x: {
                "id": x.get("parent_asin", ""),
                "name": " ".join(x["title"]) if isinstance(x.get("title", ""), list) else x.get("title", ""),
                "description": " ".join(x["description"]) if isinstance(x.get("description", ""), list) else x.get("description", ""),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            remove_columns=raw_metadata.column_names
        )

        return extracted_metadata

    def process_reviews(self, raw_reviews):
        ''' Processes reviews to extract review texts and generate unique ids
        for each review.

        Args: 
            raw_reviews (Dataset): Hugging Face dataset containing the reviews

        Returns: 
            Dataset: A dataset with review 'id' (generated), 'product_id',
                    'content' (from 'text'), and 'rating' (from 'rating') as columns.
        '''
        logger.info("PROCESSING reviews and creating reviews to product...")

        # Generate id, extract reviews & product id then remove irrelevant columns
        extracted_reviews = raw_reviews.map(
            lambda x, idx: {
                "id": idx + 1,
                "product_id": x.get("parent_asin", ""),
                "content": self._clean_text(x.get("text", "")),
                "rating": (x["rating"] - 1) / 4
            },
            with_indices=True,
            remove_columns=raw_reviews.column_names
        )

        return extracted_reviews


class AmazonDataManager(DataManager):
    '''Responsible for loading, modifying, saving, and serving datasets.'''

    def __init__(self, category_name, num_reviews, output_dir, **kwargs):
        self.category_name = category_name
        category_metadata = CATEGORY_MAPPING.get(category_name)

        # Check if category name is valid
        if category_metadata is None:
            raise ValueError(f"Invalid category name: {category_name}. "
                             f"Valid categories are: {list(CATEGORY_MAPPING.keys())}")

        self.category = category_metadata['category']
        self.sub_category = category_metadata['sub_categories']

        self.num_reviews = num_reviews
        self.meta_config = f"raw_meta_{self.category}"
        self.review_config = f"raw_review_{self.category}"
        self.output_dir = output_dir
        self.meta_data = None
        self.review_data = None
        self.processor = AmazonDataProcessor()

    def load_data(self):
        '''Loads metadata and review data for the specified category.

        Args: 
            None

        Returns: 
            None
        '''
        logger.info(f"LOADING metadata for {self.meta_config}...")
        raw_meta_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                                     self.meta_config, split="full",
                                     trust_remote_code=True,
                                     cache_dir=self.output_dir)

        # Filter metadata for subcategory
        filtered_products = raw_meta_data.filter(
            lambda x: all(subcat in x["categories"] for subcat in self.sub_category))

        logger.info(f"LOADING reviews for {self.review_config}...")
        review_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                                   self.review_config, split="full",
                                   trust_remote_code=True,
                                   cache_dir=self.output_dir)

        # Filter reviews based on filtered products & sampling 100,000 reviews
        product_asins = set(filtered_products["parent_asin"])
        filtered_reviews = review_data.filter(lambda x: x["parent_asin"] in product_asins and x["text"].strip() != "")

        self.review_data = filtered_reviews.shuffle(seed=42).select(range(min(self.num_reviews, len(filtered_reviews))))
        review_asins = review_asins = set(self.review_data["parent_asin"])
        self.meta_data = filtered_products.filter(lambda x: x["parent_asin"] in review_asins)

    def process_data(self):
        '''Processes raw metadata and reviews into the required datasets.

        Args: 
            None

        Returns:
            Dataset (products_dataset): A dataset containing product information
            Dataset (reviews_dataset): A dataset containing review information
        '''
        # Process metadata and reviews
        products_dataset = self.processor.process_metadata(self.meta_data)
        reviews_dataset = self.processor.process_reviews(self.review_data)

        return products_dataset, reviews_dataset

    def save_data(self, products_dataset, reviews_dataset):
        '''Saves the datasets to disk (or csv).

        Args: 
            Dataset (products_dataset): A dataset containing product information
            Dataset (reviews_dataset): A dataset containing review information

        Returns: 
            None
        '''
        logger.info(f"SAVING datasets to {self.output_dir}...")

        # Reorder columns for products dataset
        products_df = products_dataset.to_pandas()
        products_df = products_df[["id", "name", "description",
                                   "created_at", "updated_at"]]
        products_df.to_csv(f"{self.output_dir}/products_{self.category_name}.csv", index=False)

        # Reorder columns for reviews dataset
        reviews_df = reviews_dataset.to_pandas()
        reviews_df = reviews_df[["id", "product_id", "content", "rating"]]
        reviews_df.to_csv(f"{self.output_dir}/reviews_{self.category_name}.csv", index=False)

    def serve_data(self):
        '''Serves the datasets to be used.

        Args: 
            None

        Returns: 
            Dataset (double): The two dictionaries containing products and reviews.
        '''
        # Load csv data saved
        products_dataset = pd.read_csv(f"{self.output_dir}/products_{self.category_name}.csv")
        reviews_dataset = pd.read_csv(f"{self.output_dir}/reviews_{self.category_name}.csv")

        return {"products_dataset": products_dataset,
                "reviews_dataset": reviews_dataset}
