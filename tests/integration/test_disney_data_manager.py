import os
from src.data.disney_data_preprocess import DisneylandDataManager


# Create a cache for hugging face datasets, to avoid reaching disk quota
username = input("Enter your username (e.g. sp3924): ").strip()
hf_home = f"/vol/bitbucket/{username}/hf-cache"
os.makedirs(hf_home, exist_ok=True)
os.environ["HF_HOME"] = hf_home


def test_disney_data_preprocess():
    """
    Function to test the DisneylandDataManager, DisneylandDataProcessor classes 
    by performing data loading, processing, saving, and serving operations.

    Pipeline:
    1. Prompts the user to enter their username to set the cache directory.
    2. Initializes a DataManager instance.
    3. Loads raw data using the DataManager.
    4. Processes the loaded data into structured datasets.
    5. Saves the processed datasets to the specified output directory.
    6. Retrieves and serves the processed datasets.
    7. Prints sample data and summary statistics.
    """
    # Set cache directory based on user input
    cache_dir = f"/vol/bitbucket/{username}/data-cache"

    # Initialize the data manager
    disneyland_manager = DisneylandDataManager(output_dir=cache_dir)

    # Load data
    disneyland_manager.load_data()

    # Process data
    metadata, processed_reviews = disneyland_manager.process_data()

    # Save data
    disneyland_manager.save_data(metadata, processed_reviews)

    # Serve the processed data
    datasets = disneyland_manager.serve_data()

    # Print samples
    print("Product Samples:", datasets["products_dataset"].head())
    print("Review Samples:", datasets["reviews_dataset"].head(5))
    print("Number of products:", len(datasets["products_dataset"]))
    print("Number of reviews:", len(datasets["reviews_dataset"]))


if __name__ == "__main__":
    test_disney_data_preprocess()
