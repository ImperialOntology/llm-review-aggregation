import os
from src.data.amazon_load_preprocess import AmazonDataManager

# Create a cache for hugging face datasets, to avoid reaching disk quota
username = input("Enter your username (e.g. sp3924): ").strip()
hf_home = f"/vol/bitbucket/{username}/hf-cache"
os.makedirs(hf_home, exist_ok=True)
os.environ["HF_HOME"] = hf_home


def test_data_preprocess():
    """
    Function to test the DataManager and DataProcessor classes by performing 
    data loading, processing, saving, and serving operations.

    Pipeline:
    1. Prompts the user to enter their username to set the cache directory.
    2. Initializes a DataManager instance with a predefined category, subcategory, 
       number of reviews, and output directory.
    3. Loads raw data using the DataManager.
    4. Processes the loaded data into structured datasets.
    5. Saves the processed datasets to the specified output directory.
    6. Retrieves and serves the processed datasets.
    7. Prints sample data and summary statistics.
    """
    # Set cache directory based on user input
    cache_dir = f"/vol/bitbucket/{username}/data-cache"

    num_reviews = 10000
    category_name = "Televisions"
    data_manager = AmazonDataManager(category_name, num_reviews, output_dir=cache_dir)

    # Load data
    data_manager.load_data()

    # Process data
    products_dataset, reviews_dataset = data_manager.process_data()

    # Save data
    data_manager.save_data(products_dataset, reviews_dataset)

    # Serve data
    datasets = data_manager.serve_data()

    # Print samples
    print("Product Samples:", datasets["products_dataset"].iloc[:5, :3])
    print("Review Samples:", datasets["reviews_dataset"].head(5))
    print("Number of products:", len(datasets["products_dataset"]))
    print("Number of reviews:", len(datasets["reviews_dataset"]))


if __name__ == "__main__":
    test_data_preprocess()
