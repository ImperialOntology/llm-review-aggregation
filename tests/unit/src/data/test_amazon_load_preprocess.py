import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the classes to test
from src.data.amazon_load_preprocess import AmazonDataManager, AmazonDataProcessor

class TestDataProcessor:
    @pytest.fixture
    def setup(self):
        self.html_text = "<p>This is some <b>test</b> text</p>"
        self.special_char_text = "Test text with special characters: @#$%^&*"
        self.non_ascii_text = "Test with non-ASCII characters: café, résumé, naïve"
        
        self.sample_metadata = {
            "parent_asin": "TEST123",
            "title": "Test Product",
            "description": "This is a test product description",
            "categories": ["Test Category"]
        }
        
        self.sample_review = {
            "parent_asin": "TEST123",
            "text": "This is a <b>great</b> product! I loved it.",
            "rating": 5
        }
        
        # Mock datasets
        self.mock_metadata_dataset = MagicMock()
        self.mock_metadata_dataset.map.return_value = MagicMock()
        self.mock_metadata_dataset.column_names = ["parent_asin", "title", "description", "categories"]
        
        self.mock_reviews_dataset = MagicMock()
        self.mock_reviews_dataset.map.return_value = MagicMock()
        self.mock_reviews_dataset.column_names = ["parent_asin", "text", "rating"]

        self.amazon_data_processor = AmazonDataProcessor()
    
    def test_strip_html(self, setup):
        result = AmazonDataProcessor._strip_html(self.html_text)
        assert result == "This is some test text"
    
    def test_remove_special_characters(self, setup):
        result = AmazonDataProcessor._remove_special_characters(self.special_char_text)
        assert result == "Test text with special characters: "
    
    def test_remove_non_ascii(self, setup):
        result = AmazonDataProcessor._remove_non_ascii(self.non_ascii_text)
        assert "café" not in result
        assert "resume" in result.lower() 
        assert "naive" in result.lower()
    
    def test_clean_text(self, setup):
        test_text = "<p>This is a TEST with @special characters </p>"
        result = AmazonDataProcessor._clean_text(test_text)
        assert result == "this is a test with special characters "
    
    @patch('src.data.amazon_load_preprocess.datetime')
    def test_process_metadata(self, mock_datetime, setup):
        mock_now = datetime(2025, 4, 7, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Create a mock dataset with a single sample
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["parent_asin", "title", "description"]
        mock_dataset.map.return_value = "processed_metadata"
        
        # Call the method
        result = self.amazon_data_processor.process_metadata(mock_dataset)
        
        # Verify the result
        assert result == "processed_metadata"
        
        # Verify map was called with the right function
        mock_dataset.map.assert_called_once()
        
        # Get the lambda function that was passed to map
        map_func = mock_dataset.map.call_args[0][0]
        
        # Test the lambda function with our sample data
        result = map_func(self.sample_metadata)
        
        assert result["id"] == self.sample_metadata["parent_asin"]
        assert result["name"] == self.sample_metadata["title"]
        assert result["description"] == self.sample_metadata["description"]
        assert result["created_at"] == mock_now.isoformat()
        assert result["updated_at"] == mock_now.isoformat()
    
    def test_process_reviews(self, setup):
        # Create a mock dataset with a single sample
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["parent_asin", "text", "rating"]
        mock_dataset.map.return_value = "processed_reviews"
        
        # Call the method
        result = self.amazon_data_processor.process_reviews(mock_dataset)
        
        # Verify the result
        assert result == "processed_reviews"
        
        # Verify map was called with the right function and parameters
        mock_dataset.map.assert_called_once()
        
        # Get the lambda function that was passed to map
        map_func = mock_dataset.map.call_args[0][0]
        
        # Test the lambda function with our sample data
        result = map_func(self.sample_review, 0)
        
        assert result["id"] == 1  # idx + 1
        assert result["product_id"] == self.sample_review["parent_asin"]
        assert result["content"] == "this is a great product! i loved it."  # HTML and special chars removed
        assert result["rating"] == 1.0  # 5/5
        
        # Check that with_indices and remove_columns are set correctly
        assert mock_dataset.map.call_args[1]["with_indices"] == True
        assert mock_dataset.map.call_args[1]["remove_columns"] == mock_dataset.column_names


class TestDataManager:
    @pytest.fixture
    def setup(self):
        self.category = "Stand Mixers"
        self.sub_category = "test_subcategory"
        self.num_reviews = 100
        self.output_dir = "test_output_dir"
        
        # Create the DataManager instance
        self.data_manager = AmazonDataManager(
            self.category, 
            self.num_reviews, 
            self.output_dir
        )
        
        # Prepare mock data
        self.mock_raw_meta_data = MagicMock()
        self.mock_filtered_products = MagicMock()
        self.mock_filtered_products.__getitem__.return_value = ["PROD1", "PROD2"]
        
        self.mock_review_data = MagicMock()
        self.mock_filtered_reviews = MagicMock()
        
        self.mock_products_dataset = MagicMock()
        self.mock_reviews_dataset = MagicMock()
        
        # Sample dataframes
        self.products_df = pd.DataFrame({
            'id': ['PROD1', 'PROD2'],
            'name': ['Product 1', 'Product 2'],
            'description': ['Desc 1', 'Desc 2'],
            'metadata': [{'key': 'value1'}, {'key': 'value2'}],
            'created_at': ['2023-01-01T12:00:00', '2023-01-01T12:00:00'],
            'updated_at': ['2023-01-01T12:00:00', '2023-01-01T12:00:00']
        })
        
        self.reviews_df = pd.DataFrame({
            'id': [1, 2],
            'product_id': ['PROD1', 'PROD2'],
            'content': ['Great product!', 'Not so great.'],
            'rating': [1.0, 0.6]
        })
    
    @patch('src.data.amazon_load_preprocess.load_dataset')
    def test_load_data(self, mock_load_dataset, setup):
        mock_load_dataset.side_effect = [
            self.mock_raw_meta_data,
            self.mock_review_data
        ]
        
        self.mock_raw_meta_data.filter.return_value = self.mock_filtered_products
        
        # Setup for filtering reviews
        def mock_filter_func(filter_func):
            return self.mock_filtered_reviews
        
        self.mock_review_data.filter.side_effect = mock_filter_func
        self.mock_filtered_reviews.shuffle.return_value = self.mock_filtered_reviews
        self.mock_filtered_reviews.select.return_value = self.mock_filtered_reviews
        
        # Define sample product ASINs
        self.mock_filtered_products.__getitem__.return_value = ["PROD1", "PROD2"]
        
        # Setup review ASINs for filtering products
        self.mock_filtered_reviews.__getitem__.return_value = ["PROD1"]
        
        # Call the method
        self.data_manager.load_data()
        
        # Verify load_dataset was called correctly
        assert mock_load_dataset.call_count == 2
        meta_call, review_call = mock_load_dataset.call_args_list
        
        assert meta_call[0][0] == "McAuley-Lab/Amazon-Reviews-2023"
        assert meta_call[0][1] == f"raw_meta_{self.data_manager.category}"
        assert meta_call[1]["split"] == "full"
        assert meta_call[1]["trust_remote_code"] == True
        assert meta_call[1]["cache_dir"] == self.output_dir
        
        # Verify filtering and selection
        self.mock_raw_meta_data.filter.assert_called_once()
        self.mock_review_data.filter.assert_called_once()
        self.mock_filtered_reviews.shuffle.assert_called_once_with(seed=42)
        self.mock_filtered_reviews.select.assert_called_once()
        
        # Verify data is set correctly
        assert self.data_manager.review_data == self.mock_filtered_reviews
        assert self.data_manager.meta_data is not None
    
    def test_process_data(self, setup):
        self.data_manager.meta_data = MagicMock()
        self.data_manager.review_data = MagicMock()
        
        # Mock processor methods
        with patch.object(AmazonDataProcessor, 'process_metadata', return_value=self.mock_products_dataset) as mock_process_meta:
            with patch.object(AmazonDataProcessor, 'process_reviews', return_value=self.mock_reviews_dataset) as mock_process_reviews:
                # Call the method
                products, reviews = self.data_manager.process_data()
                
                # Verify processor methods were called
                mock_process_meta.assert_called_once_with(self.data_manager.meta_data)
                mock_process_reviews.assert_called_once_with(self.data_manager.review_data)
                
                # Verify return values
                assert products == self.mock_products_dataset
                assert reviews == self.mock_reviews_dataset
    
    def test_save_data(self, setup):
        # Convert mocks to pandas DataFrames
        self.mock_products_dataset.to_pandas.return_value = self.products_df
        self.mock_reviews_dataset.to_pandas.return_value = self.reviews_df
        
        # Mock DataFrame's to_csv method
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            # Call the method
            self.data_manager.save_data(self.mock_products_dataset, self.mock_reviews_dataset)
            
            # Verify to_pandas was called
            self.mock_products_dataset.to_pandas.assert_called_once()
            self.mock_reviews_dataset.to_pandas.assert_called_once()
            
            # Verify to_csv was called with correct parameters
            assert mock_to_csv.call_count == 2
            products_call, reviews_call = mock_to_csv.call_args_list
            
            expected_products_path = f"{self.output_dir}/products_{self.category}.csv"
            expected_reviews_path = f"{self.output_dir}/reviews_{self.category}.csv"
            
            assert products_call[0][0] == expected_products_path
            assert products_call[1]["index"] == False
            
            assert reviews_call[0][0] == expected_reviews_path
            assert reviews_call[1]["index"] == False
    
    @patch('pandas.read_csv')
    def test_serve_data(self, mock_read_csv, setup):
        # Configure mock read_csv to return our test dataframes
        mock_read_csv.side_effect = [self.products_df, self.reviews_df]
        
        # Call the method
        result = self.data_manager.serve_data()
        
        # Verify read_csv was called with correct paths
        assert mock_read_csv.call_count == 2
        products_call, reviews_call = mock_read_csv.call_args_list
        
        expected_products_path = f"{self.output_dir}/products_{self.category}.csv"
        expected_reviews_path = f"{self.output_dir}/reviews_{self.category}.csv"
        
        assert products_call[0][0] == expected_products_path
        assert reviews_call[0][0] == expected_reviews_path
        
        # Verify the result structure
        assert "products_dataset" in result
        assert "reviews_dataset" in result
        assert isinstance(result["products_dataset"], pd.DataFrame)
        assert isinstance(result["reviews_dataset"], pd.DataFrame)