import pytest
import os
import tempfile
import shutil
import pickle
from unittest.mock import patch, MagicMock, mock_open
from anytree import Node

from src.argumentation.arg_framework.argument import Argument
from src.argumentation.arg_framework.product import Product


@pytest.fixture(autouse=True)
def mock_wordnet_lemmatizer():
    mock_lemmatizer = MagicMock()
    mock_lemmatizer.lemmatize.side_effect = lambda word: word if word.endswith('s') else word
    with patch('src.argumentation.arg_framework.product.wnl', mock_lemmatizer):
        yield


@pytest.fixture
def temp_product_dir():
    original_file_dir = Product.FILE_DIR

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    Product.FILE_DIR = temp_dir

    # Create folder structure
    folder_name = "test_folder"
    new_temp_dir = temp_dir + "test_folder"

    # Create the directory
    os.rename(temp_dir, new_temp_dir)

    # Create a sample product
    root = Node("smartphone", synset=["smartphone", "mobile phone", "cell phone"])
    camera = Node("camera", parent=root, synset=["camera", "photo", "picture"])
    display = Node("display", parent=root, synset=["display", "screen"])

    # Create sample products for different categories and methods
    products = {
        "smartphone": Product(root),
        "smartphone_alt": Product(root)
    }

    # Save the products to the temp directory
    for cat, product in products.items():
        # Regular product
        cat = cat
        file_path = os.path.join(new_temp_dir, cat + Product.FILE_EXTENSION)
        with open(file_path, 'wb') as f:
            pickle.dump(product, f)

    yield new_temp_dir, folder_name

    # Clean up
    shutil.rmtree(new_temp_dir)
    Product.FILE_DIR = original_file_dir


class TestProduct:
    def setup_method(self):
        """Set up test case with a sample product hierarchy."""
        # Create a simple tree structure for testing
        self.root = Node("smartphone", synset=["smartphone", "mobile phone", "cell phone"])
        self.camera = Node("camera", parent=self.root, synset=["camera", "photo", "picture"])
        self.display = Node("display", parent=self.root, synset=["display", "screen"])
        self.battery = Node("battery", parent=self.root, synset=["battery", "power"])

        # Create child nodes for features
        self.resolution = Node("resolution", parent=self.camera, synset=["resolution", "megapixels"])
        self.size = Node("size", parent=self.display, synset=["size", "dimensions"])

        # Create a dictionary for synonym testing
        self.syn_dict = {
            "smartphone": ["smartphone", "mobile phone", "cell phone"],
            "camera": ["camera", "photo", "picture"],
            "display": ["display", "screen"],
            "battery": ["battery", "power"],
            "resolution": ["resolution", "megapixels"],
            "size": ["size", "dimensions"]
        }

        # Create Product instances for testing
        self.product = Product(self.root)
        self.product_with_syn_dict = Product(self.root, self.syn_dict)

    def test_init(self):
        assert self.product.root == self.root

        assert set(self.product.feature_nodes) == {self.camera, self.display, self.battery, self.resolution, self.size}

        assert set(self.product.argument_nodes) == {self.root, self.camera,
                                                    self.display, self.battery, self.resolution, self.size}

        for node in self.product.argument_nodes:
            assert node in self.product.glossary
            assert all(isinstance(syn, list) for syn in self.product.glossary[node])

        for node in self.product.argument_nodes:
            assert node in self.product.singularities

    def test_init_with_syn_dict(self):
        for node in self.product_with_syn_dict.argument_nodes:
            if node.name in self.syn_dict:
                expected_synsets = [syn.split(' ') for syn in self.syn_dict[node.name]]
                assert self.product_with_syn_dict.glossary[node] == expected_synsets

    def test_argument_node_for_id(self):
        # Test retrieving argument nodes by their indices
        for i, node in enumerate(self.product.argument_nodes):
            assert self.product.argument_node_for_id(i) == node

    def test_name(self):
        assert self.product.name() == "smartphone"

    @patch('builtins.print')
    def test_print_latex(self, mock_print):
        self.product.print_latex()
        # Verify that print was called
        assert mock_print.called

    def test_aspect_to_latex(self):
        words = ["test", "words"]
        expected = "\\mbox{test} \\mbox{words}"
        assert Product.aspect_to_latex(words) == expected

    def test_get_product(self, temp_product_dir):
        temp_dir, folder_name = temp_product_dir

        # Test getting regular product
        product = Product.get_product(folder_name, "smartphone")
        assert isinstance(product, Product)
        assert product.name() == "smartphone"

    def test_get_product_not_found(self, temp_product_dir):
        temp_dir, folder_name = temp_product_dir

        # Test getting non-existent product
        with pytest.raises(Exception) as excinfo:
            Product.get_product(folder_name, "nonexistent")

        assert "No representation found for product" in str(excinfo.value)

        # Test getting non-existent product with method
        with pytest.raises(Exception) as excinfo:
            Product.get_product(folder_name, "smartphone", method="nonexistent")

        assert "No representation found for product" in str(excinfo.value)
