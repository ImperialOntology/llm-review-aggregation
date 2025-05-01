import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from anytree import Node
import pickle
import sys


# Mock the imports
@pytest.fixture(autouse=True)
def mock_imports():
    old_sys_modules = sys.modules.copy()
    mods = {
        'src.argumentation.sentiment.bert_analyzer': Mock(),
        'src.argumentation.arg_framework.product': Mock(),
        'src.argumentation.arg_framework.review': Mock(),
        'src.argumentation.arg_framework.argument': Mock(),
    }
    with patch.dict('sys.modules', mods):
        yield

    # cleanup
    sys.modules = old_sys_modules


# Import the Framework class after mocking dependencies
@pytest.fixture
def framework_class():
    with patch('anytree.PostOrderIter') as mock_post_order_iter:
        from src.argumentation.arg_framework.framework import Framework
        return Framework


class TestFramework:
    def setup_method(self):
        # Mock BertAnalyzer
        self.mock_bert_analyzer = Mock()
        self.mock_bert_analyzer.get_batch_sentiment_polarity.return_value = [0.8, -0.5, 0.95]
        
        # Create mock product structure
        self.product_node = Mock()
        self.product_node.name = "Product"
        
        # Create feature nodes
        self.feature1 = Mock()
        self.feature1.name = "Feature1"
        self.feature1.children = []
        
        self.feature2 = Mock()
        self.feature2.name = "Feature2"
        self.feature2.children = []
        
        # Set up product node children
        self.product_node.children = [self.feature1, self.feature2]
        
        # Mock Product
        self.mock_product = Mock()
        self.mock_product.root = self.product_node
        self.mock_product.argument_nodes = [self.product_node, self.feature1, self.feature2]
        self.mock_product.feature_nodes = [self.feature1, self.feature2]
        
        # Mock reviews DataFrame
        review_data = {
            'id': [1, 2],
            'content': ["This is a great product!", "The feature1 is terrible."]
        }
        self.review_df = pd.DataFrame(review_data)
        
        # Mock Review class and instances
        self.mock_reviews = []
        for i in range(2):
            mock_review = Mock()
            mock_review.id = i + 1
            mock_review.phrases = []
            
            # Add phrases with arguments
            phrase1 = Mock()
            phrase1.text = f"Review {i+1} Phrase 1"
            phrase1.args = [Mock()]
            phrase1.args[0].start = 0
            phrase1.args[0].end = 10
            phrase1.n_args.return_value = 1
            phrase1.get_vote.return_value = 0.9 if i == 0 else -0.7
            phrase1.get_votes.return_value = {self.product_node: 0.9 if i == 0 else -0.7}
            
            # Set review votes
            mock_review.phrases = [phrase1]
            mock_review.get_votes.return_value = {
                self.product_node: 0.8 if i == 0 else -0.5,
                self.feature1: 0.9 if i == 0 else -0.6,
                self.feature2: 0.7 if i == 0 else -0.4
            }
            
            self.mock_reviews.append(mock_review)
        
        # Patch the Review class constructor to return our mock reviews
        self.review_patcher = patch('src.argumentation.arg_framework.review.Review')
        self.mock_review_class = self.review_patcher.start()
        self.mock_review_class.side_effect = self.mock_reviews
        
        # Setup PostOrderIter for gradual semantics
        self.post_order_patcher = patch('anytree.PostOrderIter')
        self.mock_post_order = self.post_order_patcher.start()
        self.mock_post_order.return_value = [self.feature1, self.feature2, self.product_node]
        
        # Initialize Framework
        with patch('src.argumentation.sentiment.bert_analyzer.BertAnalyzer', return_value=self.mock_bert_analyzer):
            from src.argumentation.arg_framework.framework import Framework
            self.framework = Framework(
                product=self.mock_product,
                product_id="test_product",
                review_df=self.review_df,
                bert_analyser=self.mock_bert_analyzer
            )
    
    def teardown_method(self):
        """Clean up patchers."""
        self.review_patcher.stop()
        self.post_order_patcher.stop()
    
    def test_initialization(self):
        assert self.framework.product_id == "test_product"
        assert self.framework.product == self.mock_product
        assert self.framework.product_node == self.product_node
        assert self.framework.arguments == [self.product_node, self.feature1, self.feature2]
        assert self.framework.features == [self.feature1, self.feature2]
        
        # Check if methods were called
        self.mock_bert_analyzer.get_batch_sentiment_polarity.assert_called()
    
    def test_extract_votes(self):
        # Reset mock
        self.mock_bert_analyzer.get_batch_sentiment_polarity.reset_mock()
        self.mock_bert_analyzer.get_batch_sentiment_polarity.return_value = [0.7, 0.8]
        
        # Call extract_votes with our mock reviews
        self.framework.extract_votes(self.mock_reviews)
        
        # Check if sentiments were set
        for review in self.mock_reviews:
            for phrase in review.phrases:
                for arg in phrase.args:
                    arg.set_sentiment.assert_called()
    
    def test_get_aggregates(self):
        ra, vote_sum, vote_phrases = self.framework.get_aggregates(self.mock_reviews)
        
        # Check if ra contains expected structure
        assert len(ra) == 6  # 3 arguments x 2 reviews
        
        # Check vote_sum
        assert self.product_node in vote_sum
        assert self.feature1 in vote_sum
        assert self.feature2 in vote_sum
        
        # Check vote_phrases
        assert self.product_node in vote_phrases
    
    def test_get_qbaf(self, monkeypatch):
        # Mock data for get_qbaf
        ra = [
            {'review_id': 1, 'argument': self.product_node, 'vote': 0.8},
            {'review_id': 2, 'argument': self.product_node, 'vote': -0.5},
            {'review_id': 1, 'argument': self.feature1, 'vote': 0.9},
            {'review_id': 2, 'argument': self.feature1, 'vote': -0.6},
            {'review_id': 1, 'argument': self.feature2, 'vote': 0.7},
            {'review_id': 2, 'argument': self.feature2, 'vote': -0.4}
        ]
        
        # Execute get_qbaf
        qbaf, arg_polarities = self.framework.get_qbaf(ra, 2)
        
        # Check qbaf structure
        assert 'supporters' in qbaf
        assert 'attackers' in qbaf
        assert 'base_strengths' in qbaf
        
        # Check argument polarities
        assert arg_polarities[self.product_node]  # Should be positive
        assert arg_polarities[self.feature1]  # Should be positive
        assert arg_polarities[self.feature2]  # Should be positive
    
    def test_get_strengths(self):
        # Setup mock qbaf
        qbaf = {
            'supporters': {
                self.product_node: [self.feature1],
                self.feature1: [],
                self.feature2: []
            },
            'attackers': {
                self.product_node: [self.feature2],
                self.feature1: [],
                self.feature2: []
            },
            'base_strengths': {
                self.product_node: 0.65,
                self.feature1: 0.75,
                self.feature2: 0.15
            }
        }
        
        # Get strengths
        strengths = self.framework.get_strengths(qbaf)
        
        # Check strengths for all arguments
        assert self.product_node in strengths
        assert self.feature1 in strengths
        assert self.feature2 in strengths
        
        # Verify strength calculations
        assert strengths[self.feature1] == 0.75  # No supporters/attackers
        assert strengths[self.feature2] == 0.15  # No supporters/attackers
    
    
    def test_supporting_and_attacking_phrases(self):
        # Setup mock vote_phrases
        phrase1 = Mock()
        phrase1.get_vote.return_value = 0.8
        
        phrase2 = Mock()
        phrase2.get_vote.return_value = -0.5
        
        self.framework.vote_phrases = {
            self.product_node: [phrase1, phrase2]
        }
        
        # Test supporting_phrases
        supporting = self.framework.supporting_phrases(self.product_node)
        assert len(supporting) == 1
        assert supporting[0] == phrase1
        
        # Test attacking_phrases
        attacking = self.framework.attacking_phrases(self.product_node)
        assert len(attacking) == 1
        assert attacking[0] == phrase2
    
    def test_get_product_strength_percentage(self):
        # Set mock strengths
        self.framework.strengths = {
            self.product_node: 0.75
        }
        
        # Calculate percentage
        percentage = self.framework.get_product_strength_percentage()
        assert percentage == 75.0
    
    def test_is_well_formatted(self):
        from src.argumentation.arg_framework.framework import Framework
        
        # Test various phrases
        assert Framework.is_well_formatted("This is a good product.")
        assert Framework.is_well_formatted("Feature works well!")
        assert Framework.is_well_formatted("I like the price (very affordable).")
        assert not Framework.is_well_formatted("Contains weird characters like ☺ and ♥")
    
    def test_get_strongest_supporting_subfeature(self):
        # Setup mock qbaf and strengths
        self.framework.qbaf = {
            'supporters': {
                self.product_node: [self.feature1, self.feature2]
            }
        }
        
        self.framework.strengths = {
            self.feature1: 0.8,
            self.feature2: 0.6
        }
        
        # Get strongest supporting subfeature
        strongest = self.framework.get_strongest_supporting_subfeature(self.product_node)
        assert strongest == self.feature1
        
        # Test when no supporters
        self.framework.qbaf['supporters'][self.product_node] = []
        strongest = self.framework.get_strongest_supporting_subfeature(self.product_node)
        assert strongest is None