import pytest
from unittest.mock import patch, MagicMock
import sys
from anytree import Node

# Create mock modules and classes
mock_treebank = MagicMock()
mock_lemmatizer = MagicMock()
mock_sent_tokenize = MagicMock()

# Create nested structure for nltk modules
mock_tokenize = MagicMock()
mock_tokenize.TreebankWordTokenizer = MagicMock(return_value=mock_treebank)
mock_tokenize.sent_tokenize = mock_sent_tokenize

mock_stem = MagicMock()
mock_stem.WordNetLemmatizer = MagicMock(return_value=mock_lemmatizer)

mock_nltk = MagicMock()
mock_nltk.tokenize = mock_tokenize
mock_nltk.stem = mock_stem

# Create anytree mocks
mock_post_order_iter = MagicMock()
mock_anytree = MagicMock()
mock_anytree.PostOrderIter = mock_post_order_iter
mock_anytree.Node = Node  # Keep the real Node class

# Set up mock for MAX_SEQ_LEN constant
mock_bert_dataset = MagicMock()
mock_bert_dataset.MAX_SEQ_LEN = 100

@pytest.fixture(autouse=True)
def mock_dependencies():
    old_sys_modules = sys.modules.copy()
    # Setup the module mocks
    sys.modules['nltk'] = mock_nltk
    sys.modules['nltk.tokenize'] = mock_tokenize
    sys.modules['nltk.stem'] = mock_stem
    sys.modules['anytree'] = mock_anytree
    sys.modules['src.argumentation.sentiment.bert_dataset'] = mock_bert_dataset
    
    yield

    # Teardown the mocks
    sys.modules = old_sys_modules


# Import the actual classes to test after patching the system modules
with patch.dict('sys.modules', {
    'nltk': mock_nltk,
    'nltk.tokenize': mock_tokenize,
    'nltk.stem': mock_stem,
    'anytree': mock_anytree,
    'src.argumentation.sentiment.bert_dataset': mock_bert_dataset
}):
    from src.argumentation.arg_framework.review import Review, Phrase, Arg


class TestArg:
    def setup_method(self):
        node = MagicMock()
        form = "test form"
        start = 0
        end = 5
        self.arg = Arg(node, form, start, end)
    
    def test_set_sentiment(self):
        sentiment = -0.75
        self.arg.set_sentiment(sentiment)
        
        assert self.arg.sentiment == sentiment


class TestPhrase:
    def setup_method(self):
        # Create mock product
        self.product = MagicMock()
        self.root = Node("root")
        self.feature1 = Node("feature1", parent=self.root)
        
        self.product.root = self.root
        self.product.glossary = {
            self.root: [["root"]],
            self.feature1: [["feature", "one"]]
        }
        
        # Set up tokenizer behavior
        self.text = "This is feature one."
        self.spans = [(0, 4), (5, 7), (8, 15), (16, 19), (20, 21)]
        self.tokens = ["This", "is", "feature", "one", "."]
        
        mock_treebank.span_tokenize.return_value = self.spans
        
        # Create a phrase with mocked matching
        with patch('src.argumentation.arg_framework.review.Phrase.matching_subsequences') as self.mock_matching:
            self.mock_matching.return_value = [(2, 4)]  # Match "feature one"
            self.phrase = Phrase(self.text, self.product)
            
            # Mock args
            self.mock_arg = MagicMock()
            self.mock_arg.node = self.feature1
            self.mock_arg.sentiment = 0.98
            self.mock_arg.form = "feature one"
            self.mock_arg.start = 2
            self.mock_arg.end = 4
            
            # Replace actual args with mock
            self.phrase.args = [self.mock_arg]
    
    def test_get_votes(self):
        # Set threshold
        Review.SENTIMENT_THRESHOLD = 0.95
        
        votes = self.phrase.get_votes()
        
        # Check vote was recorded (sentiment is above threshold)
        assert votes[self.feature1] == 0.98
        
        # Test below threshold scenario
        self.mock_arg.sentiment = 0.90
        self.phrase.votes = {}
        votes = self.phrase.get_votes()
        
        # Should not add vote when below threshold
        assert not votes
    
    def test_get_arg_mentions(self):
        mentions = self.phrase.get_arg_mentions(self.feature1)
        
        # Should return form and position information
        assert mentions == [(self.mock_arg.form, 8, 19)]
        
        # Test with non-matching node
        other_node = Node("other")
        mentions = self.phrase.get_arg_mentions(other_node)
        
        # Should return empty list
        assert mentions == []
    
    def test_n_args(self):
        assert self.phrase.n_args() == 1
        
        # Test with empty args
        self.phrase.args = []
        assert self.phrase.n_args() == 0
    
    def test_remove_ancestors(self):
        # Create test nodes
        root = Node("root")
        parent = Node("parent", parent=root)
        child = Node("child", parent=parent)
        sibling = Node("sibling", parent=root)
        
        # Create test list
        test_list = [root, parent, sibling]
        
        # Remove ancestors of child
        self.phrase.remove_ancestors(child, test_list)
        
        # Parent and root should be removed
        assert root not in test_list
        assert parent not in test_list
        
        # Sibling should remain
        assert sibling in test_list
    
    def test_matching_subsequences(self):
        matches = Phrase.matching_subsequences(["feature", "one"], self.tokens)
        
        # Should find one match at positions 2-4
        assert matches == [(2, 4)]
        
        # Test no matches
        matches = Phrase.matching_subsequences(["nonexistent"], self.tokens)
        assert matches == []


class TestReview:
    def setup_method(self):
        # Configure mocks
        self.review_id = "r123"
        self.review_body = "This is a test review. I like feature one but dislike feature two."
        
        # Create mock product
        self.product = MagicMock()
        self.root = Node("root")
        self.feature1 = Node("feature1", parent=self.root)
        self.feature2 = Node("feature2", parent=self.root)
        
        self.product.root = self.root
        self.product.glossary = {
            self.root: [["root"]],
            self.feature1: [["feature", "one"]],
            self.feature2: [["feature", "two"]]
        }
        
        # Mock sent_tokenize
        mock_sent_tokenize.return_value = [
            "This is a test review.",
            "I like feature one but dislike feature two."
        ]
        

        self.mock_phrase1 = MagicMock()
        self.mock_phrase1.votes = {self.feature1: 0.98}
        
        self.mock_phrase2 = MagicMock()
        self.mock_phrase2.votes = {self.feature2: -0.97}

        with patch.object(Review, 'extract_phrases', return_value=[self.mock_phrase1, self.mock_phrase2]):
            self.review = Review(self.review_id, self.review_body, self.product)
                        
            # Create the review
            self.review = Review(self.review_id, self.review_body, self.product)
    
    def test_init(self):
        assert self.review.id == self.review_id
        assert self.review.body == self.review_body
        assert self.review.product == self.product
        assert len(self.review.phrases) == 2
    
    def test_extract_phrases(self):
        # Mock span_tokenize for phrases
        mock_treebank.span_tokenize.return_value = [(0, 4), (5, 7), (8, 15)]
        # Mock the matching_subsequences method
        with patch('src.argumentation.arg_framework.review.Phrase.matching_subsequences', return_value=[]):
            phrases = Review.extract_phrases(self.review_body, self.product)
            
            # With our mocked split, should create at least 3 phrases
            # (one for original sentence and two for split parts)
            assert len(phrases) >= 3
    
    def test_get_votes(self):
        votes = self.review.get_votes()
        
        # Check that votes are normalized
        assert votes[self.feature1] == 1  # Positive
        assert votes[self.feature2] == -1  # Negative
    
    def test_augment_votes(self):
        # Setup initial votes
        self.review.votes = {
            self.feature1: 1,   # Positive
            self.feature2: -1,  # Negative
        }
        
        # Mock PostOrderIter to return controlled node list
        mock_post_order_iter.return_value = [self.feature1, self.feature2, self.root]
        
        # Run augmentation
        self.review.augment_votes()
        
        # Root should get a vote based on children
        # In this case, feature1 and feature2 have opposite votes, so sum is 0
        # According to code, root shouldn't get a vote if sum is 0
        assert self.root not in self.review.votes
        
        # Now test case where sum is positive
        self.review.votes = {
            self.feature1: 1,
            self.feature2: 1,
        }
        self.review.augment_votes()
        
        # Root should get positive vote
        assert self.review.votes[self.root] == 1
    
    def test_is_voting(self):
        # Should be true when phrases have votes
        assert self.review.is_voting() is True
        
        # Should be false when no phrases have votes
        self.mock_phrase1.votes = {}
        self.mock_phrase2.votes = {}
        assert self.review.is_voting() is False