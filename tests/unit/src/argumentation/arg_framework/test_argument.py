import pytest 
from unittest.mock import Mock
from src.argumentation.arg_framework.argument import Argument

class TestArgument:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_phrase = Mock()
        self.mock_phrase.text = "phrase_text"
        self.mock_text = "simple_text"

    def test_pos_phrase(self):
        text = self.mock_text
        phrase = self.mock_phrase
        arg = Argument(text, True, "supp", "att", phrase, 10)

        assert arg.polarity == "POS"
        assert arg.phrase == self.mock_phrase.text

    def test_neg_no_phrase(self):
        text = self.mock_text
        arg = Argument(text, False, "supp", "att", None, 10)

        assert arg.polarity == "NEG"
        assert arg.phrase == "-"
        
        