import os
import tempfile
from xml.etree.ElementTree import parse
from src.argumentation.sentiment.sentiment_annotation import prepare_bert_trained_dataset
from unittest import mock

class TestPrepareBertTrainedDataset:
    
    def setup_method(self):
        """Setup test data before each test method is executed."""
        # Sample data mimicking the structure from hugging face
        self.test_data = [
            {
                'sentenceId': '1',
                'text': 'The food was delicious but the service was slow.',
                'aspectTerms': [
                    {
                        'term': 'food',
                        'from': '4',
                        'to': '8',
                        'polarity': 'positive'
                    },
                    {
                        'term': 'service',
                        'from': '33',
                        'to': '40',
                        'polarity': 'negative'
                    }
                ]
            },
            {
                'sentenceId': '2',
                'text': 'Great ambiance.',
                'aspectTerms': [
                    {
                        'term': 'ambiance',
                        'from': '6',
                        'to': '14',
                        'polarity': 'positive'
                    }
                ]
            },
            {
                'sentenceId': '3',
                'text': 'Nothing special about this place.',
                'aspectTerms': []
            }
        ]
        
        # Create a temporary file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.xml_path = os.path.join(self.temp_dir.name, 'test_output.xml')
    
    def teardown_method(self):
        """Clean up after each test method is executessd."""
        # Remove temporary directory and its contents
        self.temp_dir.cleanup()
    
    def test_prepare_bert_trained_dataset(self):
        """Comprehensive test for the prepare_bert_trained_dataset function."""
        # Generate the XML file
        prepare_bert_trained_dataset(self.test_data, self.xml_path)
        
        # Verify the file exists and has content
        assert os.path.exists(self.xml_path)
        assert os.path.getsize(self.xml_path) > 0
        
        # Parse the XML file
        tree = parse(self.xml_path)
        root = tree.getroot()
        
        # Check root element name
        assert root.tag == 'reviews'
        
        # Check sentenceId elements
        sentence_ids = root.findall('sentenceId')
        assert len(sentence_ids) == 3
        assert sentence_ids[0].text == '1'
        assert sentence_ids[1].text == '2'
        assert sentence_ids[2].text == '3'
        
        # Check sentences and their text
        sentences = root.findall('sentences')
        assert len(sentences) == 3
        assert sentences[0].find('text').text == 'The food was delicious but the service was slow.'
        assert sentences[1].find('text').text == 'Great ambiance.'
        assert sentences[2].find('text').text == 'Nothing special about this place.'
        
        # Check aspect terms for first sentence (has two terms)
        aspect_terms_1 = sentences[0].find('aspectTerms').findall('term')
        assert len(aspect_terms_1) == 2
        assert aspect_terms_1[0].text == 'food'
        assert aspect_terms_1[0].attrib['from'] == '4'
        assert aspect_terms_1[0].attrib['to'] == '8'
        assert aspect_terms_1[0].attrib['polarity'] == 'positive'
        assert aspect_terms_1[1].text == 'service'
        assert aspect_terms_1[1].attrib['from'] == '33'
        assert aspect_terms_1[1].attrib['to'] == '40'
        assert aspect_terms_1[1].attrib['polarity'] == 'negative'
        
        # Check aspect terms for second sentence (has one term)
        aspect_terms_2 = sentences[1].find('aspectTerms').findall('term')
        assert len(aspect_terms_2) == 1
        assert aspect_terms_2[0].text == 'ambiance'
        assert aspect_terms_2[0].attrib['from'] == '6'
        assert aspect_terms_2[0].attrib['to'] == '14'
        assert aspect_terms_2[0].attrib['polarity'] == 'positive'
        
        # Check aspect terms for third sentence (has no terms)
        aspect_terms_3 = sentences[2].find('aspectTerms').findall('term')
        assert len(aspect_terms_3) == 0

def tearDown(self):
    # Reset your mocks here
    mock.reset_mock()