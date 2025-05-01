import re
import unicodedata
from abc import ABC
from bs4 import BeautifulSoup


class DataProcessor(ABC):
    '''Abstract class for processing raw datasets.'''

    @staticmethod
    def _strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    @staticmethod
    def _remove_special_characters(text, punctuation=".,!?;:\"'"):
        punctuation_pattern = re.escape(punctuation)
        pattern = r'[^a-zA-Z0-9\s' + punctuation_pattern + r']'
        text = re.sub(pattern, '', text)
        return text

    @staticmethod
    def _remove_non_ascii(text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    @staticmethod
    def _clean_text(input):
        cleaned = DataProcessor._strip_html(input)
        cleaned = DataProcessor._remove_special_characters(cleaned)
        cleaned = DataProcessor._remove_non_ascii(cleaned)
        cleaned = cleaned.lower()
        return cleaned

    @staticmethod
    def process_metadata(raw_metadata):
        ''' Abstract method to process metadata '''
        pass

    @staticmethod
    def process_reviews(raw_metadata):
        ''' Abstract method to process reviews '''
        pass


class DataManager(ABC):
    '''Responsible for loading, modifying, saving, and serving datasets.'''

    def __init__(self):
        pass

    def load_data(self):
        '''Loads metadata and review data for the specified category.
        '''
        raise NotImplementedError("Subclasses should implement this method.")

    def process_data(self):
        '''Processes raw metadata and reviews into the required datasets.
        '''
        raise NotImplementedError("Subclasses should implement this method.")

    def save_data(self, products_dataset, reviews_dataset):
        '''Saves the datasets to disk (or csv).

        Args: 
            Dataset (products_dataset): A dataset containing product information
            Dataset (reviews_dataset): A dataset containing review information

        Returns: 
            None
        '''
        raise NotImplementedError("Subclasses should implement this method.")

    def serve_data(self):
        '''Serves the datasets to be used.

        Returns: 
            Dataset (double): The two dictionaries containing products and reviews.
        '''
        raise NotImplementedError("Subclasses should implement this method.")
