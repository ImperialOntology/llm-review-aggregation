import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from transformers import BertTokenizer
from src.argumentation.sentiment.tdbertnet import TRAINED_WEIGHTS, HIDDEN_OUTPUT_FEATURES
import re

MAX_SEQ_LEN = 128
polarity_indices = {'positive': 0, 'negative': 1, 'neutral': 2, 'conflict': 3}
tokenizer = BertTokenizer.from_pretrained(TRAINED_WEIGHTS)
MASK_TOKEN = '[MASK]'


def generate_batch(batch):
    encoded = tokenizer.batch_encode_plus([entry['text'] for entry in batch], add_special_tokens=True,
                                          max_length=MAX_SEQ_LEN, padding='max_length', truncation=True,
                                          return_tensors='pt')
    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']

    max_tg_len = max(entry['to'] - entry['from'] for entry in batch)
    target_indices = torch.tensor([[[min(t, entry['to'])] * HIDDEN_OUTPUT_FEATURES
                                    for t in range(entry['from'], entry['from'] + max_tg_len + 1)]
                                   for entry in batch])

    polarity_labels = torch.tensor([entry['polarity'] for entry in batch])

    return input_ids, attn_mask, target_indices, polarity_labels


def token_for_char(char_idx, text, tokens):
    compressed_idx = len(re.sub(r'\s+', '', text[:char_idx+1])) - 1
    token_idx = -1
    while compressed_idx >= 0:
        token_idx += 1
        compressed_idx -= len(tokens[token_idx].replace('##', ''))
    return token_idx


def polarity_index(polarity):
    return polarity_indices[polarity]


class BertDataset(Dataset):

    def __init__(self):
        self.data = []
        self.mask_target = False

    @staticmethod
    def from_file(file, mask_target=False):
        '''
        Convert from the xml file to the dataset that could be used in pytorch

        Args:
        file(src): location of the xml file

        Returns:
        dataset: pytorch dataset
        '''
        dataset = BertDataset()
        tree = ET.parse(file)
        dataset.data = []
        dataset.mask_target = mask_target
        root = tree.getroot()
        for review in root.findall('sentences'):
            text = review.find('text').text
            aspect_terms = review.find('aspectTerms')
            if aspect_terms:
                for term in aspect_terms:
                    char_from = int(term.attrib['from'])
                    char_to = int(term.attrib['to'])
                    polarity = term.attrib['polarity']
                    dataset.data.append(
                        (Instance(text, char_from, char_to), polarity))
        return dataset

    @staticmethod
    def from_data(data):
        '''
        Convert from the data file into pytorch dataset

        Args:
        data: Curently we do not have a fixed format

        Returns:
        dataset: pytorch dataset
        '''
        dataset = BertDataset()
        dataset.data = [(Instance(text, char_from, char_to), 'neutral')
                        for text, char_from, char_to in data]
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance, polarity_str = self.data[idx]
        text, idx_from, idx_to = instance.get(self.mask_target)
        polarity = polarity_index(polarity_str)

        return {'text': text, 'from': idx_from, 'to': idx_to, 'polarity': polarity}


class Instance:
    def __init__(self, text, char_from, char_to):
        self.text = text
        self.char_from = char_from
        self.char_to = char_to

    def get(self, mask_target):
        '''
        Get the text, inx_from and indx_to from the dataset
        '''
        tokens = tokenizer.tokenize(self.text)
        idx_from = token_for_char(self.char_from, self.text, tokens)
        idx_to = token_for_char(self.char_to-1, self.text, tokens) + 1
        if mask_target:
            tokens[idx_from:idx_to] = [MASK_TOKEN] * (idx_to - idx_from)
        return self.text, idx_from + 1, idx_to  # +1 for [CLS] token

    def to_tensor(self):
        text, idx_from, idx_to = self.get(mask_target=False)
        text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_SEQ_LEN,
                                     return_tensors='pt')
        target_indices = torch.tensor(
            [[[t] * HIDDEN_OUTPUT_FEATURES for t in range(idx_from, idx_to + 1)]])
        return text, target_indices
