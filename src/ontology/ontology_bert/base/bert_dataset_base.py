from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from src.constants import (
    MAX_SEQ_LEN,
    TRAINED_WEIGHTS,
)
from logger import logger


class BaseDataset(Dataset):
    """Abstract base class for entity and relation datasets for BERT.
    """
    tokenizer = BertTokenizer.from_pretrained(TRAINED_WEIGHTS)

    def __init__(self, df, training=True, size=None):
        """Initialize the BaseDataset.

        Args:
            df (pd.DataFrame): The DataFrame containing the dataset.
                The specific columns depend on the dataset type. 
                See child classes for reference.
            training (bool): Whether the dataset is for training (default: True).
            size (int, optional): The maximum size of the dataset. If specified, the dataset is sampled to this size.
        """
        self.df = df
        self.training = training
        # Sample data if a size is specified
        if size is not None and size < len(self):
            self.df = self.df.sample(size, replace=False)

    def __len__(self):
        """Get the number of instances in the dataset.
        """
        return len(self.df.index)

    def __getitem__(self, idx):
        """Get an instance from the dataset by index.
        """
        return self.instance_from_row(self.df.iloc[idx])

    @classmethod
    def for_extraction(cls, df):
        df = cls.filter_df(df)  # filter the DataFrame
        dataset = cls(df, training=False)
        logger.info(f'Obtained dataset of size {len(dataset)}')
        return dataset

    @classmethod
    def from_files(cls,
                   categories: List[str],
                   path: str,
                   valid_frac=None,
                   size=None,
                   random_state=42):
        """
        Combine instances from all categories into a single dataset.

        Args:
            categories (List[str]): A list of categories to include.
            path (str): The path to the directory containing the data files.
            valid_frac (float, optional): The fraction of the dataset to use for validation.
            size (int, optional): The maximum size of the dataset.
            random_state (int): The random seed for shuffling (default: 42).

        Returns:
            tuple: A tuple containing:
                - BaseDataset: The training dataset.
                - BaseDataset or None: The validation dataset, if `valid_frac` is specified.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.\n"
                                    "You might need to clone the data repository from\n"
                                    "git@gitlab.doc.ic.ac.uk:g24mai05/bert-train-data.git")

        dfs = []
        for category in categories:
            category_path = path / f'{category}_{cls.data_file_suffix}.tsv'
            if not category_path.exists():
                raise FileNotFoundError(f"File {category_path} does not exist.\n"
                                        "You might need to clone the data repository from\n"
                                        "git@gitlab.doc.ic.ac.uk:g24mai05/bert-train-data.git")
            df = pd.read_csv(category_path, sep='\t')
            dfs.append(df)
        dataset = pd.concat(dfs, ignore_index=True)

        if size is None or size > len(dataset):
            size = len(dataset)

        dataset = dataset.sample(size,
                                 replace=False,
                                 random_state=random_state).reset_index(drop=True)
        dataset = cls.filter_df(dataset)

        if valid_frac is None:
            logger.info(f'Obtained dataset of size {len(dataset)}')
            return cls(dataset), None
        else:
            split_idx = int(len(dataset) * (1 - valid_frac))
            train_df, valid_df = np.split(dataset, [split_idx], axis=0)
            train_df = pd.DataFrame(train_df, columns=dataset.columns).reset_index(drop=True)
            valid_df = pd.DataFrame(valid_df, columns=dataset.columns).reset_index(drop=True)
            logger.info(f'Obtained train set of size {len(train_df)}, and validation set of size {len(valid_df)}')
            return cls(train_df), cls(valid_df)

    @staticmethod
    def filter_df(df: pd.DataFrame):
        """Filter the DataFrame to remove invalid rows.
        """
        raise NotImplementedError("Subclasses must implement filter_df")

    def instance_from_row(self, row):
        """Generate an instance from a row in the DataFrame.
        """
        raise NotImplementedError("Subclasses must implement instance_from_row")

    @classmethod
    def generate_batch(cls, instances: List):
        """
        Generate a batch of input IDs and attention masks from a list of instances.

        Args:
            instances (List): A list of instances.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The input IDs.
                - torch.Tensor: The attention masks.
        """
        tokens_list = [instance.tokens for instance in instances]
        encoded = cls.tokenizer.batch_encode_plus(tokens_list,
                                                  add_special_tokens=True,
                                                  max_length=MAX_SEQ_LEN,
                                                  padding='max_length',
                                                  is_split_into_words=True,
                                                  truncation=True,
                                                  return_tensors='pt')
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        return input_ids, attn_mask

    @classmethod
    def generate_production_batch(cls, instances: List):
        """
        Generate a batch of input IDs, attention masks, and instances for production.

        Args:
            instances (List): A list of instances.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The input IDs.
                - torch.Tensor: The attention masks.
                - List: The original instances.
        """
        input_ids, attn_mask = cls.generate_batch(instances)
        return input_ids, attn_mask, instances
