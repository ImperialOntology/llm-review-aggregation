import torch
import pandas as pd
from typing import List
from ast import literal_eval
from src.ontology.ontology_bert.base.bert_dataset_base import BaseDataset
from src.constants import MAX_SEQ_LEN


class EntityInstance:

    def __init__(self, tokens, entity_idx, label=None, entity=None):
        self.tokens = tokens
        self.entity_idx = entity_idx
        self.label = label
        self.entity = entity


class EntityDataset(BaseDataset):
    data_file_suffix = 'entity_instances'

    def __init__(self, df, training=True, size=None):
        super().__init__(df, training, size)

    @staticmethod
    def filter_df(df: pd.DataFrame):
        """
        Filters the DataFrame to remove rows whose entity indexes are
        greater than allowed max sequence length.

        :param df: DataFrame to filter
        :return: filtered DataFrame
        """
        df = df[df['entity_idx'] < MAX_SEQ_LEN]
        df = df.copy().reset_index(drop=True)

        return df

    def instance_from_row(self, row):
        if self.training:
            return EntityInstance(literal_eval(row['tokens']),
                                  row['entity_idx'],
                                  label=row['label'])
        else:
            return EntityInstance(row['tokens'],
                                  row['entity_idx'],
                                  entity=row['entity'])

    @staticmethod
    def generate_batch(instances: List[EntityInstance]):
        input_ids, attn_mask = BaseDataset.generate_batch(instances)
        entity_indices = torch.tensor([instance.entity_idx for instance in instances])
        labels = torch.tensor([instance.label for instance in instances])
        return input_ids, attn_mask, entity_indices, labels

    @staticmethod
    def generate_production_batch(instances: List[EntityInstance]):
        input_ids, attn_mask, instances = BaseDataset.generate_production_batch(instances)
        entity_indices = torch.tensor([instance.entity_idx for instance in instances])
        return input_ids, attn_mask, entity_indices, instances
