import torch
import pandas as pd
from typing import List
from ast import literal_eval
from src.ontology.ontology_bert.base.bert_dataset_base import BaseDataset
from src.constants import MAX_SEQ_LEN


class RelationInstance:

    def __init__(self, tokens, entity_indices, label=None, entity_labels=None):
        self.tokens = tokens
        self.fst_idx = entity_indices[0]
        self.snd_idx = entity_indices[1]
        self.label = label
        self.fst_a = entity_labels[0] if entity_labels else None
        self.snd_a = entity_labels[1] if entity_labels else None


class RelationDataset(BaseDataset):
    data_file_suffix = 'relation_instances'

    def __init__(self, df, training=True, size=None):
        super().__init__(df, training, size)

    @staticmethod
    def _eval_or_string(x):
        if isinstance(x, list):
            return x
        else:
            return literal_eval(x)

    @staticmethod
    def filter_df(df: pd.DataFrame):
        """
        Filters the DataFrame to remove rows whose entity indexes are
        greater than allowed max sequence length.

        :param df: DataFrame to filter
        :return: filtered DataFrame
        """
        df = df[df['entity_indices'].apply(
            lambda x: max(RelationDataset._eval_or_string(x)) < MAX_SEQ_LEN)]
        df = df.copy().reset_index(drop=True)

        return df

    def instance_from_row(self, row):
        if self.training:
            return RelationInstance(literal_eval(row['tokens']),
                                    literal_eval(row['entity_indices']),
                                    label=row['label'])
        else:
            return RelationInstance(row['tokens'],
                                    row['entity_indices'],
                                    entity_labels=row['entity_labels'])

    @staticmethod
    def generate_batch(instances: List[RelationInstance]):
        input_ids, attn_mask = BaseDataset.generate_batch(instances)
        fst_indices = torch.tensor([instance.fst_idx for instance in instances])
        snd_indices = torch.tensor([instance.snd_idx for instance in instances])
        labels = torch.tensor([instance.label for instance in instances])
        return input_ids, attn_mask, fst_indices, snd_indices, labels

    @staticmethod
    def generate_production_batch(instances: List[RelationInstance]):
        input_ids, attn_mask, instances = BaseDataset.generate_production_batch(instances)
        fst_indices = torch.tensor([instance.fst_idx for instance in instances])
        snd_indices = torch.tensor([instance.snd_idx for instance in instances])
        return input_ids, attn_mask, fst_indices, snd_indices, instances
