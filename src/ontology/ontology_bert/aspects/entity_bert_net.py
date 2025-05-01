import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from src.constants import (
    ENTITY_NUM_CLASSES,
    HIDDEN_OUTPUT_FEATURES,
    TRAINED_WEIGHTS)


class EntityBertNet(nn.Module):

    def __init__(self):
        super(EntityBertNet, self).__init__()
        config = BertConfig.from_pretrained(TRAINED_WEIGHTS)
        self.bert_base = BertModel.from_pretrained(TRAINED_WEIGHTS, config=config)
        self.fc = nn.Linear(HIDDEN_OUTPUT_FEATURES, ENTITY_NUM_CLASSES)

    def forward(self, input_ids, attn_mask, entity_indices):
        # BERT
        bert_output = self.bert_base(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

        # max pooling at entity locations
        entity_pooled_output = bert_output[torch.arange(0, bert_output.shape[0]), entity_indices]

        # fc layer (softmax activation done in loss function)
        x = self.fc(entity_pooled_output)
        return x
