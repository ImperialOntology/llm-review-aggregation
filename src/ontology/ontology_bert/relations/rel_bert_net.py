import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from src.constants import (
    RELATION_NUM_CLASSES,
    HIDDEN_OUTPUT_FEATURES,
    TRAINED_WEIGHTS
)


class RelBertNet(nn.Module):

    def __init__(self):
        super(RelBertNet, self).__init__()
        config = BertConfig.from_pretrained(TRAINED_WEIGHTS)
        self.bert_base = BertModel.from_pretrained(TRAINED_WEIGHTS, config=config)
        self.fc = nn.Linear(HIDDEN_OUTPUT_FEATURES * 2, RELATION_NUM_CLASSES)

    def forward(self, input_ids, attn_mask, fst_indices, snd_indices):
        # BERT
        bert_output = self.bert_base(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

        # max pooling at entity locations
        rng = torch.arange(0, bert_output.shape[0])
        fst_pooled_output = bert_output[rng, fst_indices]
        snd_pooled_output = bert_output[rng, snd_indices]

        # concat pooled outputs from two entities
        combined = torch.cat((fst_pooled_output, snd_pooled_output), dim=1)

        # fc layer (softmax activation done in loss function)
        x = self.fc(combined)
        return x
