import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

HIDDEN_OUTPUT_FEATURES = 768
TRAINED_WEIGHTS = 'bert-base-uncased'


class TDBertNet(nn.Module):

    def __init__(self, num_class):
        super(TDBertNet, self).__init__()
        config = BertConfig.from_pretrained(TRAINED_WEIGHTS)
        self.bert_base = BertModel.from_pretrained(
            TRAINED_WEIGHTS, config=config)
        # n of hidden features, n of output labels
        self.fc = nn.Linear(HIDDEN_OUTPUT_FEATURES, num_class)

    def forward(self, input_ids, attn_mask, target_indices):
        # BERT
        outputs = self.bert_base(input_ids=input_ids, attention_mask=attn_mask)
        bert_output = outputs.last_hidden_state
        # bert_output, _ = self.bert_base(input_ids=input_ids, attention_mask=attn_mask)
        # max pooling at target locations
        target_outputs = torch.gather(bert_output, dim=1, index=target_indices)
        pooled_output = torch.max(target_outputs, dim=1)[0]
        # fc layer with softmax activation
        x = F.softmax(self.fc(pooled_output), 1)
        return x
