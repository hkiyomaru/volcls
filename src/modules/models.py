from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel


class Encoder(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs[1]
        return pooler_output


class Classifier(nn.Module):
    def __init__(self, hidden_size: int):
        super(Classifier, self).__init__()
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, pooler_output: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        logits = self.cls(pooler_output).view(-1)
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.view(-1))
        return logits, loss


class Discriminator(nn.Module):
    def __init__(self, hidden_size: int):
        super(Discriminator, self).__init__()
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, pooler_output: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        logits = self.cls(pooler_output).view(-1)
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.view(-1))
            return logits, loss
        return logits
