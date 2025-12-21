"""
Multi-sample dropout token classification model.

This module defines a thin wrapper around a HuggingFace encoder
(e.g. DeBERTa) that applies multiple dropout masks to the same
hidden states during training and averages the resulting logits.

This file only contains the model definition.
Training logic (loss weighting, metrics, etc.) lives elsewhere.
"""

from typing import Optional, Dict, Any

import torch
from torch import nn

from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import CrossEntropyLoss



class MultiDropoutTokenClassifier(nn.Module):
    """
    Encoder + token classification head with multi-sample dropout.

    During training:
        logits = average over K dropout samples

    During evaluation:
        logits = single forward pass (deterministic)
    """

    def __init__(
        self,
        backbone: nn.Module,
        config,
        num_labels: int,
        k: int = 5,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        # store core components
        self.backbone = backbone
        self.config = config
        self.num_labels = num_labels

        # multi-dropout parameters
        self.k = k
        self.dropout = nn.Dropout(dropout_p)

        # classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_labels: int,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        k: int = 5,
        dropout_p: float = 0.2,
        **kwargs,
    ):
        """
        Factory method to construct the model from a pretrained HF checkpoint.
        """
    
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        config.num_labels = num_labels
        if id2label is not None:
            config.id2label = id2label
        if label2id is not None:
            config.label2id = label2id

        backbone = AutoModel.from_pretrained(model_name_or_path, config=config)
        return cls(
            backbone=backbone,
            config=config,
            num_labels=num_labels,
            k=k,
            dropout_p=dropout_p,
        )


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        """
        Forward pass. To be implemented.
        """
  
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        hidden = outputs.last_hidden_state  # [B, T, H]

        # Multi-sample dropout during training; single pass during eval
        if self.training and self.k > 1:
            logits_sum = 0.0
            for _ in range(self.k):
                logits_sum = logits_sum + self.classifier(self.dropout(hidden))
            logits = logits_sum / self.k
        else:
            logits = self.classifier(self.dropout(hidden))

        loss = None
        if labels is not None:
            # labels: [B, T], ignore_index=-100 for padded/ignored positions
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
