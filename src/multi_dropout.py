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
        raise NotImplementedError("from_pretrained not implemented yet")

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
        raise NotImplementedError("forward not implemented yet")
