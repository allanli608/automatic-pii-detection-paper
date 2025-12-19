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


# class MultiSampleDropoutTokenClassifier(nn.Module):

#     def __init__(self, base_model, num_labels, k=5, p=0.2): ## K dropout samples, dropout prob p
#         ...
#     def forward(...):
#         H = base_model(...).last_hidden_state
#         logits = mean( classifier(dropout_i(H)) for i in 1..k )
#         return TokenClassifierOutput(logits=logits, ...)

