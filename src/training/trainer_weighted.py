"""Trainer extensions for class-weighted token classification."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    """HuggingFace Trainer with optional class-weighted loss."""

    def __init__(self, class_weights: Optional[torch.Tensor], ignore_index: int = -100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.ignore_index = ignore_index

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # To avoid "backward through the graph a second time" with gradient checkpointing,
        # we don't pass labels to the model if we're going to compute the loss ourselves.
        if labels is not None:
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
        else:
            outputs = model(**inputs)

        logits = outputs.get("logits")

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=self.ignore_index)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # Fallback for cases where labels are not provided (e.g. some eval paths)
            loss = outputs.get("loss")

        return (loss, outputs) if return_outputs else loss
