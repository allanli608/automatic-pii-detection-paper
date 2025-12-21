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

import json
from pathlib import Path
from transformers import PretrainedConfig


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
        num_labels: int | None = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        k: int = 5,
        dropout_p: float = 0.2,
        **kwargs,
    ):
        """
        Construct from a HF checkpoint name OR a local directory created by save_pretrained().

        If loading from a local directory:
          - loads config.json
          - loads backbone weights via AutoModel.from_pretrained(local_dir)
          - loads wrapper head weights from wrapper_head.pt if present
          - loads wrapper_meta.json (k, dropout_p, num_labels) if present
        """
        path = Path(model_name_or_path)
        is_local_dir = path.exists() and path.is_dir()

        # 1) Load config
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)

        # 2) If local dir, try to load wrapper meta (k/dropout/num_labels)
        if is_local_dir:
            meta_path = path / "wrapper_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                # Allow caller to override these, but default from meta if not passed
                if num_labels is None:
                    num_labels = meta.get("num_labels")
                if "k" in meta and k is None:
                    k = meta["k"]
                if "dropout_p" in meta and dropout_p is None:
                    dropout_p = meta["dropout_p"]

        if num_labels is None:
            raise ValueError("num_labels must be provided (or present in wrapper_meta.json).")

        config.num_labels = num_labels
        if id2label is not None:
            config.id2label = id2label
        if label2id is not None:
            config.label2id = label2id

        # 3) Load backbone
        backbone = AutoModel.from_pretrained(model_name_or_path, config=config)

        # 4) Build wrapper
        model = cls(
            backbone=backbone,
            config=config,
            num_labels=num_labels,
            k=k,
            dropout_p=dropout_p,
        )

        # 5) If local dir, load head weights
        if is_local_dir:
            head_path = path / "wrapper_head.pt"
            if head_path.exists():
                state = torch.load(head_path, map_location="cpu")
                model.classifier.load_state_dict(state)

        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save in a HuggingFace-like folder format:
          - config.json (via config.save_pretrained)
          - backbone weights (via backbone.save_pretrained)
          - wrapper_head.pt (classifier weights)
          - wrapper_meta.json (k/dropout/num_labels)
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save backbone + config in HF format
        # AutoModel supports save_pretrained; this writes weights + config.json too.
        # But we still ensure config is present and includes label mappings.
        self.config.save_pretrained(str(save_dir))
        self.backbone.save_pretrained(str(save_dir))

        # Save wrapper head separately
        torch.save(self.classifier.state_dict(), save_dir / "wrapper_head.pt")

        # Save wrapper metadata
        meta = {
            "num_labels": self.num_labels,
            "k": self.k,
            "dropout_p": float(self.dropout.p),
        }
        (save_dir / "wrapper_meta.json").write_text(json.dumps(meta, indent=2))



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
