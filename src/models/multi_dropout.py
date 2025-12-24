"""Multi-sample dropout token classification model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput


class MultiDropoutTokenClassifier(nn.Module):
    """Encoder + token classification head with multi-sample dropout."""

    def __init__(
        self,
        backbone: nn.Module,
        config,
        num_labels: int,
        k: int = 5,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.num_labels = num_labels

        self.k = k
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_labels: Optional[int] = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        k: Optional[int] = 5,
        dropout_p: Optional[float] = 0.2,
        **kwargs,
    ) -> "MultiDropoutTokenClassifier":
        """Construct from a HF checkpoint or a local save directory."""
        path = Path(model_name_or_path)
        is_local_dir = path.exists() and path.is_dir()

        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)

        if is_local_dir:
            meta_path = path / "wrapper_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if num_labels is None:
                    num_labels = meta.get("num_labels")
                if k is None:
                    k = meta.get("k")
                if dropout_p is None:
                    dropout_p = meta.get("dropout_p")

        if num_labels is None:
            raise ValueError("num_labels must be provided (or present in wrapper_meta.json).")

        config.num_labels = num_labels
        if id2label is not None:
            config.id2label = id2label
        if label2id is not None:
            config.label2id = label2id

        backbone = AutoModel.from_pretrained(model_name_or_path, config=config)

        model = cls(
            backbone=backbone,
            config=config,
            num_labels=num_labels,
            k=k or 1,
            dropout_p=dropout_p or 0.0,
        )

        if is_local_dir:
            head_path = path / "wrapper_head.pt"
            if head_path.exists():
                state = torch.load(head_path, map_location="cpu")
                model.classifier.load_state_dict(state)

        return model

    def save_pretrained(self, save_directory: str) -> None:
        """Save backbone, config, and wrapper head in a HF-like directory."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.config.save_pretrained(str(save_dir))
        self.backbone.save_pretrained(str(save_dir))

        torch.save(self.classifier.state_dict(), save_dir / "wrapper_head.pt")

        meta = {
            "num_labels": self.num_labels,
            "k": self.k,
            "dropout_p": float(self.dropout.p),
        }
        (save_dir / "wrapper_meta.json").write_text(json.dumps(meta, indent=2))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            return self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        if hasattr(self.backbone, "gradient_checkpointing"):
            self.backbone.gradient_checkpointing = True
            return None
        raise AttributeError("Backbone does not support gradient checkpointing.")

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            return self.backbone.gradient_checkpointing_disable()
        if hasattr(self.backbone, "gradient_checkpointing"):
            self.backbone.gradient_checkpointing = False
            return None
        raise AttributeError("Backbone does not support gradient checkpointing.")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        hidden = outputs.last_hidden_state

        if self.training and self.k > 1:
            logits_sum = 0.0
            for _ in range(self.k):
                logits_sum = logits_sum + self.classifier(self.dropout(hidden))
            logits = logits_sum / self.k
        else:
            logits = self.classifier(self.dropout(hidden))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
