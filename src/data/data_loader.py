"""Dataset and fold-splitting utilities for token classification."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DebertaV2TokenizerFast


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return int(hash(value))


class PIIDataset(Dataset):
    """Token classification dataset with word-to-token label alignment."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 1024,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item.get("labels", [])

        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        word_ids = tokenized.word_ids()
        aligned_labels: List[int] = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                label_str = labels[word_idx]
                aligned_labels.append(self.label2id.get(label_str, -100))
            else:
                aligned_labels.append(-100)
            prev_word_idx = word_idx

        return {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def _sample_external_data(
    external_data: List[Dict[str, Any]],
    external_fraction: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if external_fraction >= 1.0:
        return external_data
    if external_fraction <= 0.0:
        return []

    rng = random.Random(seed)
    sample_size = max(1, int(len(external_data) * external_fraction))
    return rng.sample(external_data, k=sample_size)


def get_fold_datasets(
    fold_idx: int,
    num_folds: int,
    config,
) -> Tuple[PIIDataset, PIIDataset, Dict[str, int], Dict[int, str], Any]:
    """Load datasets for one fold, with external data added to train only."""
    with open(config.COMPETITION_DATA, "r") as f:
        real_data = json.load(f)

    external_data: List[Dict[str, Any]] = []
    if config.EXTERNAL_DATA and os.path.exists(config.EXTERNAL_DATA):
        with open(config.EXTERNAL_DATA, "r") as f:
            external_data = json.load(f)

    all_labels = set()
    for doc in real_data + external_data:
        all_labels.update(doc["labels"])

    sorted_labels = sorted(list(all_labels))
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_docs: List[Dict[str, Any]] = []
    val_docs: List[Dict[str, Any]] = []

    for doc in real_data:
        doc_id = _safe_int(doc.get("document"))
        if doc_id % num_folds == fold_idx:
            val_docs.append(doc)
        else:
            train_docs.append(doc)

    external_fraction = getattr(config, "EXTERNAL_FRACTION", 1.0)
    if external_data:
        train_docs.extend(_sample_external_data(external_data, external_fraction, config.SEED))

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if (
        "deberta-v" in config.MODEL_NAME.lower()
        and not isinstance(tokenizer, DebertaV2TokenizerFast)
    ):
        tokenizer = DebertaV2TokenizerFast.from_pretrained(config.MODEL_NAME)

    train_ds = PIIDataset(train_docs, tokenizer, label2id, max_length=config.MAX_LENGTH)
    val_ds = PIIDataset(val_docs, tokenizer, label2id, max_length=config.MAX_LENGTH)

    return train_ds, val_ds, label2id, id2label, tokenizer
