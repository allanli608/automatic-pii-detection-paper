import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DebertaV2TokenizerFast

class PIIDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item.get("labels", [])

        # Pre-tokenized input requires is_split_into_words=True
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        
        word_ids = tokenized.word_ids()
        aligned_labels = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                # Label only the first token of a word
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

def get_fold_datasets(fold_idx, num_folds, config):
    """
    Loads data and splits based on Modulo-4 logic on document ID.
    External data is added strictly to the training set.
    """
    # Load Real Data
    with open(config.COMPETITION_DATA, "r") as f:
        real_data = json.load(f)

    # Load External Data
    external_data = []
    if config.EXTERNAL_DATA and os.path.exists(config.EXTERNAL_DATA):
        with open(config.EXTERNAL_DATA, "r") as f:
            external_data = json.load(f)

    # Build dynamic label map from all available data
    all_labels = set()
    for doc in real_data + external_data:
        all_labels.update(doc["labels"])
    
    sorted_labels = sorted(list(all_labels))
    label2id = {l: i for i, l in enumerate(sorted_labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Split Logic
    train_docs = []
    val_docs = []

    for doc in real_data:
        try:
            doc_id = int(doc["document"])
        except ValueError:
            doc_id = int(hash(doc["document"])) # Fallback for non-int IDs

        if doc_id % num_folds == fold_idx:
            val_docs.append(doc)
        else:
            train_docs.append(doc)

    # Inject synthetic data into train only
    train_docs.extend(external_data)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if "deberta-v" in config.MODEL_NAME.lower() and not isinstance(tokenizer, DebertaV2TokenizerFast):
        tokenizer = DebertaV2TokenizerFast.from_pretrained(config.MODEL_NAME)

    train_ds = PIIDataset(train_docs, tokenizer, label2id)
    val_ds = PIIDataset(val_docs, tokenizer, label2id)

    return train_ds, val_ds, label2id, id2label, tokenizer