import torch
from torch.utils.data import Dataset
from transformers import DebertaV2TokenizerFast, DataCollatorForTokenClassification
import json
from sklearn.model_selection import train_test_split


class PIIDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_length=512, inference_mode=False):
        """
        Args:
            data (list): List of dicts from the raw JSON.
                         Each item has at least "tokens", and during training also "labels".
            tokenizer: Hugging Face tokenizer instance.
            label2id (dict): Mapping of label strings to integers.
            max_length (int): Max token length.
            inference_mode (bool): If True, we ignore labels (for test.json).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.inference_mode = inference_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # We use the provided tokenization directly
        tokens = item["tokens"]

        # Tokenize as pre-split words
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )

        word_ids = enc.word_ids()  # list of length seq_len
        labels_ids = []

        if self.inference_mode:
            # No labels in test mode
            labels_ids = [-100] * len(word_ids)
        else:
            orig_labels = item["labels"]
            for w_id in word_ids:
                if w_id is None:
                    # special tokens: [CLS], [SEP], padding, etc.
                    labels_ids.append(-100)
                else:
                    label_str = orig_labels[w_id]
                    labels_ids.append(self.label2id[label_str])

        enc["labels"] = labels_ids

        out = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(enc["labels"], dtype=torch.long),
        }

        # Preserve document id in inference mode if present
        if self.inference_mode and "document" in item:
            out["document"] = item["document"]

        return out


def get_loaders(train_json_path, test_json_path, model_name, batch_size=4):

    # --- 1. SETUP TRAIN DATA ---
    with open(train_json_path, "r", encoding="utf-8") as f:
        raw_train = json.load(f)

    # Create Label Mappings from TRAIN data only
    all_labels = sorted(list(set(l for item in raw_train for l in item["labels"])))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Split Train into Train/Validation
    train_data, val_data = train_test_split(raw_train, test_size=0.2, random_state=42)
    # debug mode: use smaller dataset
    train_data = train_data[:64]
    val_data = val_data[:16]

    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    train_ds = PIIDataset(train_data, tokenizer, label2id, inference_mode=False)
    val_ds = PIIDataset(val_data, tokenizer, label2id, inference_mode=False)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    # --- 2. SETUP TEST DATA (INFERENCE) ---
    with open(test_json_path, "r", encoding="utf-8") as f:
        raw_test = json.load(f)

    test_ds = PIIDataset(raw_test, tokenizer, label2id, inference_mode=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, label2id, id2label
