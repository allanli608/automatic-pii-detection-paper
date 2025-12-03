import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split


ALL_LABELS = [
    "B-EMAIL",
    "B-ID_NUM",
    "B-NAME_STUDENT",
    "B-PHONE_NUM",
    "B-STREET_ADDRESS",
    "B-URL_PERSONAL",
    "B-USERNAME",
    "I-ID_NUM",
    "I-NAME_STUDENT",
    "I-PHONE_NUM",
    "I-STREET_ADDRESS",
    "I-URL_PERSONAL",
    "O",
]
LABEL2ID = {l: i for i, l in enumerate(ALL_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class PIIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = []
        token_map = []
        labels = []

        for i, (t, ws) in enumerate(
            zip(example["tokens"], example["trailing_whitespace"])
        ):
            text.append(t)
            token_map.extend([i] * len(t))

            if not self.is_test:
                l = example["labels"][i]
                labels.extend([l] * len(t))

            if ws:
                text.append(" ")
                token_map.append(-1)
                if not self.is_test:
                    labels.append("O")

        full_text = "".join(text)
        tokenized = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )

        token_labels = []

        for start_idx, end_idx in tokenized.offset_mapping:
            if start_idx == 0 and end_idx == 0:
                token_labels.append(-100)
                continue

            if full_text[start_idx].isspace():
                start_idx += 1

            try:
                if start_idx < len(labels):
                    label_str = labels[start_idx]
                    token_labels.append(LABEL2ID.get(label_str, LABEL2ID["O"]))
                else:
                    token_labels.append(-100)
            except IndexError:
                token_labels.append(-100)

        output = {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                tokenized["attention_mask"], dtype=torch.long
            ),
        }

        if not self.is_test:
            output["labels"] = torch.tensor(token_labels, dtype=torch.long)
        else:
            output["document"] = example["document"]

        return output


def get_datasets(json_path, model_path="microsoft/deberta-v3-base"):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create Label Mappings from TRAIN data only
    all_labels = sorted(list(set([l for item in raw_train for l in item["labels"]])))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Split Train into Train/Validation
    # This is crucial: "Train" is for learning, "Val" is to check performance.
    train_data, val_data = train_test_split(raw_train, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    train_dataset = PIIDataset(train_data, tokenizer, is_test=False)
    val_dataset = PIIDataset(val_data, tokenizer, is_test=False)

    return train_dataset, val_dataset, tokenizer
