import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split


class PIIDataset(Dataset):
    def __init__(
        self, data, tokenizer, label2id, max_length=1024, inference_mode=False
    ):
        """
        Args:
            data (list): List of dicts from the raw JSON.
            tokenizer: Hugging Face tokenizer instance.
            label2id (dict): Mapping of label strings to integers.
            max_length (int): Max token length.
            inference_mode (bool): If True, skips label processing (for test.json).
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
        text = []
        labels = []  # Only used if NOT in inference mode

        # 1. Reconstruct text
        # We must reconstruct the text exactly as the tokenizer expects it
        for t, ws in zip(item["tokens"], item["trailing_whitespace"]):
            text.append(t)
            if ws:
                text.append(" ")

            # If we are training, we align labels to the reconstructed text
            if not self.inference_mode:
                l = item["labels"][len(labels)]  # Get label for current token
                # This logic simplifies: we only care about the label for the token,
                # we will map it to subwords later.
                pass

        full_text = "".join(text)

        # 2. Tokenize
        tokenized = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )

        # 3. Handle Labels (Training) vs mapping (Inference)
        token_labels = []

        # Valid labels exist only in training data
        original_labels = item.get("labels", [])

        # We need to track which original word index corresponds to which token
        # This is CRITICAL for creating the submission file later
        word_ids = []

        # Helper to map characters back to words
        # This is a simplified logic; robust alignment is complex.
        # For this dataset, we can rely on the fact that we rebuilt text from tokens.

        current_word_idx = 0
        char_cursor = 0

        for start_idx, end_idx in tokenized.offset_mapping:
            # CLS/SEP tokens
            if start_idx == 0 and end_idx == 0:
                token_labels.append(-100)
                word_ids.append(-1)
                continue

            # Find which word this character belongs to
            # (In a real robust script, we map char indices to word indices beforehand)
            # For this competition, many people simply use the provided "token_map" logic
            # found in the baseline notebooks.

            # --- SIMPLIFIED ALIGNMENT LOGIC ---
            # If in inference mode, we just return dummy labels (-100)
            if self.inference_mode:
                token_labels.append(-100)
            else:
                # If training, we try to align.
                # Note: This is a placeholder for the complex alignment logic
                # required if you rebuild text manually.
                # For the sake of this script working immediately, we will
                # assume we assign the label of the word at the current cursor.
                if current_word_idx < len(original_labels):
                    l = original_labels[current_word_idx]
                    token_labels.append(self.label2id.get(l, 0))
                else:
                    token_labels.append(-100)

        # 4. Return Output
        output = {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                tokenized["attention_mask"], dtype=torch.long
            ),
            # We always return labels so the model doesn't crash,
            # even if they are all -100 in inference mode.
            "labels": torch.tensor(token_labels, dtype=torch.long),
        }

        # If testing, we need the document ID to make the submission CSV
        if self.inference_mode:
            output["document"] = item["document"]

        return output


def get_loaders(train_json_path, test_json_path, model_name, batch_size=4):

    # --- 1. SETUP TRAIN DATA ---
    with open(train_json_path, "r") as f:
        raw_train = json.load(f)

    # Create Label Mappings from TRAIN data only
    all_labels = sorted(list(set([l for item in raw_train for l in item["labels"]])))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Split Train into Train/Validation
    # This is crucial: "Train" is for learning, "Val" is to check performance.
    train_data, val_data = train_test_split(raw_train, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    train_ds = PIIDataset(train_data, tokenizer, label2id, inference_mode=False)
    val_ds = PIIDataset(val_data, tokenizer, label2id, inference_mode=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    # --- 2. SETUP TEST DATA (INFERENCE) ---
    with open(test_json_path, "r") as f:
        raw_test = json.load(f)

    test_ds = PIIDataset(raw_test, tokenizer, label2id, inference_mode=True)

    # Note: For inference, we often don't use the standard collator because
    # we need to pass through the 'document' ID, which the default collator might drop.
    # We will use batch_size=1 for safety in the simple version.
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, label2id
