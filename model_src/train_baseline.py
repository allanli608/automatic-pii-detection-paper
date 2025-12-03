import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import (
    MODEL_NAME,
    TRAIN_JSON,
    TEST_JSON,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE as LR,
    WARMUP_FRACTION,
    DEVICE,
)
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup
from data_utils.data_loader import get_loaders
from metrics import compute_p_r_f5_from_batches
from torch.optim import AdamW

import torch

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # batch is already padded by DataCollatorForTokenClassification
        batch = {k: v.to(device) for k, v in batch.items() if k != "document"}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, device, id2label):
    model.eval()
    total_loss = 0.0

    all_logits = []
    all_labels = []

    for batch in loader:
        # drop 'document' if present
        batch_on_device = {k: v.to(device) for k, v in batch.items() if k != "document"}

        outputs = model(**batch_on_device)
        loss = outputs.loss
        total_loss += loss.item()

        all_logits.append(outputs.logits.cpu())
        all_labels.append(batch_on_device["labels"].cpu())

    avg_loss = total_loss / len(loader)
    metrics = compute_p_r_f5_from_batches(all_logits, all_labels, id2label)

    print(f"val_loss={avg_loss:.4f}, "
          f"precision={metrics['precision_pii']:.4f}, "
          f"recall={metrics['recall_pii']:.4f}, "
          f"f5={metrics['f5_pii']:.4f}")

    return avg_loss, metrics


def main():
    print("Using device:", DEVICE)

    # 1. Get loaders + label mappings from your helper
    train_loader, val_loader, test_loader, label2id, id2label = get_loaders(
        TRAIN_JSON,
        TEST_JSON,
        MODEL_NAME,
        batch_size=BATCH_SIZE,
    )

    num_labels = len(label2id)

    # 2. Load DeBERTa for token classification
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    # 3. Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LR)

    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 4. Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss = eval_one_epoch(model, val_loader, DEVICE)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # 5. Save model at the end
    model.save_pretrained("deberta_baseline_model")
    print("Model saved to deberta_baseline_model/")


if __name__ == "__main__":
    main()
