import os
import sys
import torch
import torch.nn as nn
from functools import partial
from collections import Counter
from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

import config
from src.data_loader import get_fold_datasets
from src.metrics import compute_metrics


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = (
            class_weights.to(self.args.device) if class_weights is not None else None
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_global_weights(config):
    print(">>> Computing Global Class Weights...")
    
    # Load Fold 0 to access the full dataset structure (Train + Val covers all)
    train_ds, val_ds, label2id, _, _ = get_fold_datasets(0, config.NUM_FOLDS, config)
    
    label_counts = Counter()
    total_valid_tokens = 0
    
    def scan_dataset(dataset):
        nonlocal total_valid_tokens
        for i in range(len(dataset)):
            item = dataset[i]
            labels = item['labels'] if isinstance(item, dict) else item.labels
            valid_labels = [l for l in labels if l != -100]
            label_counts.update(valid_labels)
            total_valid_tokens += len(valid_labels)

    scan_dataset(train_ds)
    scan_dataset(val_ds)

    num_classes = len(label2id)
    weights = torch.ones(num_classes)
    
    # Balanced Weight Formula: n_samples / (n_classes * n_count)
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            weights[label_id] = total_valid_tokens / (num_classes * count)
        else:
            weights[label_id] = 1.0
            
    print(f"Global Weights: {weights}")
    return weights


def train_fold(fold, class_weights):
    print(f"\n>>> Starting Fold {fold}/{config.NUM_FOLDS - 1}")

    train_ds, val_ds, label2id, id2label, tokenizer = get_fold_datasets(
        fold, config.NUM_FOLDS, config
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )

    training_args = TrainingArguments(
        output_dir=f"{config.OUTPUT_DIR_BASE}/{config.RUN_NAME}/fold_{fold}",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        dataloader_num_workers=config.NUM_WORKERS,
        fp16=config.FP16,
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="fbeta",
        greater_is_better=True,
        report_to="none",
        logging_steps=50,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=partial(compute_metrics, id2label=id2label),
    )

    trainer.train()

    save_path = f"models/{config.RUN_NAME}/model_fold_{fold}"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    with open(os.path.join(save_path, "DONE"), "w") as f:
        f.write(f"Fold {fold} completed\n")

    del model, trainer, train_ds, val_ds
    torch.cuda.empty_cache()


def main():
    global_weights = compute_global_weights(config)
    
    for i in range(config.NUM_FOLDS):
        train_fold(i, class_weights=global_weights)


if __name__ == "__main__":
    main()