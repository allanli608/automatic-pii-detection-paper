import os
import sys
import torch
import torch.nn as nn
from functools import partial
from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

import config
from data_utils.data_loader import get_fold_datasets
from metrics import compute_metrics

class WeightedTrainer(Trainer):
    """
    Subclassing Trainer to implement Class-Weighted Cross Entropy.
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Flatten for loss calculation: (N * seq_len, num_classes)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def train_fold(fold):
    print(f"\n>>> Starting Fold {fold}/{config.NUM_FOLDS - 1}")
    
    # Prepare Data
    train_ds, val_ds, label2id, id2label, tokenizer = get_fold_datasets(fold, config.NUM_FOLDS, config)
    
    # Setup Weights: O=0.05, PII=1.0
    num_labels = len(label2id)
    weights = torch.ones(num_labels)
    if "O" in label2id:
        weights[label2id["O"]] = 0.05
    
    # Model Init
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True    
    )

    training_args = TrainingArguments(
        output_dir=f"{config.OUTPUT_DIR_BASE}/fold_{fold}",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        dataloader_num_workers=config.NUM_WORKERS,
        fp16=config.FP16,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="fbeta",
        greater_is_better=True,
        report_to="none",
        logging_steps=50
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=partial(compute_metrics, id2label=id2label)
    )

    trainer.train()

    # Save artifacts
    save_path = f"models/model_fold_{fold}"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Cleanup
    del model, trainer, train_ds, val_ds
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for i in range(config.NUM_FOLDS):
        train_fold(i)