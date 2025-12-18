## WORK IN PROGRESS: recompute evaluation metrics on saved checkpoint
## probably better alternative: save metrics during training with TrainerCallback

import os, json
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from data_loader import get_datasets  
from metrics import compute_metrics
from config import ID2LABEL  # adjust to where id2label lives

ckpt="output/fold_0/checkpoint-2352"

model = AutoModelForTokenClassification.from_pretrained(ckpt)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

# build your fold_0 eval dataset exactly as in training
train_ds, eval_ds = get_datasets(fold=0)  # <-- adjust this to your loader

args = TrainingArguments(
    output_dir="/tmp/eval",
    per_device_eval_batch_size=8,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=lambda p: compute_metrics(p, ID2LABEL),
)

print(trainer.evaluate())
