"""Fold training entry point."""

from __future__ import annotations

import inspect
import json
import math
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
)

from src.data.data_loader import get_fold_datasets
from src.inference.predict import predict_logits, save_fold_logits
from src.metrics.metrics import compute_metrics
from src.models.multi_dropout import MultiDropoutTokenClassifier
from src.training.trainer_weighted import WeightedTrainer
from src.training.weights import (
    make_dynamic_weights_from_datasets,
    make_o_weighting,
    make_uniform_weights,
)

StatusHook = Callable[[int, str, Optional[Dict[str, Any]]], None]


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _cleanup_checkpoints(path: Path) -> None:
    for child in path.glob("checkpoint-*"):
        shutil.rmtree(child, ignore_errors=True)


def _filtered_training_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(TrainingArguments.__init__)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _build_training_args(config, output_dir: Path, train_len: int) -> TrainingArguments:
    eval_strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )

    save_strategy = "epoch"
    save_steps = None
    if getattr(config, "SAVE_EVERY_N_EPOCHS", None):
        steps_per_epoch = max(
            1,
            math.ceil(
                train_len
                / (config.BATCH_SIZE * max(1, config.GRADIENT_ACCUMULATION_STEPS))
            ),
        )
        save_steps = int(steps_per_epoch * config.SAVE_EVERY_N_EPOCHS)
        save_strategy = "steps"
    elif config.SAVE_TOTAL_LIMIT != 1:
        save_strategy = "no"

    args_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": config.LEARNING_RATE,
        "per_device_train_batch_size": config.BATCH_SIZE,
        "per_device_eval_batch_size": config.BATCH_SIZE,
        "num_train_epochs": config.NUM_EPOCHS,
        "warmup_ratio": config.WARMUP_RATIO,
        "weight_decay": config.WEIGHT_DECAY,
        "dataloader_num_workers": config.NUM_WORKERS,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "fp16": config.FP16,
        "bf16": config.BF16,
        "gradient_checkpointing": config.GRADIENT_CHECKPOINTING,
        "gradient_checkpointing_kwargs": None,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "eval_accumulation_steps": config.EVAL_ACCUMULATION_STEPS,
        "load_best_model_at_end": config.SAVE_BEST_ONLY,
        "metric_for_best_model": config.METRIC_FOR_BEST_MODEL,
        "greater_is_better": True,
        "report_to": config.REPORT_TO,
        "logging_steps": config.LOGGING_STEPS,
        "save_total_limit": config.SAVE_TOTAL_LIMIT,
        "save_strategy": save_strategy,
    }
    args_kwargs[eval_strategy_key] = config.EVAL_STRATEGY

    if save_steps is not None:
        args_kwargs["save_steps"] = max(1, save_steps)

    if getattr(config, "TORCH_COMPILE", False):
        args_kwargs["torch_compile"] = True

    if config.GRADIENT_CHECKPOINTING:
        gc_kwargs = getattr(config, "GRADIENT_CHECKPOINTING_KWARGS", None)
        args_kwargs["gradient_checkpointing_kwargs"] = gc_kwargs or {"use_reentrant": False}
    else:
        args_kwargs.pop("gradient_checkpointing_kwargs", None)

    if getattr(config, "SAVE_ONLY_MODEL", True):
        args_kwargs["save_only_model"] = True

    return TrainingArguments(**_filtered_training_args(args_kwargs))


def _build_weights(config, variant, label2id, train_ds, val_ds):
    if variant.weight_mode == "uniform":
        return None
    if variant.weight_mode == "o_weight":
        return make_o_weighting(label2id, variant.o_weight or config.O_WEIGHT)
    if variant.weight_mode == "dynamic":
        return make_dynamic_weights_from_datasets(
            label2id,
            datasets=[train_ds, val_ds],
            ignore_index=config.IGNORE_INDEX,
            clamp_min=config.WEIGHT_CLAMP_MIN,
            clamp_max=config.WEIGHT_CLAMP_MAX,
        )
    raise ValueError(f"Unknown weight_mode: {variant.weight_mode}")


def _build_model(config, variant, label2id, id2label):
    num_labels = len(label2id)
    if variant.model_mode == "multi_dropout":
        k = variant.md_k or config.MD_K
        p = variant.md_p or config.MD_P
        return MultiDropoutTokenClassifier.from_pretrained(
            config.MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            k=k,
            dropout_p=p,
            use_safetensors=True,
        )

    return AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )


def train_one_fold(
    fold_index: int,
    config,
    variant,
    status_hook: Optional[StatusHook] = None,
) -> Dict[str, Any]:
    """Train one fold and return evaluation metrics."""
    if status_hook:
        status_hook(fold_index, "running", None)

    try:
        train_ds, val_ds, label2id, id2label, tokenizer = get_fold_datasets(
            fold_index,
            config.NUM_FOLDS,
            config,
        )

        run_dir = Path(config.OUTPUT_DIR_BASE) / config.RUN_NAME / f"fold_{fold_index}"
        model_dir = Path(config.MODEL_DIR_BASE) / config.RUN_NAME / f"fold_{fold_index}"
        _reset_dir(run_dir)
        _reset_dir(model_dir)

        class_weights = _build_weights(config, variant, label2id, train_ds, val_ds)
        model = _build_model(config, variant, label2id, id2label)

        training_args = _build_training_args(config, run_dir, train_len=len(train_ds))

        trainer = WeightedTrainer(
            class_weights=class_weights,
            ignore_index=config.IGNORE_INDEX,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=partial(compute_metrics, id2label=id2label),
        )

        trainer.train()
        metrics = trainer.evaluate()

        trainer.save_model(str(model_dir))
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

        model_info = {
            "run_name": config.RUN_NAME,
            "fold": fold_index,
            "model_mode": variant.model_mode,
            "weight_mode": variant.weight_mode,
            "label2id": label2id,
            "id2label": id2label,
        }
        (model_dir / "model_info.json").write_text(json.dumps(model_info, indent=2))

        (model_dir / "DONE").write_text(f"Fold {fold_index} completed\n")

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        trainer.state.save_to_json(str(run_dir / "trainer_state.json"))

        logits, labels, attention_mask = predict_logits(
            val_ds,
            model=model,
            tokenizer=tokenizer,
            batch_size=config.BATCH_SIZE,
            device=training_args.device,
        )
        save_fold_logits(
            config.RUN_NAME,
            fold_index,
            logits=logits,
            labels=labels,
            attention_mask=attention_mask,
            output_dir_base=config.OUTPUT_DIR_BASE,
        )

        _cleanup_checkpoints(run_dir)

        if status_hook:
            status_hook(fold_index, "success", {"metrics": metrics})

        return metrics
    except Exception as exc:
        if status_hook:
            status_hook(fold_index, "failed", {"error": str(exc)})
        raise
    finally:
        torch.cuda.empty_cache()
