"""Base configuration for training and inference."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

COMPETITION_DATA = str(DATA_DIR / "default_dataset" / "train.json")
EXTERNAL_DATA = str(DATA_DIR / "synthetic_dataset" / "train_clean.json")

MODEL_NAME = "microsoft/deberta-v3-base"

OUTPUT_DIR_BASE = "outputs"
MODEL_DIR_BASE = "models"

RUN_NAME = "baseline"
SEED = 42

NUM_FOLDS = 4
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

MAX_LENGTH = 1024
EXTERNAL_FRACTION = 1.0

MD_K = 1
MD_P = 0.2
O_WEIGHT = 0.05

NUM_WORKERS = 4
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = False
MAX_GRAD_NORM = 1.0
EVAL_ACCUMULATION_STEPS = None

FP16 = True
BF16 = False
TORCH_COMPILE = False

IGNORE_INDEX = -100

EVAL_STRATEGY = "epoch"
METRIC_FOR_BEST_MODEL = "fbeta"
REPORT_TO = "none"
LOGGING_STEPS = 50

SAVE_TOTAL_LIMIT = 1
SAVE_BEST_ONLY = True
SAVE_EVERY_N_EPOCHS = None
SAVE_ONLY_MODEL = True

WEIGHT_CLAMP_MIN = 0.1
WEIGHT_CLAMP_MAX = 10.0
