import os
import torch

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

COMPETITION_DATA = os.path.join(DATA_DIR, "default_dataset/train.json") 
EXTERNAL_DATA = os.path.join(DATA_DIR, "synthetic_dataset/train_clean.json")

# Model & Training
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR_BASE = "output"

NUM_FOLDS = 4
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# multi-dropout (off by default)
MD_K = 1      # 1 = disabled (single pass)
MD_P = 0.2    # dropout prob used in the multi-dropout head


NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP16 = True