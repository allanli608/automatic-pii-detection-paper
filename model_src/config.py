# Data paths (relative to project root)
DATA_DIR = "data"
TRAIN_JSON = f"{DATA_DIR}/train.json"
TEST_JSON = f"{DATA_DIR}/test.json"   # can be dummy if you don't have one yet

# Model
MODEL_NAME = "microsoft/deberta-v3-base"

# Training hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_FRACTION = 0.1  # fraction of training steps used for warmup

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")