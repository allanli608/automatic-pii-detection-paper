import os
import sys
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model_src.train_baseline import main as run_once


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    seeds = [42, 123, 7]

    for seed in seeds:
        print(f"\n====================")
        print(f"Running experiment with seed = {seed}")
        print(f"====================\n")

        set_global_seed(seed)
        run_once()   # your train_baseline.main() uses the global RNG state


if __name__ == "__main__":
    main()
