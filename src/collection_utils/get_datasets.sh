#!/bin/bash

# This script downloads necessary datasets for the project.
# 1. Please ensure you all dependencies are installed by running: pip install -r requirements.txt
# 2. Ensure OpenRouter API key is set in your environment variables or .env file (OPENROUTER_API_KEY = "your_api_key_here").
# 3. Run this script: bash data_utils/get_datasets.sh

SYNTHETIC_NUM_SAMPLES=10

python data_utils/data_collector.py --download_all
python data_utils/synthetic_data_generator.py --num_samples $SYNTHETIC_NUM_SAMPLES

