import numpy as np
import pandas as pd
import re
from typing import List, Tuple
from model import XGBoostMetaModel
from sklearn.metrics import log_loss, accuracy_score


def load_data(
    df: pd.DataFrame, model_names: List[str], num_classes: int = 13
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms a DataFrame containing base model predictions into X and y for training.

    Args:
        df: Pandas DataFrame containing the OOF predictions.
        model_names: List of strings identifying the 7 model variants (e.g. ['deberta_v3', 'roberta']).
        num_classes: Number of PII classes (default 13).

    Returns:
        X: Feature matrix of shape (N_samples, N_models * N_classes)
        y: Label vector of shape (N_samples,)
    """

    feature_cols = []

    # 1. Enforce Column Ordering
    # We iterate explicitly to guarantee the feature vector is always built
    # as: [Model1_Class0...Model1_Class12, Model2_Class0...Model2_Class12, ...]
    print("Validating input dataframe structure...")

    for model in model_names:
        for c_idx in range(num_classes):
            col_name = f"pred_{model}_{c_idx}"

            # Defensive Programming: Check if column exists
            if col_name not in df.columns:
                raise ValueError(
                    f"Missing expected column: {col_name}. Check your dataframe generation."
                )

            feature_cols.append(col_name)

    print(f"Constructed feature list with {len(feature_cols)} features.")

    # 2. Extract X (Features)
    # This selects all columns in the exact order of 'feature_cols'
    X = df[feature_cols].values.astype(np.float32)

    # 3. Extract y (Labels)
    if "label" not in df.columns:
        raise ValueError("DataFrame is missing the 'label' target column.")

    y = df["label"].values.astype(np.int32)

    return X, y


def main():
    # --- Configuration ---
    # These must match exactly how you named your columns in the CSV
    MODEL_VARIANTS = [
        "deberta_v3_large",
        "deberta_v3_base",
        "distilroberta",
        "custom_model_23",
        "exp073",
        "custom_model_bilstm",
        "basic_model_2",
    ]

    # 1. Load Data (Simulating a CSV load here)
    # In reality: df = pd.read_csv("oof_inference_results.csv")
    print("Loading data...")

    # MOCK DATA CREATION (Remove this block when using real data)
    # ---------------------------------------------------------
    N_SAMPLES = 1000
    mock_data = {"label": np.random.randint(0, 13, N_SAMPLES)}
    for m in MODEL_VARIANTS:
        for c in range(13):
            # Simulate random probabilities
            mock_data[f"pred_{m}_{c}"] = np.random.rand(N_SAMPLES)
    df = pd.DataFrame(mock_data)
    # ---------------------------------------------------------

    # Use the adapter function
    X, y = load_data(df, MODEL_VARIANTS)

    print(f"Data Loaded. X shape: {X.shape}, y shape: {y.shape}")

    # 2. Initialize Model
    model = XGBoostMetaModel()

    # 3. Train
    print("Training meta-classifier...")
    model.fit(X, y)

    # 4. Validate
    preds = model.predict_proba(X)
    acc = accuracy_score(y, np.argmax(preds, axis=1))
    print(f"Training Accuracy: {acc:.4f}")

    # 5. Save
    model.save("meta_classifier.pkl")


if __name__ == "__main__":
    main()
