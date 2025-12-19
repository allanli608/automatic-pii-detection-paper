import numpy as np
import os
from model import MetaClassifier, XGBoostMetaModel

# Global variable to cache the model so we don't reload it for every single batch
_LOADED_MODEL = None


def get_model(model_path: str = "meta_classifier.pkl") -> MetaClassifier:
    """
    Singleton-like accessor to load the model only once per process.
    """
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Did you run train_metaclassifier.py?"
            )

        # We use the static load method defined in your Abstract Base Class
        _LOADED_MODEL = XGBoostMetaModel.load(model_path)
        print(f"Meta-classifier loaded successfully from {model_path}")

    return _LOADED_MODEL


def run_meta_inference(
    ensemble_features: np.ndarray, model_path: str = "meta_classifier.pkl"
) -> np.ndarray:
    """
    Takes the raw outputs from the ensemble of base models and generates
    the final calibrated probabilities.

    Args:
        ensemble_features (np.ndarray): A 2D numpy array of shape (Batch_Size, Input_Features).
                                        Batch_Size can be 1 (single token) or N (full text).
                                        Input_Features must match training (e.g., 7 models * 13 classes = 91).
        model_path (str): Path to the pickled model file.

    Returns:
        np.ndarray: A 2D numpy array of shape (Batch_Size, Num_Classes) containing
                    probabilities summing to 1.0 for each token.
    """
    # 1. Input Validation
    # Ensure input is 2D (even if it's just one token)
    if ensemble_features.ndim == 1:
        ensemble_features = ensemble_features.reshape(1, -1)

    # 2. Model Retrieval
    model = get_model(model_path)

    # 3. Validation of Feature Count
    # XGBoost saves the expected number of features. We can check this to prevent silent failures.
    # (Note: This specific check depends on XGBoost version, but standard practice is to rely on the wrapper)
    # If using the wrapper from model.py, we rely on predict_proba to handle internal checks.

    # 4. Inference
    try:
        probabilities = model.predict_proba(ensemble_features)
    except Exception as e:
        raise RuntimeError(
            f"Inference failed. Check if your input shape {ensemble_features.shape} matches the trained model."
        ) from e

    return probabilities


# Example usage for testing
if __name__ == "__main__":
    # Mock input: 1 token, 7 models * 13 classes = 91 features
    mock_input = np.random.rand(1, 91).astype(np.float32)

    try:
        result = run_meta_inference(mock_input)
        print("Inference successful.")
        print(f"Input Shape: {mock_input.shape}")
        print(f"Output Shape: {result.shape}")  # Should be (1, 13)
        print(f"Output Probabilities: {result}")
    except FileNotFoundError as e:
        print(e)
