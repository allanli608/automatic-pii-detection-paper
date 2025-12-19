import abc
import pickle
import xgboost as xgb
import numpy as np


# 1. The Contract (Abstract Base Class)
class MetaClassifier(abc.ABC):
    """
    Abstract base class to ensure all meta-classifiers (XGBoost, MLP, etc.)
    follow the same structure.
    """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model.
        X: Input features (Shape: [N_samples, N_base_models * N_classes])
        y: Target labels (Shape: [N_samples])
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probabilities for each class.
        Output Shape: [N_samples, N_classes]
        """
        pass

    @abc.abstractmethod
    def save(self, filepath: str):
        """Saves the trained model to disk."""
        pass

    @staticmethod
    @abc.abstractmethod
    def load(filepath: str):
        """Loads a model from disk."""
        pass


# 2. The Implementation (XGBoost)
class XGBoostMetaModel(MetaClassifier):
    def __init__(self, params=None):
        # Default parameters for multi-class classification
        self.params = (
            params
            if params
            else {
                "objective": "multi:softprob",  # Essential for outputting probabilities
                "num_class": 13,  # Total PII classes including 'O'
                "eval_metric": "mlogloss",
                "tree_method": "hist",  # Faster training
            }
        )
        self.model = None

    def fit(self, X, y):
        # XGBoost requires DMatrix format
        dtrain = xgb.DMatrix(X, label=y)
        # Train the model (num_boost_round is effectively 'epochs')
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)

    def predict_proba(self, X):
        if not self.model:
            raise Exception("Model not trained yet!")
        dtest = xgb.DMatrix(X)
        # Returns [N_samples, 13]
        return self.model.predict(dtest)

    def save(self, filepath):
        # We pickling the wrapper class to keep the params and model together
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
