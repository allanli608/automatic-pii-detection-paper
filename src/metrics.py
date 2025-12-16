import torch
import numpy as np
from typing import Dict

def compute_metrics(eval_pred, id2label: Dict[int, str]):
    """
    Computes strict F5 score. Misclassifying a PII type counts as FP and FN.
    Expects eval_pred to be (logits, labels).
    """
    predictions, labels = eval_pred
    
    # Convert logits to class IDs
    preds = np.argmax(predictions, axis=-1)
    
    # Flatten and mask ignored indices (-100)
    mask = labels != -100
    valid_preds = preds[mask]
    valid_labels = labels[mask]

    tp = 0
    fp = 0
    fn = 0

    # Calculate strict metrics
    for p_id, t_id in zip(valid_preds, valid_labels):
        p_str = id2label[p_id]
        t_str = id2label[t_id]

        if t_str != "O":
            if p_str == t_str:
                tp += 1
            else:
                fn += 1 # Missed the specific PII type
        
        if p_str != "O":
            if p_str != t_str:
                fp += 1 # Predicted PII but it was wrong type or O

    epsilon = 1e-12
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    beta = 5.0
    f5 = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + epsilon)

    return {
        "fbeta": f5,
        "precision": precision,
        "recall": recall,
    }