from typing import Dict, List

import numpy as np
import torch


def is_pii_label(label_str: str) -> bool:
    """Treat everything except 'O' as PII."""
    return label_str != "O"


def compute_p_r_f5_from_batches(
    all_logits: List[torch.Tensor],
    all_labels: List[torch.Tensor],
    id2label: Dict[int, str],
) -> Dict[str, float]:
    """
    all_logits: list of (batch_size, seq_len, num_labels) tensors
    all_labels: list of (batch_size, seq_len) tensors with -100 for ignored
    id2label: mapping from label id to label string
    """
    # Concatenate across all batches
    logits = torch.cat(all_logits, dim=0)       # (N, L, C)
    labels = torch.cat(all_labels, dim=0)       # (N, L)

    # argmax over label dimension
    preds = torch.argmax(logits, dim=-1)        # (N, L)

    preds = preds.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    true_labels = []
    pred_labels = []

    for p_id, l_id in zip(preds, labels):
        if l_id == -100:
            continue
        true_labels.append(id2label[int(l_id)])
        pred_labels.append(id2label[int(p_id)])

    # micro P/R/F over "PII" vs "not PII (O)"
    tp = sum(is_pii_label(t) and is_pii_label(p) for t, p in zip(true_labels, pred_labels))
    fp = sum((not is_pii_label(t)) and is_pii_label(p) for t, p in zip(true_labels, pred_labels))
    fn = sum(is_pii_label(t) and (not is_pii_label(p)) for t, p in zip(true_labels, pred_labels))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    beta = 5.0
    f5 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)

    return {
        "precision_pii": precision,
        "recall_pii": recall,
        "f5_pii": f5,
    }
