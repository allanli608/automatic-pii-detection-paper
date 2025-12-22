"""Backward-compatible re-export for data loading utilities."""

from src.data.data_loader import PIIDataset, get_fold_datasets

__all__ = ["PIIDataset", "get_fold_datasets"]
