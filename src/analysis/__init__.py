"""Analysis utilities for runs and ensembles."""

from src.analysis.collect_runs import collect_runs
from src.analysis.plots import plot_metric_curves, plot_variant_comparison, plot_confidence_histogram
from src.analysis.confusion import build_confusion_matrix
from src.analysis.compare_variants import compare_variants

__all__ = [
    "collect_runs",
    "plot_metric_curves",
    "plot_variant_comparison",
    "plot_confidence_histogram",
    "build_confusion_matrix",
    "compare_variants",
]
