"""GSR-CACL dataset loading and enrichment."""

from gsr_cacl.datasets.gsr_document import GSRDocument, extract_table
from gsr_cacl.datasets.wrappers import (
    load_t2ragbench_split,
    load_t2ragbench_train_samples,
    build_gsr_corpus,
    get_template_coverage_stats,
)

__all__ = [
    "GSRDocument",
    "extract_table",
    "load_t2ragbench_split",
    "load_t2ragbench_train_samples",
    "build_gsr_corpus",
    "get_template_coverage_stats",
]
