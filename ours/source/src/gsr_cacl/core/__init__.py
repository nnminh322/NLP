"""Core data structures for GSR-CACL.

Replaces g4k.internal.abstractions with lightweight, self-contained types.
These are the only shared types across the entire GSR-CACL codebase.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single document in the corpus (text + table + metadata).

    Matches the T²-RAGBench schema:
      - page_content: concatenated text narrative + markdown table
      - meta_data:    dict with keys like company_name, report_year, company_sector
      - id:           unique identifier (auto-generated if not provided)
    """
    page_content: str
    meta_data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __repr__(self) -> str:
        preview = self.page_content[:80].replace("\n", " ")
        return f"Document(id={self.id!r}, content={preview!r}...)"


@dataclass
class RetrievalResult:
    """Result for a single query evaluation.

    Used by the benchmark to compute MRR, Recall, NDCG.
    """
    query: str
    retrieved_docs: list[Document]
    ground_truth_id: str = ""
    meta_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplit:
    """A loaded dataset split containing QA pairs and the document corpus.

    Attributes:
        queries:     list of query strings
        ground_truth_ids:  list of ground-truth document IDs (parallel to queries)
        corpus:      list of all Document objects in the corpus
        meta_data:   list of per-query metadata dicts (parallel to queries)
        name:        human-readable name (e.g. "FinQA", "TAT-DQA")
    """
    queries: list[str]
    ground_truth_ids: list[str]
    corpus: list[Document]
    meta_data: list[dict[str, Any]]
    name: str = ""
