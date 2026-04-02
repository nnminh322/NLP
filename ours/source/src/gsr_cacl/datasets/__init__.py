"""GSR-CACL dataset wrappers for T²-RAGBench.

Wraps the g4k dataset classes and enriches documents with:
  - Constraint KG pre-computation
  - CHAP negative metadata
  - Template matching diagnostics

Supports: FinQA, ConvFinQA, TAT-DQA (T²-RAGBench subsets).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from g4k.internal.abstractions import (
    BatchInferenceRunner,
    ResponseWrapper,
    Document,
    MetaData,
    RAGMethodInterface,
)
from g4k.datasets.finqa.finqa_qa import FinQADatasetCollection
from g4k.datasets.tatqa.tatdqa_qa import TatQADatasetCollection

from gsr_cacl.kg import (
    ConstraintKG,
    build_constraint_kg,
    build_kg_from_markdown,
)
from gsr_cacl.negative_sampler import CHAPNegativeSampler


# ----------------------------------------------------------------------
# GSR-enriched document
# ----------------------------------------------------------------------

@dataclass
class GSRDocument:
    """
    Extended document with GSR-specific metadata.
    """
    base: Document
    kg: ConstraintKG
    template_name: str = ""
    template_confidence: float = 0.0
    n_constraint_edges: int = 0
    n_positional_edges: int = 0
    n_cells: int = 0

    @classmethod
    def from_document(cls, doc: Document) -> "GSRDocument":
        """Build GSRDocument from a g4k Document."""
        table_md = _extract_table(doc.page_content)
        kg = build_constraint_kg(table_md, epsilon=1e-4)

        return cls(
            base=doc,
            kg=kg,
            template_name=kg.template.name if kg.template else "none",
            template_confidence=kg.template_confidence,
            n_constraint_edges=len([e for e in kg.edges if e.edge_type == "accounting"]),
            n_positional_edges=len([e for e in kg.edges if e.edge_type == "positional"]),
            n_cells=len(kg.nodes),
        )


# ----------------------------------------------------------------------
# Table extraction helper
# ----------------------------------------------------------------------

def _extract_table(content: str) -> str:
    """
    Extract the first markdown table from a document's page_content.
    Returns empty string if no table found.
    """
    lines = content.split("\n")
    table_lines = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if "|" in stripped:
            in_table = True
            table_lines.append(stripped)
        elif in_table:
            break
    if len(table_lines) < 2:
        return ""
    return "\n".join(table_lines)


# ----------------------------------------------------------------------
# GSR-enriched FinQA wrapper
# ----------------------------------------------------------------------

class GSRFinQADataset(FinQADatasetCollection):
    """
    GSR-enriched FinQA dataset.

    Extends FinQADatasetCollection to pre-compute:
      - Constraint KGs for all corpus documents
      - Template matching diagnostics
      - CHAP negative pre-generation for training
    """

    def __init__(
        self,
        df: pd.DataFrame,
        retrieval_query: str = "",
        meta_data_keys: list[str] = [],
        document_percentage: float = 1.0,
        corpus_df: Optional[pd.DataFrame] = None,
        build_gsr_metadata: bool = True,
    ) -> None:
        super().__init__(
            df=df,
            retrieval_query=retrieval_query,
            meta_data_keys=meta_data_keys,
            document_percentage=document_percentage,
            corpus_df=corpus_df,
        )

        if build_gsr_metadata:
            self._build_gsr_metadata()

    def _build_gsr_metadata(self) -> None:
        """
        Pre-compute GSR metadata for all corpus documents.
        Stores template name, confidence, and edge counts.
        """
        self.gsr_corpus: list[GSRDocument] = []
        for doc in self.context_collection:
            gsr_doc = GSRDocument.from_document(doc)
            self.gsr_corpus.append(gsr_doc)

    def get_gsr_corpus(self) -> list[GSRDocument]:
        """Return GSR-enriched corpus."""
        return getattr(self, "gsr_corpus", [])

    def get_template_coverage_stats(self) -> dict[str, Any]:
        """
        Return template coverage statistics for the corpus.
        Useful for validating §2.2 template coverage claims.
        """
        if not hasattr(self, "gsr_corpus"):
            return {"error": "GSR metadata not built"}

        templates: dict[str, int] = {}
        high_conf = 0   # confidence >= 0.7
        total = len(self.gsr_corpus)

        for gsr_doc in self.gsr_corpus:
            tname = gsr_doc.template_name
            templates[tname] = templates.get(tname, 0) + 1
            if gsr_doc.template_confidence >= 0.7:
                high_conf += 1

        return {
            "total_documents": total,
            "template_distribution": templates,
            "high_confidence_count": high_conf,
            "high_confidence_ratio": high_conf / total if total else 0,
            "avg_confidence": sum(d.template_confidence for d in self.gsr_corpus) / total if total else 0,
        }


# ----------------------------------------------------------------------
# GSR-enriched TAT-DQA wrapper
# ----------------------------------------------------------------------

class GSRTatQADataset(TatQADatasetCollection):
    """
    GSR-enriched TAT-DQA dataset.

    TAT-DQA has the most diverse tables (~70% template coverage estimate).
    This wrapper provides the same GSR metadata enrichment.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        retrieval_query: str = "",
        meta_data_keys: list[str] = [],
        document_percentage: float = 1.0,
        corpus_df: Optional[pd.DataFrame] = None,
        build_gsr_metadata: bool = True,
    ) -> None:
        super().__init__(
            df=df,
            retrieval_query=retrieval_query,
            meta_data_keys=meta_data_keys,
            document_percentage=document_percentage,
            corpus_df=corpus_df,
        )

        if build_gsr_metadata:
            self._build_gsr_metadata()

    def _build_gsr_metadata(self) -> None:
        """Pre-compute GSR metadata."""
        self.gsr_corpus: list[GSRDocument] = []
        for doc in self.context_collection:
            gsr_doc = GSRDocument.from_document(doc)
            self.gsr_corpus.append(gsr_doc)

    def get_gsr_corpus(self) -> list[GSRDocument]:
        return getattr(self, "gsr_corpus", [])

    def get_template_coverage_stats(self) -> dict[str, Any]:
        if not hasattr(self, "gsr_corpus"):
            return {"error": "GSR metadata not built"}

        templates: dict[str, int] = {}
        high_conf = 0
        total = len(self.gsr_corpus)

        for gsr_doc in self.gsr_corpus:
            tname = gsr_doc.template_name
            templates[tname] = templates.get(tname, 0) + 1
            if gsr_doc.template_confidence >= 0.7:
                high_conf += 1

        return {
            "total_documents": total,
            "template_distribution": templates,
            "high_confidence_count": high_conf,
            "high_confidence_ratio": high_conf / total if total else 0,
            "avg_confidence": sum(d.template_confidence for d in self.gsr_corpus) / total if total else 0,
        }


# ----------------------------------------------------------------------
# Cross-dataset convenience
# ----------------------------------------------------------------------

GSR_DATASET_WRAPPERS: dict[str, type] = {
    "FinQADatasetCollection": GSRFinQADataset,
    "TatQADatasetCollection": GSRTatQADataset,
    "ConvFinQADataset": GSRFinQADataset,   # inherits from FinQA
}


def wrap_dataset(dataset_obj) -> Any:
    """
    Wrap a g4k dataset object with GSR metadata.
    Returns the same object if no wrapper available.
    """
    cls_name = type(dataset_obj).__name__
    wrapper_cls = GSR_DATASET_WRAPPERS.get(cls_name)
    if wrapper_cls is not None:
        # Re-instantiate with same parameters
        return wrapper_cls(
            df=dataset_obj.get_data_frame(),
            retrieval_query=dataset_obj.retrieval_query,
            meta_data_keys=dataset_obj.meta_data_keys,
            document_percentage=dataset_obj.document_percentage,
            build_gsr_metadata=True,
        )
    return dataset_obj
