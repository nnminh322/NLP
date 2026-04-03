"""Dataset loaders for T²-RAGBench subsets.

Loads FinQA, ConvFinQA, TAT-DQA from the public HuggingFace dataset
"G4KMU/t2-ragbench" and converts them into GSR-CACL core types.

No dependency on g4k — uses datasets + pandas directly.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

from gsr_cacl.core import Document, DatasetSplit
from gsr_cacl.datasets.gsr_document import GSRDocument

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# HuggingFace → Document conversion
# ------------------------------------------------------------------

def _row_to_document(row: dict[str, Any]) -> Document:
    """Convert a single HuggingFace dataset row to a Document."""
    page_content = str(row.get("context", "") or "")
    meta = {
        "company_name": str(row.get("company_name", "")),
        "report_year": str(row.get("report_year", "")),
        "company_sector": str(row.get("company_sector", "")),
    }
    doc_id = str(row.get("id", row.get("_id", "")))
    if not doc_id:
        import uuid
        doc_id = str(uuid.uuid4())
    return Document(page_content=page_content, meta_data=meta, id=doc_id)


def _build_corpus(df: pd.DataFrame) -> list[Document]:
    """Build a de-duplicated corpus of Documents from a DataFrame."""
    seen_ids: set[str] = set()
    corpus: list[Document] = []
    for _, row in df.iterrows():
        doc = _row_to_document(row.to_dict())
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            corpus.append(doc)
    return corpus


def _build_queries(df: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """Extract queries, ground-truth IDs, and per-query metadata."""
    queries: list[str] = []
    gt_ids: list[str] = []
    metas: list[dict] = []
    for _, row in df.iterrows():
        q = str(row.get("question", ""))
        company = str(row.get("company_name", ""))
        if company:
            q = f"{company}: {q}"
        queries.append(q)
        gt_ids.append(str(row.get("id", row.get("_id", ""))))
        metas.append({
            "company_name": str(row.get("company_name", "")),
            "report_year": str(row.get("report_year", "")),
            "company_sector": str(row.get("company_sector", "")),
            "reference_document": {"id": str(row.get("id", row.get("_id", "")))},
        })
    return queries, gt_ids, metas


# ------------------------------------------------------------------
# Public loaders
# ------------------------------------------------------------------

def load_t2ragbench_split(
    config_name: str,
    split: str = "test",
    corpus_splits: list[str] | None = None,
    sample_size: int | None = None,
) -> DatasetSplit:
    """
    Load a T2-RAGBench subset as a DatasetSplit.

    Args:
        config_name: "FinQA", "ConvFinQA", or "TAT-DQA"
        split:       which split for QA queries (default "test")
        corpus_splits: which splits to include in corpus (default: all)
        sample_size: limit QA samples (for debugging)
    Returns:
        DatasetSplit with queries, ground-truth IDs, corpus, metadata
    """
    dataset_name = "G4KMU/t2-ragbench"

    # Load QA split
    logger.info(f"Loading {dataset_name}/{config_name} split={split}")
    qa_df = load_dataset(dataset_name, config_name, split=split).to_pandas()
    if sample_size and sample_size < len(qa_df):
        qa_df = qa_df.sample(n=sample_size, random_state=42)

    # Load corpus (all splits)
    logger.info("Loading full corpus...")
    full = load_dataset(dataset_name, config_name)
    corpus_dfs = []
    if isinstance(full, DatasetDict):
        target_splits = corpus_splits or list(full.keys())
        for s in target_splits:
            if s in full:
                corpus_dfs.append(full[s].to_pandas())
    else:
        corpus_dfs.append(full.to_pandas())
    corpus_df = pd.concat(corpus_dfs, ignore_index=True)

    corpus = _build_corpus(corpus_df)
    queries, gt_ids, metas = _build_queries(qa_df)

    logger.info(f"Loaded {len(queries)} queries, {len(corpus)} corpus documents")
    return DatasetSplit(
        queries=queries,
        ground_truth_ids=gt_ids,
        corpus=corpus,
        meta_data=metas,
        name=config_name,
    )


# ------------------------------------------------------------------
# GSR-enriched corpus builder
# ------------------------------------------------------------------

def build_gsr_corpus(corpus: list[Document]) -> list[GSRDocument]:
    """Pre-compute KG metadata for all documents in a corpus."""
    gsr_docs: list[GSRDocument] = []
    for doc in tqdm(corpus, desc="Building GSR metadata"):
        gsr_docs.append(GSRDocument.from_document(doc))
    return gsr_docs


def get_template_coverage_stats(gsr_corpus: list[GSRDocument]) -> dict[str, Any]:
    """Return template coverage statistics for a GSR-enriched corpus."""
    templates: dict[str, int] = {}
    high_conf = 0
    total = len(gsr_corpus)
    for gsr_doc in gsr_corpus:
        tname = gsr_doc.template_name
        templates[tname] = templates.get(tname, 0) + 1
        if gsr_doc.template_confidence >= 0.7:
            high_conf += 1
    return {
        "total_documents": total,
        "template_distribution": templates,
        "high_confidence_count": high_conf,
        "high_confidence_ratio": high_conf / total if total else 0,
        "avg_confidence": (
            sum(d.template_confidence for d in gsr_corpus) / total if total else 0
        ),
    }
