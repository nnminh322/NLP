#!/usr/bin/env python3
"""
GSR-CACL Retrieval Benchmark.

Runs GSR and HybridGSR on T²-RAGBench (§5 of overall_idea.md).
Evaluation metrics: MRR@3, Recall@1/3/5, NDCG@3.

Usage:
    python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa
    python -m gsr_cacl.benchmark_gsr --mode hybridgsr --dataset tatqa
    python -m gsr_cacl.benchmark_gsr --mode gsr --dataset convfinqa
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from gsr_cacl.core import Document, RetrievalResult, DatasetSplit
from gsr_cacl.datasets.wrappers import (
    load_t2ragbench_split,
    build_gsr_corpus,
    get_template_coverage_stats,
)
from gsr_cacl.methods import GSRRetrieval, HybridGSR, GSR_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict] = {
    "finqa": {"config_name": "FinQA", "split": "FinQA"},
    "convfinqa": {"config_name": "ConvFinQA", "split": "ConvFinQA"},
    "tatqa": {"config_name": "TAT-DQA", "split": "TAT-DQA"},
}

# ---------------------------------------------------------------------------
# Method configs (GSR §4 hyperparameters)
# ---------------------------------------------------------------------------

METHOD_CONFIGS: dict[str, dict] = {
    "gsr": {
        "class": GSRRetrieval,
        "alpha": 0.5,
        "beta": 0.3,
        "gamma": 0.2,
        "gat_hidden_dim": 256,
        "gat_num_heads": 4,
        "gat_num_layers": 2,
    },
    "hybridgsr": {
        "class": HybridGSR,
        "alpha": 0.4,
        "beta": 0.3,
        "gamma": 0.2,
        "gat_hidden_dim": 256,
        "gat_num_heads": 4,
        "gat_num_layers": 2,
        "rrf_k": 60,
    },
}


# ---------------------------------------------------------------------------
# Metrics (§5.1)
# ---------------------------------------------------------------------------

def compute_mrr(results: list[RetrievalResult], k: int = 3) -> float:
    """Mean Reciprocal Rank at K."""
    mrr = 0.0
    for r in results:
        for rank, doc in enumerate(r.retrieved_docs[:k], start=1):
            if str(doc.id) == str(r.ground_truth_id):
                mrr += 1.0 / rank
                break
    return mrr / len(results) if results else 0.0


def compute_recall(results: list[RetrievalResult], k: int = 3) -> float:
    """Recall at K."""
    hits = 0
    for r in results:
        ids = {str(d.id) for d in r.retrieved_docs[:k]}
        if str(r.ground_truth_id) in ids:
            hits += 1
    return hits / len(results) if results else 0.0


def compute_ndcg(results: list[RetrievalResult], k: int = 3) -> float:
    """NDCG at K (binary relevance)."""
    total = 0.0
    for r in results:
        for rank, doc in enumerate(r.retrieved_docs[:k], start=1):
            if str(doc.id) == str(r.ground_truth_id):
                total += 1.0 / math.log2(rank + 1)
                break
    return total / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_gsr_benchmark(
    mode: str,
    dataset: str,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
    top_k: int = 3,
    device: str | None = None,
    output_dir: Path | None = None,
    sample_size: int | None = None,
) -> dict:
    """
    Run GSR retrieval benchmark on a T²-RAGBench subset.

    Returns dict of metric results.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Resolve configs
    ds_cfg = DATASET_CONFIGS.get(dataset)
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_CONFIGS)}")
    method_cfg = dict(METHOD_CONFIGS.get(mode, {}))
    if not method_cfg:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(METHOD_CONFIGS)}")

    # Load dataset
    split_data = load_t2ragbench_split(
        config_name=ds_cfg["config_name"],
        split=ds_cfg["split"],
        sample_size=sample_size,
    )
    logger.info(f"Queries: {len(split_data.queries)}, Corpus: {len(split_data.corpus)}")

    # Template coverage diagnostics (§5.3 Exp 5)
    gsr_corpus = build_gsr_corpus(split_data.corpus)
    stats = get_template_coverage_stats(gsr_corpus)
    logger.info("Template coverage stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Embedding model
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
    )

    # Build retrieval method
    RAGClass = method_cfg.pop("class")
    rag_method = RAGClass(
        corpus=split_data.corpus,
        embedding_function=embeddings,
        top_k=top_k,
        device=device,
        **method_cfg,
    )

    # Run retrieval
    logger.info(f"Running {mode} retrieval for {len(split_data.queries)} queries...")
    all_retrieved = rag_method.retrieve_batch(
        queries=split_data.queries,
        queries_meta=split_data.meta_data,
    )

    # Build evaluation results
    eval_results: list[RetrievalResult] = []
    for i, (query, docs) in enumerate(zip(split_data.queries, all_retrieved)):
        eval_results.append(RetrievalResult(
            query=query,
            retrieved_docs=docs,
            ground_truth_id=split_data.ground_truth_ids[i],
            meta_data=split_data.meta_data[i],
        ))

    # Compute metrics
    metrics = {
        "MRR@3": compute_mrr(eval_results, k=3),
        "Recall@1": compute_recall(eval_results, k=1),
        "Recall@3": compute_recall(eval_results, k=3),
        "Recall@5": compute_recall(eval_results, k=5),
        "NDCG@3": compute_ndcg(eval_results, k=3),
    }

    # Print results
    print("\n" + "=" * 60)
    print(f"GSR-CACL BENCHMARK  [{mode.upper()} on {dataset.upper()}]")
    print("=" * 60)
    for name, val in metrics.items():
        print(f"  {name:>12}: {val:.4f}")
    print("=" * 60 + "\n")

    # Save outputs
    if output_dir is None:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        output_dir = Path(f"outputs/gsr_benchmark/{dataset}/{mode}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSR-CACL Retrieval Benchmark on T²-RAGBench",
    )
    parser.add_argument("--mode", type=str, required=True, choices=["gsr", "hybridgsr"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["finqa", "convfinqa", "tatqa"])
    parser.add_argument("--embedding", type=str,
                        default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit number of QA samples (for debugging)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    try:
        run_gsr_benchmark(
            mode=args.mode,
            dataset=args.dataset,
            embedding_model=args.embedding,
            top_k=args.top_k,
            device=args.device,
            output_dir=output_dir,
            sample_size=args.sample,
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
