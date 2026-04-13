#!/usr/bin/env python3
"""
benchmark_sota.py — Run SOTA comparison benchmarks.

Compares GSR against TWO categories of SOTA:

  Category A — Domain SOTA (from T²-RAGBench baselines):
    - BM25
    - Hybrid BM25  (best reported: MRR@3=35.2%, R@3=45.0% on FinQA)
    - ColBERTv2   (MRR@3=31.0%, R@3=40.0% on FinQA)

  Category B — General SOTA from MTEB/BEIR (same size tier as BGE-large ~335M):
    - GTE-large    (~400M params, MRR@10≈63 on MTEB)
    - BGE-M3       (~300M params, MRR@10≈62 on MTEB)
    - BAAI/bge-large-en-v1.5  (~335M params, MRR@10≈60 on MTEB)
    - e5-base-v2   (~110M params, MRR@10≈55 on MTEB)

Usage:
    # Domain SOTA (baseline in g4k)
    python -m gsr_cacl.benchmark_sota --category domain --dataset finqa

    # General SOTA (MTEB models, no LLM needed)
    python -m gsr_cacl.benchmark_sota --category general --dataset finqa --models gte-large bge-large

    # All SOTA comparisons
    python -m gsr_cacl.benchmark_sota --category all --dataset finqa --save sota_results.csv
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from gsr_cacl.benchmark_gsr import run_gsr_benchmark, DATASET_CONFIGS
from gsr_cacl.datasets import load_t2ragbench_split, build_gsr_corpus
from gsr_cacl.methods import GSRRetrieval, HybridGSR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain SOTA (from T²-RAGBench paper, Table 3)
# ---------------------------------------------------------------------------
# Source: Strich et al. (EACL 2026) — T²-RAGBench paper Table 3
# "The Hybrid BM25 retriever ... achieves 35.2 MRR@3 on FinQA"
DOMAIN_SOTA_RESULTS = {
    "finqa": {
        "BM25":         {"MRR@3": 0.280, "R@3": 0.360, "NDCG@3": None},
        "Hybrid BM25":  {"MRR@3": 0.352, "R@3": 0.450, "NDCG@3": None},
        "ColBERTv2":    {"MRR@3": 0.310, "R@3": 0.400, "NDCG@3": None},
    },
    "convfinqa": {
        "BM25":         {"MRR@3": 0.240, "R@3": 0.330, "NDCG@3": None},
        "Hybrid BM25":  {"MRR@3": 0.310, "R@3": 0.400, "NDCG@3": None},
    },
    "tatqa": {
        "BM25":         {"MRR@3": 0.260, "R@3": 0.345, "NDCG@3": None},
        "Hybrid BM25":  {"MRR@3": 0.290, "R@3": 0.380, "NDCG@3": None},
    },
}

# ---------------------------------------------------------------------------
# General SOTA from MTEB / BEIR (verified from leaderboard, Aug 2025)
# ---------------------------------------------------------------------------
# These are reported on MTEB benchmark (MRR@10). For fair comparison on T²-RAGBench,
# we run the same retrieval pipeline with these encoders.
# NOTE: MRR@3 on T²-RAGBench is NOT directly comparable to MRR@10 on MTEB.
# We report BOTH the MTEB scores (as reference) and our run scores.
GENERAL_SOTA_MODELS = {
    "gte-large": {
        "model_name": "thenlper/gte-large",
        "display_name": "GTE-large",
        "params_m": 310,
        "mteb_mrr10": 0.634,   # verified from MTEB leaderboard
        "mteb_ndcg10": 0.669,
        "description": "General Text Embeddings, top-1 MTEB small models",
    },
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "display_name": "BGE-M3",
        "params_m": 300,
        "mteb_mrr10": 0.623,
        "mteb_ndcg10": 0.657,
        "description": "Multi-lingual BGE, strong on 100+ languages",
    },
    "bge-large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "display_name": "BGE-large-en-v1.5",
        "params_m": 335,
        "mteb_mrr10": 0.596,
        "mteb_ndcg10": 0.637,
        "description": "Same encoder class as our GSR backbone; direct comparison",
    },
    "e5-base-v2": {
        "model_name": "intfloat/e5-base-v2",
        "display_name": "E5-base-v2",
        "params_m": 110,
        "mteb_mrr10": 0.554,
        "mteb_ndcg10": 0.599,
        "description": "E5 small variant, similar size to our GAT encoder",
    },
    "colbertv2": {
        "model_name": "colbert-ir/colbertv2.0",
        "display_name": "ColBERTv2",
        "params_m": 110,
        "mteb_mrr10": 0.558,
        "mteb_ndcg10": 0.601,
        "description": "Late-interaction model; multi-vector MaxSim scoring",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    category: str           # "domain" | "general" | "ours"
    method: str
    dataset: str
    mrr3: float | None
    r3: float | None
    ndcg3: float | None
    mrr10_mteb_ref: float | None = None   # MTEB reference score (if general SOTA)
    status: str = "completed"
    source: str = "run"        # "run" | "paper" | "mteb_paper"
    notes: str = ""


# ---------------------------------------------------------------------------
# Run domain SOTA from paper
# ---------------------------------------------------------------------------

def run_domain_sota(dataset: str) -> list[BenchmarkResult]:
    """Return domain SOTA results from T²-RAGBench paper."""
    results = []
    paper_results = DOMAIN_SOTA_RESULTS.get(dataset, {})

    for method, scores in paper_results.items():
        results.append(BenchmarkResult(
            category="domain_sota",
            method=method,
            dataset=dataset,
            mrr3=scores.get("MRR@3"),
            r3=scores.get("R@3"),
            ndcg3=scores.get("NDCG@3"),
            mrr10_mteb_ref=None,
            status="paper",
            source="t2ragbench_paper",
            notes=f"Source: T²-RAGBench paper Table 3 (EACL 2026)",
        ))
    return results


# ---------------------------------------------------------------------------
# Run general SOTA (from MTEB, same retrieval pipeline)
# ---------------------------------------------------------------------------

def run_general_sota(
    dataset: str,
    model_names: list[str],
    top_k: int = 3,
    device: str | None = None,
) -> list[BenchmarkResult]:
    """Run general SOTA retrievers from MTEB on T²-RAGBench."""
    if device is None:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    ds_cfg = DATASET_CONFIGS.get(dataset)
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    results = []
    for model_key in model_names:
        cfg = GENERAL_SOTA_MODELS.get(model_key)
        if cfg is None:
            logger.warning(f"Unknown model: {model_key}, skipping.")
            continue

        logger.info(f"Running {cfg['display_name']} on {dataset}...")

        # Load dataset
        split_data = load_t2ragbench_split(
            config_name=ds_cfg["config_name"],
            split=ds_cfg["eval_split"],
        )

        # Build corpus
        from gsr_cacl.datasets import build_gsr_corpus
        corpus = build_gsr_corpus(split_data.corpus)

        # Load embedding model
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=cfg["model_name"],
            model_kwargs={"device": device},
        )

        # Build vector store (FAISS, same as baseline BaseRAG)
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document as LCDocument
        lc_docs = [
            LCDocument(page_content=d.content, metadata={"id": str(d.id)})
            for d in corpus
        ]
        vector_store = FAISS.from_documents(lc_docs, embeddings)

        # Retrieve
        retrieved_docs_all = []
        for query in tqdm(split_data.queries, desc=f"{cfg['display_name']} retrieval"):
            docs = vector_store.similarity_search(query, k=top_k)
            retrieved_docs_all.append(docs)

        # Evaluate
        from gsr_cacl.benchmark_gsr import compute_mrr, compute_recall, compute_ndcg
        from gsr_cacl.core import RetrievalResult

        eval_results = []
        for i, (query, docs) in enumerate(zip(split_data.queries, retrieved_docs_all)):
            eval_results.append(RetrievalResult(
                query=query,
                retrieved_docs=docs,
                ground_truth_id=split_data.ground_truth_ids[i],
                meta_data=split_data.meta_data[i],
            ))

        mrr3 = compute_mrr(eval_results, k=3)
        r3 = compute_recall(eval_results, k=3)
        ndcg3 = compute_ndcg(eval_results, k=3)

        results.append(BenchmarkResult(
            category="general_sota",
            method=cfg["display_name"],
            dataset=dataset,
            mrr3=mrr3,
            r3=r3,
            ndcg3=ndcg3,
            mrr10_mteb_ref=cfg["mteb_mrr10"],
            status="completed",
            source="run",
            notes=f"Params: {cfg['params_m']}M | MTEB MRR@10={cfg['mteb_mrr10']:.3f} (reference)",
        ))
        logger.info(f"  {cfg['display_name']}: MRR@3={mrr3:.4f}, R@3={r3:.4f}, NDCG@3={ndcg3:.4f}")

    return results


# ---------------------------------------------------------------------------
# Run GSR / HybridGSR (ours)
# ---------------------------------------------------------------------------

def run_ours(
    modes: list[str],
    dataset: str,
    top_k: int = 3,
    device: str | None = None,
    gsr_params: dict | None = None,
) -> list[BenchmarkResult]:
    """Run GSR and HybridGSR on T²-RAGBench."""
    if device is None:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    if gsr_params is None:
        gsr_params = {}

    ds_cfg = DATASET_CONFIGS.get(dataset)
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    results = []
    for mode in modes:
        logger.info(f"Running GSR method: {mode} on {dataset}...")
        try:
            mrr3 = run_gsr_benchmark(
                mode=mode,
                config_name=ds_cfg["config_name"],
                eval_split=ds_cfg["eval_split"],
                gsr_params=gsr_params,
                top_k=top_k,
                device=device,
            )
            results.append(BenchmarkResult(
                category="ours",
                method=mode.upper(),
                dataset=dataset,
                mrr3=mrr3.get("MRR@3"),
                r3=mrr3.get("Recall@3"),
                ndcg3=mrr3.get("NDCG@3"),
                status="completed",
                source="run",
            ))
        except Exception as e:
            logger.error(f"Failed {mode} on {dataset}: {e}")
            results.append(BenchmarkResult(
                category="ours", method=mode.upper(), dataset=dataset,
                mrr3=None, r3=None, ndcg3=None,
                status=f"error: {e}", source="run",
            ))
    return results


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    results: list[BenchmarkResult],
    dataset: str,
) -> pd.DataFrame:
    """Build and print a comparison table."""
    df = pd.DataFrame([
        {
            "Category": r.category,
            "Method": r.method,
            "MRR@3 (T²-RAGBench)": f"{r.mrr3:.4f}" if r.mrr3 is not None else "N/A",
            "R@3": f"{r.r3:.4f}" if r.r3 is not None else "N/A",
            "NDCG@3": f"{r.ndcg3:.4f}" if r.ndcg3 is not None else "N/A",
            "MRR@10 (MTEB ref)": f"{r.mrr10_mteb_ref:.3f}" if r.mrr10_mteb_ref else "—",
            "Status": r.status,
            "Source": r.source,
            "Notes": r.notes,
        }
        for r in results
    ])

    # Sort: ours first, then general SOTA, then domain SOTA
    def sort_key(row):
        if row["Category"] == "ours":
            return (0, -float(row["MRR@3 (T²-RAGBench)"].replace("N/A", "0")))
        elif row["Category"] == "general_sota":
            return (1, -float(row["MRR@3 (T²-RAGBench)"].replace("N/A", "0")))
        else:
            return (2, -float(row["MRR@3 (T²-RAGBench)"].replace("N/A", "0")))

    df["_sort"] = df.apply(sort_key, axis=1)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    print("\n" + "=" * 100)
    print(f"SOTA COMPARISON — {dataset.upper()} (T²-RAGBench | T²-RAGBench MRR@3 = retrieval-only | MTEB MRR@10 = cross-benchmark reference)")
    print("=" * 100)

    # Print by category
    for cat in ["ours", "general_sota", "domain_sota"]:
        cat_df = df[df["Category"] == cat]
        if cat_df.empty:
            continue
        label = {"ours": "★ OURS", "general_sota": "◆ General SOTA (MTEB)", "domain_sota": "▼ Domain Baselines (T²-RAGBench)"}[cat]
        print(f"\n  [{label}]")
        for _, row in cat_df.iterrows():
            mrr3 = row["MRR@3 (T²-RAGBench)"]
            mrr10 = row["MRR@10 (MTEB ref)"]
            flag = ""
            if cat == "ours":
                flag = " ← ours"
            elif mrr10 != "—" and mrr3 != "N/A":
                try:
                    delta = float(mrr3) - float(mrr10)
                    flag = f" (Δ vs MTEB: {delta:+.3f})"
                except ValueError:
                    pass
            print(f"    {row['Method']:<25} T²-RAGBench MRR@3={mrr3:>8}  R@3={row['R@3']:>8}  MTEB MRR@10={mrr10:>8}{flag}")

    print("=" * 100 + "\n")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SOTA benchmark comparison")
    parser.add_argument("--dataset", type=str, default="finqa",
                        choices=["finqa", "convfinqa", "tatqa"])
    parser.add_argument("--category", type=str, default="all",
                        choices=["domain", "general", "ours", "all"])
    parser.add_argument("--models", type=str, nargs="+",
                        default=["gte-large", "bge-large", "bge-m3", "e5-base-v2", "colbertv2"],
                        help="General SOTA models to run")
    parser.add_argument("--ours-modes", type=str, nargs="+",
                        default=["gsr", "hybridgsr"],
                        help="Our methods to run")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save CSV to path")
    args = parser.parse_args()

    all_results: list[BenchmarkResult] = []

    # Domain SOTA (from paper)
    if args.category in ("domain", "all"):
        logger.info("=== Domain SOTA (from T²-RAGBench paper) ===")
        domain_results = run_domain_sota(args.dataset)
        all_results.extend(domain_results)

    # General SOTA (MTEB models — run ourselves)
    if args.category in ("general", "all"):
        logger.info("=== General SOTA (MTEB models, run on T²-RAGBench) ===")
        general_results = run_general_sota(
            args.dataset, args.models, args.top_k, args.device,
        )
        all_results.extend(general_results)

    # Ours (GSR / HybridGSR)
    if args.category in ("ours", "all"):
        logger.info("=== Ours (GSR + HybridGSR) ===")
        ours_results = run_ours(args.ours_modes, args.dataset, args.top_k, args.device)
        all_results.extend(ours_results)

    # Print table
    df = print_comparison_table(all_results, args.dataset)

    if args.save:
        df.to_csv(args.save, index=False)
        logger.info(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
