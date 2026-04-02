#!/usr/bin/env python3
"""
evaluate_comparison.py — Run all methods and compare with baseline results.

Compares:
  - GSR (ours)
  - HybridGSR (ours)
  - Baseline HybridBM25 (from g4k)

Usage:
    python -m gsr_cacl.evaluate_comparison --dataset finqa
    python -m gsr_cacl.evaluate_comparison --dataset tatqa --methods gsr hybridgsr
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

from gsr_cacl.benchmark_gsr import run_gsr_benchmark


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline results (from T²-RAGBench paper)
# ---------------------------------------------------------------------------
BASELINE_RESULTS = {
    "finqa": {
        "HybridBM25": {"mrr_at_k": 0.352, "recall_at_k": 0.45},
        "BM25":       {"mrr_at_k": 0.28,  "recall_at_k": 0.36},
        "ColBERTv2":  {"mrr_at_k": 0.31,  "recall_at_k": 0.40},
    },
    "tatqa": {
        "HybridBM25": {"mrr_at_k": 0.29, "recall_at_k": 0.38},
        "BM25":       {"mrr_at_k": 0.22, "recall_at_k": 0.30},
    },
    "convfinqa": {
        "HybridBM25": {"mrr_at_k": 0.31, "recall_at_k": 0.40},
        "BM25":       {"mrr_at_k": 0.24, "recall_at_k": 0.33},
    },
}


def load_previous_results(output_dir: Path) -> dict:
    """Load metrics from a previous run."""
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def run_all_methods(
    dataset: str,
    methods: list[str],
    embedding: str,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Run all specified methods and return comparison table.
    """
    rows = []
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    for method in methods:
        logger.info(f"Running method: {method} on {dataset}")
        try:
            results = run_gsr_benchmark(
                mode=method,
                dataset=dataset,
                embedding_model=embedding,
                top_k=top_k,
            )
            rows.append({
                "method": method.upper(),
                "MRR@3": results.get("mrr_at_k", 0.0),
                "R@1": results.get("recall_at_k", 0.0),
                "R@3": results.get("recall_at_k", 0.0),
                "NDCG@3": results.get("ndcg_at_k", 0.0),
                "status": "completed",
            })
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            rows.append({
                "method": method.upper(),
                "MRR@3": None,
                "R@1": None,
                "R@3": None,
                "NDCG@3": None,
                "status": f"error: {e}",
            })

    # Add baseline rows
    baseline = BASELINE_RESULTS.get(dataset, {})
    for name, scores in baseline.items():
        rows.append({
            "method": name,
            "MRR@3": scores["mrr_at_k"],
            "R@1": scores["recall_at_k"],
            "R@3": scores["recall_at_k"],
            "NDCG@3": None,
            "status": "baseline",
        })

    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 80)
    print("RETRIEVAL COMPARISON TABLE — T²-RAGBench")
    print("=" * 80)

    # Sort: ours first, then baselines
    def sort_key(method: str) -> tuple[int, str]:
        if "gsr" in method.lower():
            return (0, method)
        return (1, method)

    df = df.copy()
    df["_sort"] = df["method"].apply(lambda m: sort_key(m))
    df = df.sort_values("_sort").drop(columns=["_sort"])

    print(f"\n{'Method':<20} {'MRR@3':>10} {'R@1':>10} {'R@3':>10} {'NDCG@3':>10} {'Status':>12}")
    print("-" * 80)
    for _, row in df.iterrows():
        mrr = f"{row['MRR@3']:.4f}" if row['MRR@3'] is not None else "  N/A  "
        r1 = f"{row['R@1']:.4f}" if row['R@1'] is not None else "  N/A  "
        r3 = f"{row['R@3']:.4f}" if row['R@3'] is not None else "  N/A  "
        ndcg = f"{row['NDCG@3']:.4f}" if row.get('NDCG@3') is not None else "  N/A  "
        status = row.get('status', 'unknown')
        print(f"{row['method']:<20} {mrr:>10} {r1:>10} {r3:>10} {ndcg:>10} {status:>12}")

    print("=" * 80)

    # Highlight best
    valid = df[df['MRR@3'].notna()]
    if not valid.empty:
        best_idx = valid['MRR@3'].idxmax()
        best_method = valid.loc[best_idx, 'method']
        print(f"\n  ★ Best MRR@3: {best_method} ({valid.loc[best_idx, 'MRR@3']:.4f})")

        # Improvement over baseline
        baselines = df[df['status'] == 'baseline']
        if not baselines.empty:
            best_baseline = baselines['MRR@3'].idxmax()
            baseline_mrr = baselines.loc[best_baseline, 'MRR@3']
            our_best = valid.loc[best_idx, 'MRR@3']
            delta = our_best - baseline_mrr
            print(f"  Δ vs best baseline ({baselines.loc[best_baseline, 'method']}): "
                  f"{delta:+.4f} ({delta/baseline_mrr*100:+.1f}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="GSR vs Baseline comparison")
    parser.add_argument("--dataset", type=str, default="finqa",
                        choices=["finqa", "convfinqa", "tatqa", "all"])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["gsr", "hybridgsr"],
                        help="Methods to run")
    parser.add_argument("--embedding", type=str,
                        default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--save", type=str, default=None,
                        help="Save comparison table to CSV path")
    args = parser.parse_args()

    df = run_all_methods(
        dataset=args.dataset,
        methods=args.methods,
        embedding=args.embedding,
        top_k=args.top_k,
    )

    print_comparison_table(df)

    if args.save:
        df.to_csv(args.save, index=False)
        logger.info(f"Comparison table saved to {args.save}")


if __name__ == "__main__":
    main()
