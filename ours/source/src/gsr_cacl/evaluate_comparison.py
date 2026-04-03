#!/usr/bin/env python3
"""
evaluate_comparison.py — Run GSR methods and compare with published baselines.

Compares GSR/HybridGSR against baseline results from T²-RAGBench paper.

Usage:
    python -m gsr_cacl.evaluate_comparison --dataset finqa
    python -m gsr_cacl.evaluate_comparison --dataset tatqa --methods gsr hybridgsr
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import pandas as pd

from gsr_cacl.benchmark_gsr import run_gsr_benchmark, DATASET_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Published baseline results (from T²-RAGBench paper, Table 3)
BASELINE_RESULTS = {
    "finqa": {
        "BM25":       {"MRR@3": 0.280, "Recall@3": 0.36},
        "HybridBM25": {"MRR@3": 0.352, "Recall@3": 0.45},
        "ColBERTv2":  {"MRR@3": 0.310, "Recall@3": 0.40},
    },
    "tatqa": {
        "BM25":       {"MRR@3": 0.220, "Recall@3": 0.30},
        "HybridBM25": {"MRR@3": 0.290, "Recall@3": 0.38},
    },
    "convfinqa": {
        "BM25":       {"MRR@3": 0.240, "Recall@3": 0.33},
        "HybridBM25": {"MRR@3": 0.310, "Recall@3": 0.40},
    },
}


def run_comparison(
    dataset: str,
    methods: list[str],
    embedding: str,
    top_k: int = 3,
) -> pd.DataFrame:
    """Run specified methods and combine with baselines into a comparison table."""
    rows = []

    for method in methods:
        logger.info(f"Running {method} on {dataset}")
        ds_cfg = DATASET_CONFIGS.get(dataset)
        if ds_cfg is None:
            logger.error(f"Unknown dataset: {dataset}")
            continue
        try:
            results = run_gsr_benchmark(
                mode=method,
                config_name=ds_cfg["config_name"],
                eval_split=ds_cfg["eval_split"],
                embedding_model=embedding, top_k=top_k,
            )
            rows.append({
                "Method": method.upper(),
                "MRR@3": results.get("MRR@3", 0.0),
                "Recall@1": results.get("Recall@1", 0.0),
                "Recall@3": results.get("Recall@3", 0.0),
                "Recall@5": results.get("Recall@5", 0.0),
                "NDCG@3": results.get("NDCG@3", 0.0),
                "Source": "ours",
            })
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            rows.append({"Method": method.upper(), "Source": f"error: {e}"})

    # Append baselines
    for name, scores in BASELINE_RESULTS.get(dataset, {}).items():
        rows.append({
            "Method": name,
            "MRR@3": scores.get("MRR@3"),
            "Recall@3": scores.get("Recall@3"),
            "Source": "baseline",
        })

    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, dataset: str) -> None:
    """Pretty-print comparison table."""
    print("\n" + "=" * 80)
    print(f"COMPARISON TABLE — {dataset.upper()} (T²-RAGBench)")
    print("=" * 80)

    cols = ["Method", "MRR@3", "Recall@1", "Recall@3", "Recall@5", "NDCG@3", "Source"]
    present = [c for c in cols if c in df.columns]
    print(df[present].to_string(index=False, float_format="%.4f"))

    # Highlight improvement
    valid = df[df["MRR@3"].notna()]
    if not valid.empty:
        best_idx = valid["MRR@3"].idxmax()
        best = valid.loc[best_idx]
        baselines = df[df["Source"] == "baseline"]
        if not baselines.empty:
            bl_best = baselines.loc[baselines["MRR@3"].idxmax()]
            delta = best["MRR@3"] - bl_best["MRR@3"]
            print(f"\n  Best: {best['Method']} ({best['MRR@3']:.4f})")
            print(f"  vs {bl_best['Method']}: {delta:+.4f} ({delta/bl_best['MRR@3']*100:+.1f}%)")

    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="GSR vs Baseline comparison")
    parser.add_argument("--dataset", type=str, default="finqa",
                        choices=["finqa", "convfinqa", "tatqa"])
    parser.add_argument("--methods", type=str, nargs="+", default=["gsr", "hybridgsr"])
    parser.add_argument("--embedding", type=str,
                        default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--save", type=str, default=None, help="Save CSV to path")
    args = parser.parse_args()

    df = run_comparison(args.dataset, args.methods, args.embedding, args.top_k)
    print_table(df, args.dataset)

    if args.save:
        df.to_csv(args.save, index=False)
        logger.info(f"Table saved to {args.save}")


if __name__ == "__main__":
    main()
