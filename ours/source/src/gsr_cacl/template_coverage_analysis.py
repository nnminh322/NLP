#!/usr/bin/env python3
"""
template_coverage_analysis.py — Survey template coverage on T²-RAGBench.

Runs template matching on a sample of tables to estimate coverage.
Addresses Reviewer 4's critique: "Template coverage chưa được khảo sát"

Usage:
    python -m gsr_cacl.template_coverage_analysis --dataset finqa --sample 200
    python -m gsr_cacl.template_coverage_analysis --dataset tatqa --sample 300
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from gsr_cacl.templates import TEMPLATES, match_template, _normalize_header

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_table_headers(content: str) -> list[str]:
    """Extract headers from the first markdown table in content."""
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if "|" in stripped:
            cells = [c.strip() for c in stripped.split("|")]
            # Remove empty cells from leading/trailing pipes
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            if len(cells) >= 2:
                return cells
    return []


def survey_coverage(
    dataset_name: str,
    config_name: str | None,
    split: str,
    sample_size: int = 200,
) -> dict:
    """
    Survey template matching coverage on a sample of documents.

    Returns coverage statistics.
    """
    logger.info(f"Loading dataset: {dataset_name}/{config_name} ({split})")
    df = load_dataset(dataset_name, config_name, split=split).to_pandas()

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    results = {
        "total": len(df),
        "with_table": 0,
        "with_header": 0,
        "matched_template": 0,
        "high_confidence": 0,   # confidence >= 0.7
        "no_match": 0,
        "template_distribution": Counter(),
        "confidence_scores": [],
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Surveying templates"):
        content = row.get("context", "") or row.get("table", "")
        if not content:
            continue

        headers = extract_table_headers(content)
        if not headers:
            continue

        results["with_table"] += 1
        results["with_header"] += 1

        # Canonicalize and match
        canonical = [_normalize_header(h) for h in headers]
        template, confidence = match_template(canonical)
        results["confidence_scores"].append(confidence)

        if template is not None:
            results["matched_template"] += 1
            results["template_distribution"][template.name] += 1
            if confidence >= 0.7:
                results["high_confidence"] += 1
        else:
            results["no_match"] += 1

    # Summarize
    n = results["total"]
    results["coverage_ratio"] = results["matched_template"] / n if n else 0
    results["high_conf_ratio"] = results["high_confidence"] / n if n else 0
    results["avg_confidence"] = (
        sum(results["confidence_scores"]) / len(results["confidence_scores"])
        if results["confidence_scores"] else 0
    )

    return results


def print_report(results: dict, dataset: str) -> None:
    """Print a formatted coverage report."""
    print("\n" + "=" * 70)
    print(f"TEMPLATE COVERAGE ANALYSIS — {dataset.upper()}")
    print("=" * 70)

    print(f"\n  Total samples:         {results['total']}")
    print(f"  With table:            {results['with_table']} "
          f"({results['with_table']/results['total']*100:.1f}%)")
    print(f"  With headers:          {results['with_header']}")
    print(f"  Template matched:      {results['matched_template']} "
          f"({results['coverage_ratio']*100:.1f}%)")
    print(f"  High confidence (≥0.7):{results['high_confidence']} "
          f"({results['high_conf_ratio']*100:.1f}%)")
    print(f"  No match:              {results['no_match']}")
    print(f"  Avg confidence:        {results['avg_confidence']:.3f}")

    print("\n  Template distribution:")
    for tname, count in results["template_distribution"].most_common(15):
        pct = count / results["total"] * 100
        bar = "█" * int(pct / 2)
        print(f"    {tname:<25} {count:>5} ({pct:5.1f}%) {bar}")

    print("\n  Confidence histogram:")
    bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    scores = results["confidence_scores"]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = sum(1 for s in scores if lo <= s < hi)
        pct = count / len(scores) * 100 if scores else 0
        bar = "█" * int(pct / 2)
        print(f"    [{lo:.1f}–{hi:.1f}): {count:>5} ({pct:5.1f}%) {bar}")

    print("\n  Key claim validation:")
    claim_finqa = results["coverage_ratio"] >= 0.80
    claim_tatqa = results["coverage_ratio"] >= 0.70
    claim_high = results["high_conf_ratio"] >= 0.70

    print(f"    FinQA coverage ≥ 80%:     {'✅ PASS' if claim_finqa else '❌ FAIL'} "
          f"({results['coverage_ratio']*100:.1f}%)")
    print(f"    TAT-DQA coverage ≥ 70%:   {'✅ PASS' if claim_tatqa else '❌ FAIL'} "
          f"({results['coverage_ratio']*100:.1f}%)")
    print(f"    High-conf (≥0.7) ≥ 70%:  {'✅ PASS' if claim_high else '❌ FAIL'} "
          f"({results['high_conf_ratio']*100:.1f}%)")

    print("=" * 70 + "\n")


DATASET_SPLITS = {
    "finqa": ("G4KMU/t2-ragbench", "FinQA", "FinQA"),
    "convfinqa": ("G4KMU/t2-ragbench", "ConvFinQA", "ConvFinQA"),
    "tatqa": ("G4KMU/t2-ragbench", "TAT-DQA", "TAT-DQA"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Template coverage analysis")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["finqa", "convfinqa", "tatqa"])
    parser.add_argument("--sample", type=int, default=300,
                        help="Number of samples to survey")
    args = parser.parse_args()

    ds_cfg = DATASET_SPLITS[args.dataset]
    results = survey_coverage(
        dataset_name=ds_cfg[0],
        config_name=ds_cfg[1],
        split=ds_cfg[2],
        sample_size=args.sample,
    )
    print_report(results, args.dataset)


if __name__ == "__main__":
    main()
