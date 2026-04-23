#!/usr/bin/env python3
"""
Template Coverage Analysis — Exp 5 (overall_idea.md §5.3).

Runs template matching on a sample of T²-RAGBench tables to estimate
IFRS/GAAP template coverage. Validates the claim: ~80-90% FinQA, ~70% TAT-DQA.

Usage:
    python -m gsr_cacl.template_coverage_analysis --dataset finqa --sample 200
    python -m gsr_cacl.template_coverage_analysis --dataset tatqa --sample 300
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from tqdm import tqdm

from gsr_cacl.datasets.local_data import DATASET_CONFIGS, load_local_split_df
from gsr_cacl.templates import TEMPLATES, match_template, normalize_header

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def extract_table_headers(content: str) -> list[str]:
    """Extract headers from the first markdown table in content."""
    for line in content.split("\n"):
        stripped = line.strip()
        if "|" in stripped:
            cells = [c.strip() for c in stripped.split("|")]
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            if len(cells) >= 2:
                return cells
    return []


def survey_coverage(
    config_name: str | None,
    split: str,
    sample_size: int = 200,
) -> dict:
    """Survey template matching coverage on a sample of documents."""
    logger.info(f"Loading local dataset: {config_name} ({split})")
    df = load_local_split_df(config_name, split)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    results = {
        "total": len(df),
        "with_table": 0,
        "with_header": 0,
        "matched_template": 0,
        "high_confidence": 0,
        "no_match": 0,
        "template_distribution": Counter(),
        "confidence_scores": [],
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Surveying templates"):
        content = str(row.get("context", "") or row.get("table", ""))
        if not content:
            continue

        headers = extract_table_headers(content)
        if not headers:
            continue

        results["with_table"] += 1
        results["with_header"] += 1

        canonical = [normalize_header(h) for h in headers]
        template, confidence = match_template(canonical)
        results["confidence_scores"].append(confidence)

        if template is not None:
            results["matched_template"] += 1
            results["template_distribution"][template.name] += 1
            if confidence >= 0.7:
                results["high_confidence"] += 1
        else:
            results["no_match"] += 1

    n = results["total"]
    results["coverage_ratio"] = results["matched_template"] / n if n else 0
    results["high_conf_ratio"] = results["high_confidence"] / n if n else 0
    results["avg_confidence"] = (
        sum(results["confidence_scores"]) / len(results["confidence_scores"])
        if results["confidence_scores"]
        else 0
    )
    return results


def print_report(results: dict, dataset: str) -> None:
    """Print a formatted coverage report."""
    print("\n" + "=" * 70)
    print(f"TEMPLATE COVERAGE ANALYSIS — {dataset.upper()}")
    print("=" * 70)

    n = results["total"]
    print(f"\n  Total samples:         {n}")
    print(f"  With table:            {results['with_table']} "
          f"({results['with_table']/n*100:.1f}%)" if n else "")
    print(f"  Template matched:      {results['matched_template']} "
          f"({results['coverage_ratio']*100:.1f}%)")
    print(f"  High confidence (>=0.7):{results['high_confidence']} "
          f"({results['high_conf_ratio']*100:.1f}%)")
    print(f"  Avg confidence:        {results['avg_confidence']:.3f}")

    print("\n  Template distribution:")
    for tname, count in results["template_distribution"].most_common(15):
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {tname:<25} {count:>5} ({pct:5.1f}%) {bar}")

    print("\n  Claim validation:")
    print(f"    Coverage >= 80%: {'PASS' if results['coverage_ratio'] >= 0.80 else 'FAIL'} "
          f"({results['coverage_ratio']*100:.1f}%)")
    print(f"    High-conf >= 70%: {'PASS' if results['high_conf_ratio'] >= 0.70 else 'FAIL'} "
          f"({results['high_conf_ratio']*100:.1f}%)")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Template coverage analysis")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["finqa", "convfinqa", "tatqa"])
    parser.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIGS[args.dataset]
    results = survey_coverage(ds_cfg["config_name"], ds_cfg["eval_split"], args.sample)
    print_report(results, args.dataset)


if __name__ == "__main__":
    main()
