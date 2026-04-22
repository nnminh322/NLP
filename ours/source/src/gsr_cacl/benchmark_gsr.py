#!/usr/bin/env python3
"""
GSR-CACL Retrieval Benchmark.

Runs GSR and HybridGSR on T²-RAGBench (§5 of overall_idea.md).
Evaluation metrics: MRR@3, Recall@1/3/5, NDCG@3.

Usage (Hydra):
    python -m gsr_cacl.benchmark_gsr dataset=gsr_finqa mode=gsr
    python -m gsr_cacl.benchmark_gsr dataset=gsr_tatqa mode=hybridgsr
    python -m gsr_cacl.benchmark_gsr dataset=gsr_finqa mode=gsr eval.sample_size=10

Legacy CLI (without Hydra):
    python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa --sample 10
"""

from __future__ import annotations

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
# Fallback dataset configs (used when Hydra configs are not available)
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict] = {
    "finqa": {"config_name": "FinQA", "eval_split": "test", "train_split": "train"},
    "convfinqa": {"config_name": "ConvFinQA", "eval_split": "turn_0", "train_split": "turn_0"},
    "tatqa": {"config_name": "TAT-DQA", "eval_split": "test", "train_split": "train"},
}

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, type] = {
    "gsr": GSRRetrieval,
    "hybridgsr": HybridGSR,
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
    config_name: str,
    eval_split: str,
    gsr_params: dict | None = None,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
    top_k: int = 3,
    device: str | None = None,
    output_dir: Path | None = None,
    sample_size: int | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    """
    Run GSR retrieval benchmark on a T²-RAGBench subset.

    Args:
        mode: "gsr" or "hybridgsr"
        config_name: HuggingFace config name, e.g. "FinQA"
        eval_split: HuggingFace split name, e.g. "test"
        gsr_params: GSR hyperparameters (alpha, beta, gamma, etc.)
        embedding_model: HuggingFace embedding model name
        top_k: number of documents to retrieve
        device: "cuda" or "cpu" (auto-detected if None)
        output_dir: where to save metrics JSON
        sample_size: limit QA samples for debugging

    Returns:
        dict of metric results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if gsr_params is None:
        gsr_params = {}

    RAGClass = METHOD_REGISTRY.get(mode)
    if RAGClass is None:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(METHOD_REGISTRY)}")

    # Load dataset
    split_data = load_t2ragbench_split(
        config_name=config_name,
        split=eval_split,
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

    # Build retrieval method with GSR hyperparameters from config
    rag_method = RAGClass(
        corpus=split_data.corpus,
        embedding_function=embeddings,
        top_k=top_k,
        device=device,
        checkpoint_path=checkpoint_path,
        alpha=gsr_params.get("alpha", 0.5),
        beta=gsr_params.get("beta", 0.3),
        gamma=gsr_params.get("gamma", 0.2),
        gat_hidden_dim=gsr_params.get("gat_hidden_dim", 256),
        gat_num_heads=gsr_params.get("gat_num_heads", 4),
        gat_num_layers=gsr_params.get("gat_num_layers", 2),
        contr_version=gsr_params.get("contr_version", "v1"),
        rel_tol=gsr_params.get("rel_tol", 1e-3),
        contr1=gsr_params.get("contr1", "v1"),
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
    dataset_label = config_name
    print("\n" + "=" * 60)
    print(f"GSR-CACL BENCHMARK  [{mode.upper()} on {dataset_label}]")
    print("=" * 60)
    for name, val in metrics.items():
        print(f"  {name:>12}: {val:.4f}")
    print("=" * 60 + "\n")

    # Save outputs
    if output_dir is None:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        output_dir = Path(f"outputs/gsr_benchmark/{config_name}/{mode}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    return metrics


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: supports both Hydra and legacy argparse CLI."""
    # Detect Hydra mode: if sys.argv has '=' overrides or no '--' flags
    use_hydra = any("=" in a and not a.startswith("--") for a in sys.argv[1:]) or len(sys.argv) == 1

    if use_hydra:
        _main_hydra()
    else:
        _main_argparse()


def _main_hydra() -> None:
    """Hydra-based entry point using conf/ YAML files."""
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
    def _run(cfg: DictConfig) -> None:
        ds = cfg.dataset
        gsr_params = OmegaConf.to_container(ds.get("gsr", {}), resolve=True)
        embedding_model = gsr_params.pop("embedding_model", None) or \
            cfg.get("model", {}).get("embedding", {}).get("name", "intfloat/multilingual-e5-large-instruct")

        checkpoint_path = gsr_params.pop("checkpoint", None)
        run_gsr_benchmark(
            mode=cfg.mode,
            config_name=ds.config_name,
            eval_split=ds.eval_split,
            gsr_params=gsr_params,
            embedding_model=embedding_model,
            top_k=gsr_params.pop("top_k", cfg.get("eval", {}).get("top_k", 3)),
            device=None,
            sample_size=cfg.get("eval", {}).get("sample_size"),
            checkpoint_path=checkpoint_path,
        )

    _run()


def _main_argparse() -> None:
    """Legacy argparse-based entry point."""
    import argparse

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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (.pt file) for entity embeddings and scorer weights")
    parser.add_argument(
        "--contr1", type=str, default="v1", choices=["v1", "v2"],
        help="Numeric encoding version: v1=log-scale [4 features], v2=ScaleAwareNumericEncoder "
             "(magnitude bin + mantissa + unit). Default: v1."
    )
    parser.add_argument(
        "--contr2", type=str, default="v1", choices=["v1", "v2"],
        help="Constraint scoring version: v1=fixed epsilon, v2=relative tolerance. Default: v1."
    )
    parser.add_argument(
        "--rel-tol", type=float, default=1e-3,
        help="Relative tolerance for contr2 (v2 only). Default: 1e-3 (0.1%% relative error)."
    )
    args = parser.parse_args()

    from gsr_cacl.scoring.constraint_score import ConstraintScoringVersion

    ds_cfg = DATASET_CONFIGS.get(args.dataset)
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    gsr_params = {
        "contr_version": args.contr2,
        "rel_tol": args.rel_tol,
        "contr1": args.contr1,
    }
    try:
        run_gsr_benchmark(
            mode=args.mode,
            config_name=ds_cfg["config_name"],
            eval_split=ds_cfg["eval_split"],
            embedding_model=args.embedding,
            top_k=args.top_k,
            device=args.device,
            output_dir=output_dir,
            sample_size=args.sample,
            gsr_params=gsr_params,
            checkpoint_path=args.checkpoint,
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
