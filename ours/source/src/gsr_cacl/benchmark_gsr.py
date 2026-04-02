#!/usr/bin/env python3
"""
GSR-CACL Retrieval Benchmark.

Runs GSR and HybridGSR on T²-RAGBench using the same evaluation
framework as g4k's benchmark_retrieval.py, enabling direct comparison
with all baseline methods.

Usage:
    python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa
    python -m gsr_cacl.benchmark_gsr --mode hybridgsr --dataset tatqa
    python -m gsr_cacl.benchmark_gsr --mode gsr --dataset convfinqa

Evaluation metrics: MRR@3, Recall@1/3/5, NDCG@3 (same as baseline).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, DatasetDict

from g4k.internal.abstractions import BatchInferenceRunner, SamplingParams
from gsr_cacl.methods import GSRRetrieval, HybridGSR
from gsr_cacl.datasets import GSRFinQADataset, GSRTatQADataset


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset configs (mirrors g4k conf/dataset/)
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "finqa": {
        "name": "G4KMU/t2-ragbench",
        "config_name": "FinQA",
        "split": "FinQA",
        "meta_data_keys": ["report_year", "company_name", "company_sector"],
        "retrieval_query": (
            "Given a question about a company, retrieve relevant passages "
            "that answer the query."
        ),
        "dataset_wrapper": GSRFinQADataset,
    },
    "convfinqa": {
        "name": "G4KMU/t2-ragbench",
        "config_name": "ConvFinQA",
        "split": "ConvFinQA",
        "meta_data_keys": ["report_year", "company_name", "company_sector"],
        "retrieval_query": (
            "Given a multi-turn financial question, retrieve relevant passages."
        ),
        "dataset_wrapper": GSRFinQADataset,
    },
    "tatqa": {
        "name": "G4KMU/t2-ragbench",
        "config_name": "TAT-DQA",
        "split": "TAT-DQA",
        "meta_data_keys": ["report_year", "company_name", "company_sector"],
        "retrieval_query": (
            "Given a question about tabular financial data, "
            "retrieve the correct table context."
        ),
        "dataset_wrapper": GSRTatQADataset,
    },
    "all": {
        "name": "G4KMU/t2-ragbench",
        "config_name": None,   # load all
        "split": "train+validation+test",
        "meta_data_keys": ["report_year", "company_name", "company_sector"],
        "retrieval_query": (
            "Given a financial question, retrieve relevant text+table passages."
        ),
        "dataset_wrapper": GSRFinQADataset,
    },
}

# ---------------------------------------------------------------------------
# Method configs
# ---------------------------------------------------------------------------

METHOD_CONFIGS = {
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
# Metric computation (mirrors g4k factory_helper.py)
# ---------------------------------------------------------------------------

class MRRMetric:
    """Mean Reciprocal Rank at K."""

    def __init__(self, k: int = 3):
        self.k = k

    def __call__(self, responses: list) -> dict:
        mrr = 0.0
        for resp in responses:
            gt_id = str(resp.meta_data.get("reference_document", {}).get("id", ""))
            if not gt_id:
                continue
            for rank, doc in enumerate(resp.retrieved_docs[: self.k], start=1):
                if str(doc.id) == gt_id:
                    mrr += 1.0 / rank
                    break
        score = mrr / len(responses) if responses else 0.0
        return {"mrr_at_k": score, "metric": "MRR", "k": self.k}


class RecallMetric:
    """Recall at K."""

    def __init__(self, k: int = 3):
        self.k = k

    def __call__(self, responses: list) -> dict:
        hits = 0
        for resp in responses:
            gt_id = str(resp.meta_data.get("reference_document", {}).get("id", ""))
            if not gt_id:
                continue
            retrieved_ids = [str(doc.id) for doc in resp.retrieved_docs[: self.k]]
            if gt_id in retrieved_ids:
                hits += 1
        score = hits / len(responses) if responses else 0.0
        return {f"recall_at_k": score, "metric": "Recall", "k": self.k}


class NDCGMetric:
    """NDCG at K (binary relevance)."""

    def __init__(self, k: int = 3):
        self.k = k
        import math
        self.math = math

    def __call__(self, responses: list) -> dict:
        total = 0.0
        for resp in responses:
            gt_id = str(resp.meta_data.get("reference_document", {}).get("id", ""))
            if not gt_id:
                continue
            ndcg = 0.0
            for rank, doc in enumerate(resp.retrieved_docs[: self.k], start=1):
                if str(doc.id) == gt_id:
                    ndcg = 1.0 / self.math.log2(rank + 1)
                    break
            total += ndcg
        score = total / len(responses) if responses else 0.0
        return {"ndcg_at_k": score, "metric": "NDCG", "k": self.k}


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def load_corpus_for_dataset(dataset_name: str, config_name: str | None) -> pd.DataFrame:
    """Load full corpus (all splits) for a dataset."""
    full_dataset = load_dataset(dataset_name, config_name)
    dfs = []
    if isinstance(full_dataset, DatasetDict):
        for split_name in full_dataset.keys():
            dfs.append(full_dataset[split_name].to_pandas())
    else:
        dfs.append(full_dataset.to_pandas())
    return pd.concat(dfs, ignore_index=True)


def run_gsr_benchmark(
    mode: str,
    dataset: str,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
    top_k: int = 3,
    device: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Run GSR retrieval benchmark on a dataset.

    Args:
        mode:            "gsr" or "hybridgsr"
        dataset:         "finqa", "convfinqa", "tatqa", or "all"
        embedding_model: HuggingFace embedding model
        top_k:           retrieval top-k
        device:          torch device (auto-detected if None)
        output_dir:      where to save inference logs

    Returns:
        dict of metric results
    """
    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Get config
    ds_cfg = DATASET_CONFIGS.get(dataset)
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")

    method_cfg = METHOD_CONFIGS.get(mode)
    if method_cfg is None:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(METHOD_CONFIGS.keys())}")

    # ---- Load test split ----
    logger.info(f"Loading dataset: {ds_cfg['name']}/{ds_cfg['config_name']}")
    qa_dataset = load_dataset(
        ds_cfg["name"],
        ds_cfg.get("config_name"),
        split=ds_cfg["split"],
    ).to_pandas()
    logger.info(f"QA samples: {len(qa_dataset)}")

    # ---- Load full corpus ----
    logger.info("Loading full corpus for retrieval...")
    corpus_df = load_corpus_for_dataset(ds_cfg["name"], ds_cfg.get("config_name"))
    logger.info(f"Corpus size: {len(corpus_df)}")

    # ---- Build GSR dataset ----
    logger.info("Building GSR-enriched dataset...")
    DatasetWrapper = ds_cfg["dataset_wrapper"]
    dataset_obj = DatasetWrapper(
        df=qa_dataset,
        retrieval_query=ds_cfg["retrieval_query"],
        meta_data_keys=ds_cfg["meta_data_keys"],
        document_percentage=1.0,
        corpus_df=corpus_df,
        build_gsr_metadata=True,
    )
    logger.info(f"Corpus documents: {len(dataset_obj.context_collection)}")

    # ---- Template coverage diagnostics ----
    if hasattr(dataset_obj, "get_template_coverage_stats"):
        stats = dataset_obj.get_template_coverage_stats()
        logger.info("Template coverage stats:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")

    # ---- Setup embeddings ----
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
    )

    # ---- Setup RAG method ----
    logger.info(f"Initialising {mode} retrieval method...")
    RAGClass = method_cfg.pop("class")
    rag_kwargs = {k: v for k, v in method_cfg.items()}
    rag_kwargs.update({
        "context_collection": dataset_obj.get_context_collection(),
        "embedding_function": embeddings,
        "top_k": top_k,
        "retrieval_only": True,
        "device": device,
    })
    rag_method = RAGClass(**rag_kwargs)

    # ---- Dummy runner (retrieval-only doesn't need LLM) ----
    sampling_params = SamplingParams(temperature=0.0, max_tokens=0)
    runner = BatchInferenceRunner(
        sampling_params=sampling_params,
        model="dummy",
        base_url="http://localhost:9999",
    )

    # ---- Run retrieval ----
    retrieval_queries = dataset_obj._generate_retrieval_queries()
    logger.info(f"Running retrieval for {len(retrieval_queries)} queries...")
    responses = rag_method.run(
        runner=runner,
        sys_prompt="",
        user_prompts=dataset_obj.user_prompts,
        prompt_meta_data=dataset_obj.prompt_meta_data,
        retrieval_queries=retrieval_queries,
    )

    # ---- Evaluate ----
    metrics = [
        MRRMetric(k=3),
        RecallMetric(k=1),
        RecallMetric(k=3),
        RecallMetric(k=5),
        NDCGMetric(k=3),
    ]

    results: dict = {}
    print("\n" + "=" * 60)
    print(f"GSR-CACL BENCHMARK RESULTS  [{mode.upper()} on {dataset.upper()}]")
    print("=" * 60)

    for metric in metrics:
        result = metric(responses.response_data)
        results.update(result)
        key = list(result.keys())[0]
        print(f"  {metric.__class__.__name__}: {result[key]:.4f}")

    print("=" * 60 + "\n")

    # ---- Save inference log ----
    if output_dir is None:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        output_dir = Path(f"outputs/gsr_benchmark/{dataset}/{mode}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    log_path = output_dir / "inference_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in responses.response_data], f, indent=2, default=str)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSR-CACL Retrieval Benchmark on T²-RAGBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["gsr", "hybridgsr"],
        help="Retrieval method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["finqa", "convfinqa", "tatqa", "all"],
        help="Dataset subset",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="intfloat/multilingual-e5-large-instruct",
        help="HuggingFace embedding model",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Retrieval top-k (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Torch device (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        results = run_gsr_benchmark(
            mode=args.mode,
            dataset=args.dataset,
            embedding_model=args.embedding,
            top_k=args.top_k,
            device=args.device,
            output_dir=output_dir,
        )
        logger.info("Benchmark completed successfully.")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
