"""Helper functions for the evaluation factory."""

from typing import Any, Dict, Optional, Tuple

from g4k.internal.abstractions import BatchInferenceRunner, ResponseData
from g4k.internal.rag_methods import RAG_REGISTRY
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, create_model

from g4k.datasets.base.dataset_enum import Datasets
from g4k.datasets.base.dataset_interface import DatasetCollectionInterface
from g4k.evaluation.config import Config


def get_rag_method_class(method_name: str) -> type:
    """Get the RAG method class from a string."""
    try:
        # Standardize method name comparison
        method_map = {
            "Base": "Base",
            "HybridBM25": "HybridBM25",
            "Hyde": "Hyde",
            "Summarization": "Summarization"
        }
        internal_name = method_map.get(method_name, method_name)
        return RAG_REGISTRY[internal_name]
    except KeyError:
        raise ValueError(f"Unsupported RAG method: {method_name}") from None


def get_dataset_class(dataset_name: str) -> type[DatasetCollectionInterface]:
    """Get the dataset class from a string."""
    try:
        return Datasets(dataset_name).get_class()
    except ValueError:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from None


class MRRMetric:
    def __init__(self, k: int = 3):
        self.k = k
        self.name = f"MRR_at_{k}"
    def __call__(self, responses: list[ResponseData]) -> Any:
        mrr = 0.0
        for resp in responses:
            gt_id = str(resp.meta_data.get("reference_document", {}).get("id"))
            if not gt_id:
                continue
            for rank, doc in enumerate(resp.retrieved_docs[:self.k], start=1):
                if str(doc.id) == gt_id:
                    mrr += 1.0 / rank
                    break
        score = mrr / len(responses) if responses else 0.0
        return type("MetricOutput", (), {"score": score, "to_dict": lambda s: {"score": score, f"mrr_at_{self.k}": score}})()

class RecallAtKMetric:
    def __init__(self, k: int = 3):
        self.k = k
        self.name = f"Recall_at_{k}"
    def __call__(self, responses: list[ResponseData]) -> Any:
        hits = 0
        for resp in responses:
            gt_id = str(resp.meta_data.get("reference_document", {}).get("id"))
            if not gt_id:
                continue
            retrieved_ids = [str(doc.id) for doc in resp.retrieved_docs[:self.k]]
            if gt_id in retrieved_ids:
                hits += 1
        score = hits / len(responses) if responses else 0.0
        return type("MetricOutput", (), {"score": score, "to_dict": lambda s: {"score": score, f"recall_at_{self.k}": score}})()

class NDCGMetric:
    def __init__(self, k: int = 3):
        self.k = k
        self.name = f"NDCG_at_{k}"
    def __call__(self, responses: list[ResponseData]) -> Any:
        total_ndcg = 0.0
        import math
        for resp in responses:
            gt_id = str(resp.meta_data.get("reference_document", {}).get("id"))
            if not gt_id: continue
            # DCG for single ground truth at position 'rank' (1-indexed)
            # DCG = 1 / log2(rank + 1). IDCG = 1 / log2(1 + 1) = 1.
            ndcg = 0.0
            for rank, doc in enumerate(resp.retrieved_docs[:self.k], start=1):
                if str(doc.id) == gt_id:
                    ndcg = 1.0 / math.log2(rank + 1)
                    break
            total_ndcg += ndcg
        score = total_ndcg / len(responses) if responses else 0.0
        return type("MetricOutput", (), {"score": score, "to_dict": lambda s: {"score": score, f"ndcg_at_{self.k}": score}})()

def load_metrics(
    config: list[str | dict[str, dict[str, str]]], runner: BatchInferenceRunner = None
) -> list:
    """Load metrics from the config."""
    metrics = []
    for m_cfg in config:
        if isinstance(m_cfg, str):
            if m_cfg == "MeanReciprocalRank":
                metrics.append(MRRMetric(k=3))
            elif m_cfg == "RecallAtK":
                metrics.append(RecallAtKMetric(k=3))
            elif m_cfg == "NDCG":
                metrics.append(NDCGMetric(k=3))
        elif isinstance(m_cfg, dict):
            name = list(m_cfg.keys())[0]
            params = m_cfg[name]
            if name == "RecallAtK":
                metrics.append(RecallAtKMetric(k=params.get("k", 3)))
            elif name == "MeanReciprocalRank":
                metrics.append(MRRMetric(k=params.get("k", 3)))
            elif name == "NDCG":
                metrics.append(NDCGMetric(k=params.get("k", 3)))
    
    # If no metrics specified, default to R@3 and MRR@3 as per paper
    if not metrics:
        metrics = [MRRMetric(k=3), RecallAtKMetric(k=3), NDCGMetric(k=3)]
    return metrics


def get_response_format(cfg: Config) -> Optional[type[BaseModel]]:
    """Get the response model from the config."""
    if not hasattr(cfg, "dataset") or not hasattr(cfg.dataset, "response_format"):
        return None
    fields: Dict[str, Tuple[Any, Any]] = {
        k: (eval(v) if isinstance(v, str) else v, ...)
        for k, v in cfg.dataset.response_format.items()
    }
    return create_model("ResponseModel", **fields)  # type: ignore
