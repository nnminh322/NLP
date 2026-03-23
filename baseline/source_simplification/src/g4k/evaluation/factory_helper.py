"""Helper functions for the evaluation factory."""

from typing import Any, Dict, Optional, Tuple

from g4k.internal.abstractions import G4KRunner as BatchInferenceRunner
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


def load_metrics(
    config: list[str | dict[str, dict[str, str]]], runner: BatchInferenceRunner = None
) -> list:
    """Load metrics from the config. (Simplified for baseline)"""
    return []


def get_response_format(cfg: Config) -> Optional[type[BaseModel]]:
    """Get the response model from the config."""
    if not hasattr(cfg, "dataset") or not hasattr(cfg.dataset, "response_format"):
        return None
    fields: Dict[str, Tuple[Any, Any]] = {
        k: (eval(v) if isinstance(v, str) else v, ...)
        for k, v in cfg.dataset.response_format.items()
    }
    return create_model("ResponseModel", **fields)  # type: ignore
