"""Local dataset cache helpers for T²-RAGBench snapshots.

The repo uses a local-once, run-offline-afterward flow:
1) Run `data/download_t2ragbench.py` once to save the HuggingFace dataset
   into `source/data/t2-ragbench/<ConfigName>/`.
2) All training/evaluation code reads from that local snapshot only.

Set `GSR_CACL_DATA_ROOT` to override the default local data directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DatasetDict, load_from_disk

DATA_ROOT_ENV = "GSR_CACL_DATA_ROOT"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "t2-ragbench"

DATASET_CONFIGS: dict[str, dict[str, str]] = {
    "finqa": {"config_name": "FinQA", "train_split": "train", "eval_split": "test"},
    "convfinqa": {"config_name": "ConvFinQA", "train_split": "turn_0", "eval_split": "turn_0"},
    "tatqa": {"config_name": "TAT-DQA", "train_split": "train", "eval_split": "test"},
}


def get_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the local data root, preferring an explicit path or env var."""
    if data_root is not None:
        return Path(data_root).expanduser().resolve()

    env_value = os.environ.get(DATA_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()

    return DEFAULT_DATA_ROOT


def resolve_config_name(name: str) -> str:
    """Resolve a dataset key (finqa/tatqa/convfinqa) to its HF config name."""
    key = name.lower().strip()
    return DATASET_CONFIGS.get(key, {}).get("config_name", name)


def get_dataset_dir(config_name: str, data_root: str | Path | None = None) -> Path:
    """Return the on-disk directory for a saved dataset config."""
    return get_data_root(data_root) / config_name


def load_local_dataset(config_name: str, data_root: str | Path | None = None) -> Any:
    """Load a saved HuggingFace dataset snapshot from disk."""
    dataset_dir = get_dataset_dir(config_name, data_root)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Local dataset snapshot not found: {dataset_dir}. "
            f"Run `python data/download_t2ragbench.py` first or set {DATA_ROOT_ENV}."
        )
    return load_from_disk(str(dataset_dir))


def load_local_split_df(
    config_name: str,
    split: str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load one split of a saved dataset snapshot as a pandas DataFrame."""
    dataset = load_local_dataset(config_name, data_root=data_root)
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found in local dataset '{config_name}'.")
        return dataset[split].to_pandas()
    return dataset.to_pandas()
