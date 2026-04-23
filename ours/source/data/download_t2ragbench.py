#!/usr/bin/env python3
"""Download T²-RAGBench once and save it into the repository-local data cache.

Run from `source/`:

    python data/download_t2ragbench.py --datasets all

The downloaded snapshots are stored under:

    source/data/t2-ragbench/<ConfigName>/

After that, training/evaluation code reads from disk only.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Workaround: suppress multiprocess.ResourceTracker destructor error on Python 3.12.
# This avoids an exit-time AttributeError when multiprocess's ResourceTracker
# tries to access private RLock internals that changed in CPython 3.12.
try:
    import multiprocess.resource_tracker as _rt
    if hasattr(_rt, "ResourceTracker"):
        _rt.ResourceTracker.__del__ = lambda self: None
except Exception:
    pass

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gsr_cacl.datasets.local_data import DATASET_CONFIGS, get_data_root, resolve_config_name  # noqa: E402

HF_DATASET_NAME = "G4KMU/t2-ragbench"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_selection(requested: list[str]) -> list[str]:
    if not requested or requested == ["all"]:
        return list(DATASET_CONFIGS.keys())

    selection: list[str] = []
    for item in requested:
        key = item.lower().strip()
        if key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{item}'. Choose from: all, {', '.join(DATASET_CONFIGS)}")
        selection.append(key)
    return selection


def download_one(dataset_key: str, data_root: Path, force: bool = False) -> dict:
    config_name = resolve_config_name(dataset_key)
    target_dir = data_root / config_name

    if target_dir.exists():
        if force:
            logger.info(f"Removing existing snapshot: {target_dir}")
            shutil.rmtree(target_dir)
        else:
            logger.info(f"Skipping existing snapshot: {target_dir}")
            return {
                "dataset_key": dataset_key,
                "config_name": config_name,
                "path": str(target_dir),
                "skipped": True,
            }

    logger.info(f"Downloading {HF_DATASET_NAME}/{config_name} ...")
    dataset = load_dataset(HF_DATASET_NAME, config_name)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target_dir))

    split_sizes: dict[str, int] = {}
    if hasattr(dataset, "keys"):
        for split_name in dataset.keys():
            split_sizes[str(split_name)] = len(dataset[split_name])
    else:
        split_sizes["data"] = len(dataset)

    logger.info(f"Saved {config_name} to {target_dir}")
    return {
        "dataset_key": dataset_key,
        "config_name": config_name,
        "path": str(target_dir),
        "split_sizes": split_sizes,
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download T²-RAGBench into the repo-local cache")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to download: all | finqa | convfinqa | tatqa",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override the repo-local data root (defaults to source/data/t2-ragbench)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the local snapshot already exists",
    )
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    selection = _build_selection(args.datasets)
    manifest: dict[str, object] = {
        "dataset_name": HF_DATASET_NAME,
        "data_root": str(data_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "items": [],
    }

    for dataset_key in selection:
        manifest["items"].append(download_one(dataset_key, data_root, force=args.force))

    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
