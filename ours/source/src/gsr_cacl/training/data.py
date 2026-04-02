"""Training data structures and dataset wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import Dataset

from gsr_cacl.kg.data_structures import ConstraintKG


@dataclass
class RetrievalSample:
    """A single retrieval training sample."""
    query: str
    positive_context: str
    negative_contexts: list[str] = field(default_factory=list)
    positive_kg: Optional[ConstraintKG] = None
    negative_kgs: Optional[list[ConstraintKG]] = None
    company_name: str = ""
    report_year: str = ""
    company_sector: str = ""


class RetrievalDataset(Dataset):
    """Simple in-memory dataset for retrieval training."""

    def __init__(self, samples: list[RetrievalSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RetrievalSample:
        return self.samples[idx]


def collate_retrieval_samples(batch: list[RetrievalSample]) -> dict:
    """Collate function for DataLoader."""
    return {
        "queries": [s.query for s in batch],
        "positives": [s.positive_context for s in batch],
        "negatives": [s.negative_contexts for s in batch],
        "metadata": [
            {
                "company_name": s.company_name,
                "report_year": s.report_year,
                "company_sector": s.company_sector,
            }
            for s in batch
        ],
    }
