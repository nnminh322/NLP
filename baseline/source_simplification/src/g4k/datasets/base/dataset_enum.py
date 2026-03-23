"""Enum for supported datasets."""

from enum import Enum

from g4k.datasets.base.dataset_interface import DatasetCollectionInterface
from g4k.datasets.convfinqa.convfinqa_qa import ConvFinQADataset
from g4k.datasets.finqa.finqa_qa import (
    FinQADatasetCollection,
    FinQADatasetFineTuned,
    FinQADatasetOneShot,
)
from g4k.datasets.tatqa.tatdqa_qa import TatQADatasetCollection
from g4k.datasets.vqa.vaq_qa import VAQDatasetCollection


class Datasets(Enum):
    """Enum for supported Dataset methods."""

    FinQA = "G4KMU/finqa-german"
    TatQA = "G4KMU/flare-tatqa-german"
    FinQAFineTuned = "finqa-finetuned"
    FinQAFineOneShot = "G4KMU/finqa"
    VAQA = "G4KMU/va_qa"
    ConvFinQA = "G4KMU/convfinqa"

    def get_class(self) -> type[DatasetCollectionInterface]:
        """Get the class for the dataset method."""
        method_classes = {
            Datasets.FinQA: FinQADatasetCollection,
            Datasets.TatQA: TatQADatasetCollection,
            Datasets.FinQAFineTuned: FinQADatasetFineTuned,
            Datasets.VAQA: VAQDatasetCollection,
            Datasets.ConvFinQA: ConvFinQADataset,
            Datasets.FinQAFineOneShot: FinQADatasetOneShot,
        }

        return method_classes[self]
