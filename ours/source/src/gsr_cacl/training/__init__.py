"""Training loop, losses, and data utilities for GSR-CACL."""

from gsr_cacl.training.losses import TripletLoss, ConstraintViolationLoss, CACLLoss
from gsr_cacl.training.entity_supcon_loss import EntitySupConLoss, EntityRegistry
from gsr_cacl.training.data import RetrievalSample, RetrievalDataset, collate_retrieval_samples
from gsr_cacl.training.trainer import train_gsr_cacl, TrainingState

__all__ = [
    # Losses
    "TripletLoss",
    "ConstraintViolationLoss",
    "CACLLoss",
    "EntitySupConLoss",      # NEW
    "EntityRegistry",         # NEW
    # Data
    "RetrievalSample",
    "RetrievalDataset",
    "collate_retrieval_samples",
    # Training
    "train_gsr_cacl",
    "TrainingState",
]
