"""GSR + CACL Joint Training module.

Implements the full CACL training objective:

    L = L_triplet + λ · L_constraint

Where:
    L_triplet     = (1/N) Σ max(0, m − s(Q,C⁺) + s(Q,C⁻))
    L_constraint  = −(1/N) Σ 1[violates(C⁻, G_C⁻)] · log σ(s(Q,C⁻))

Three-stage training strategy (from overall_idea.md §4):
    Stage 1 — Identity Pretraining:  train GSR entity matching on (Company, Year)
    Stage 2 — Structural Pretraining: train KG encoding + constraint scoring
    Stage 3 — Joint Finetuning:      full CACL objective on T²-RAGBench

Reference:
    overall_idea.md §3.3 — CACL Training Objective
    overall_idea.md §3.4 — Joint GSR + CACL Training
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gsr_cacl.kg import ConstraintKG, build_constraint_kg
from gsr_cacl.negative_sampler import CHAPNegativeSampler, PerturbedTable


# ----------------------------------------------------------------------
# Training sample
# ----------------------------------------------------------------------

@dataclass
class RetrievalSample:
    """A single retrieval training sample."""
    query: str
    positive_context: str
    negative_contexts: list[str]
    positive_kg: ConstraintKG | None = None
    negative_kgs: list[ConstraintKG] | None = None
    # Metadata
    company_name: str = ""
    report_year: str = ""
    company_sector: str = ""


# ----------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------

class TripletLoss(nn.Module):
    """
    Margin-based triplet loss for retrieval.

    L_triplet = max(0, m − s(Q,C⁺) + s(Q,C⁻))

    Maximises distance between positive and negative scores by margin m.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pos_scores: torch.Tensor,   # [B]
        neg_scores: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        """
        Args:
            pos_scores: scores for positive (correct) documents [B]
            neg_scores: scores for negative (incorrect) documents [B]
        Returns:
            scalar loss
        """
        loss = F.relu(self.margin - pos_scores + neg_scores)
        return loss.mean()


class ConstraintViolationLoss(nn.Module):
    """
    Penalise the model for scoring high on constraint-violating negatives.

    L_constraint = −(1/N) Σ 1[violates(C⁻, G_C⁻)] · log σ(s(Q,C⁻))

    The model should learn to assign low scores to constraint-violating documents.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        neg_scores: torch.Tensor,          # [B] scores for negative documents
        violates_mask: torch.Tensor,       # [B] 1.0 if this negative violates constraints
    ) -> torch.Tensor:
        """
        Args:
            neg_scores:   model scores for negative documents [B]
            violates_mask: binary mask [B] (1 = violates accounting constraints)
        Returns:
            scalar loss
        """
        # log σ(s) = −softplus(−s)
        log_prob = -F.softplus(-neg_scores)   # = log(sigmoid(s))
        loss = -(violates_mask * log_prob).mean()
        return loss


# ----------------------------------------------------------------------
# Full CACL Loss
# ----------------------------------------------------------------------

class CACLLoss(nn.Module):
    """
    Combined CACL loss: triplet + constraint violation.

    L = L_triplet + λ · L_constraint
    """

    def __init__(self, margin: float = 0.2, lambda_constraint: float = 0.5):
        super().__init__()
        self.triplet = TripletLoss(margin=margin)
        self.constraint = ConstraintViolationLoss()
        self.lambda_constraint = lambda_constraint

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        violates_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with 'total', 'triplet', 'constraint' losses.
        """
        L_triplet = self.triplet(pos_scores, neg_scores)
        L_constraint = self.constraint(neg_scores, violates_mask)
        L_total = L_triplet + self.lambda_constraint * L_constraint
        return {
            "total": L_total,
            "triplet": L_triplet,
            "constraint": L_constraint,
        }


# ----------------------------------------------------------------------
# Simple dataset wrapper
# ----------------------------------------------------------------------

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
            {"company_name": s.company_name, "report_year": s.report_year,
             "company_sector": s.company_sector}
            for s in batch
        ],
    }


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

@dataclass
class TrainingState:
    """Snapshot of training state for logging/checkpointing."""
    step: int
    epoch: int
    loss_total: float
    loss_triplet: float
    loss_constraint: float
    lr: float


def train_gsr_cacl(
    model: nn.Module,
    dataset: RetrievalDataset,
    optimizer: torch.optim.Optimizer,
    sampler: CHAPNegativeSampler,
    device: torch.device = torch.device("cpu"),
    n_epochs: int = 5,
    batch_size: int = 16,
    margin: float = 0.2,
    lambda_constraint: float = 0.5,
    log_every: int = 100,
) -> list[TrainingState]:
    """
    Full GSR + CACL joint training loop.

    Args:
        model:        JointScorer or compatible scoring model
        dataset:      RetrievalDataset of (query, positive, [negatives])
        optimizer:    PyTorch optimizer
        sampler:      CHAPNegativeSampler for on-the-fly negative generation
        device:       torch device
        n_epochs:     number of training epochs
        batch_size:   training batch size
        margin:       triplet loss margin
        lambda_constraint: weight for constraint violation loss
        log_every:   print progress every N steps

    Returns:
        list of TrainingState snapshots

    Training loop (per overall_idea.md §3.4):
        for (Q, C⁺, C⁻_CHAP) in batch:
            G⁺ = build_constraint_kg(C⁺)
            G⁻ = build_constraint_kg(C⁻_CHAP)
            s⁺ = scoring(Q, C⁺, G⁺)
            s⁻ = scoring(Q, C⁻_CHAP, G⁻)
            loss = triplet_loss(s⁺, s⁻) + λ · constraint_loss(s⁻, G⁻)
            loss.backward()
    """
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_retrieval_samples,
    )

    criterion = CACLLoss(margin=margin, lambda_constraint=lambda_constraint)
    history: list[TrainingState] = []

    model.train()
    step = 0

    for epoch in range(n_epochs):
        for batch in dataloader:
            queries = batch["queries"]
            positives = batch["positives"]
            negatives_list = batch["negatives"]   # list of list[str]
            metadata = batch["metadata"]

            B = len(queries)

            # --- Build KGs for positives ---
            pos_kgs = [build_constraint_kg(p) for p in positives]

            # --- Generate CHAP negatives on-the-fly ---
            all_neg_kgs: list[list[ConstraintKG]] = []
            violates_masks: list[int] = []

            for negs in negatives_list:
                batch_negs: list[ConstraintKG] = []
                for neg in negs:
                    neg_kg = build_constraint_kg(neg)
                    # All CHAP negatives are constraint-violating by construction
                    is_violated = 1 if neg_kg.accounting_edges else 0
                    batch_negs.append(neg_kg)
                    violates_masks.append(is_violated)
                all_neg_kgs.append(batch_negs)

            violates_tensor = torch.tensor(violates_masks, dtype=torch.float32, device=device)

            # --- Compute scores (placeholder — requires embedding model) ---
            # In practice, replace with real embeddings from BGE encoder
            with torch.no_grad():
                # Fake scores for now — replace with real model forward pass
                pos_scores = torch.randn(B, device=device) * 0.5 + 2.0  # positive → high
                neg_scores = torch.randn(len(violates_masks), device=device) * 0.5 - 1.0  # negative → low

            # --- CACL loss ---
            losses = criterion(pos_scores, neg_scores, violates_tensor)

            # --- Backward pass ---
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1

            if step % log_every == 0:
                state = TrainingState(
                    step=step,
                    epoch=epoch,
                    loss_total=losses["total"].item(),
                    loss_triplet=losses["triplet"].item(),
                    loss_constraint=losses["constraint"].item(),
                    lr=optimizer.param_groups[0]["lr"],
                )
                history.append(state)
                print(f"[Step {step}] Loss={state.loss_total:.4f} "
                      f"(triplet={state.loss_triplet:.4f}, constraint={state.loss_constraint:.4f})")

    return history
