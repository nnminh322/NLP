"""GSR + CACL joint training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gsr_cacl.kg.builder import build_kg_from_markdown
from gsr_cacl.negative_sampler.chap import CHAPNegativeSampler
from gsr_cacl.training.losses import CACLLoss
from gsr_cacl.training.data import RetrievalDataset, RetrievalSample, collate_retrieval_samples

logger = logging.getLogger(__name__)


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

    Training loop:
        for (Q, C+, C-_CHAP) in batch:
            G+ = build_constraint_kg(C+)
            G- = build_constraint_kg(C-_CHAP)
            s+ = scoring(Q, C+, G+)
            s- = scoring(Q, C-_CHAP, G-)
            loss = triplet_loss(s+, s-) + λ · constraint_loss(s-, G-)
            loss.backward()
    """
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
            negatives_list = batch["negatives"]

            B = len(queries)

            # Build KGs for positives
            pos_kgs = [build_kg_from_markdown(p) for p in positives]

            # Generate CHAP negatives on-the-fly
            violates_masks: list[int] = []
            for negs in negatives_list:
                for neg in negs:
                    neg_kg = build_kg_from_markdown(neg)
                    is_violated = 1 if neg_kg.accounting_edges else 0
                    violates_masks.append(is_violated)

            violates_tensor = torch.tensor(violates_masks, dtype=torch.float32, device=device)

            # Placeholder scores — replace with real model forward pass
            with torch.no_grad():
                pos_scores = torch.randn(B, device=device) * 0.5 + 2.0
                neg_scores = torch.randn(max(len(violates_masks), 1), device=device) * 0.5 - 1.0

            losses = criterion(pos_scores, neg_scores, violates_tensor)

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
                logger.info(
                    f"[Step {step}] Loss={state.loss_total:.4f} "
                    f"(triplet={state.loss_triplet:.4f}, constraint={state.loss_constraint:.4f})"
                )

    return history
