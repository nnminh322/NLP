"""GSR + CACL joint training loop.

Reusable training loop that takes pre-built models and runs the full CACL objective.
Used by train.py for Stage 3, but can also be called directly.

Now supports end-to-end training through the text encoder with gradient flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gsr_cacl.encoders.gat_encoder import GATEncoder
from gsr_cacl.encoders.text_encoder import TextEncoder
from gsr_cacl.kg.builder import build_kg_from_markdown
from gsr_cacl.scoring.constraint_score import compute_constraint_score, ConstraintScoringVersion
from gsr_cacl.scoring.joint_scorer import JointScorer
from gsr_cacl.negative_sampler.chap import CHAPNegativeSampler, apply_chap_e
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


def _meta_hash(s: str) -> float:
    h = 0
    for c in s.lower().strip():
        h = (h * 31 + ord(c)) % 10000
    return h / 10000.0


def train_gsr_cacl(
    text_encoder: TextEncoder,
    scorer: JointScorer,
    gat_encoder: GATEncoder,
    dataset: RetrievalDataset,
    optimizer: torch.optim.Optimizer,
    sampler: CHAPNegativeSampler,
    device: torch.device = torch.device("cpu"),
    n_epochs: int = 5,
    batch_size: int = 16,
    margin: float = 0.2,
    lambda_constraint: float = 0.5,
    log_every: int = 100,
    contr_version: ConstraintScoringVersion = "v1",
    rel_tol: float = 1e-4,
    use_entity_signal: bool = True,
) -> list[TrainingState]:
    """
    Full GSR + CACL joint training loop with end-to-end gradient flow.

    Training loop:
        for (Q, C+, C-_CHAP) in batch:
            q  = TextEncoder(Q)         ← gradient flows through transformer
            d+ = TextEncoder(C+)        ← gradient flows through transformer
            d- = TextEncoder(C-)        ← gradient flows through transformer
            kg+ = GATEncoder(build_kg(C+))
            kg- = GATEncoder(build_kg(C-))
            s+ = JointScorer(q, d+, kg+)
            s- = JointScorer(q, d-, kg-)
            loss = CACL(s+, s-, violates)
            loss.backward()             ← updates TextEncoder + GATEncoder + JointScorer
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_retrieval_samples,
    )

    criterion = CACLLoss(margin=margin, lambda_constraint=lambda_constraint)
    history: list[TrainingState] = []

    text_encoder.train()
    scorer.train()
    gat_encoder.train()
    step = 0

    for epoch in range(n_epochs):
        for batch in dataloader:
            queries = batch["queries"]
            positives = batch["positives"]
            metadata = batch["metadata"]
            B = len(queries)

            pos_scores_list = []
            neg_scores_list = []
            violates_list = []

            for i in range(B):
                # Differentiable text embeddings
                q_emb = text_encoder.encode_single(queries[i]).unsqueeze(0)

                # Positive document
                pos_kg = build_kg_from_markdown(positives[i])
                pos_kg_embed = gat_encoder.encode_graph(pos_kg).unsqueeze(0)
                pos_d_emb = text_encoder.encode_single(positives[i][:512]).unsqueeze(0)

                meta = metadata[i]
                q_meta = torch.tensor(
                    [_meta_hash(meta["company_name"]), _meta_hash(meta["report_year"]), _meta_hash(meta["company_sector"])],
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)

                pos_cs = compute_constraint_score(pos_kg, version=contr_version, relative_tolerance=rel_tol)
                pos_cs_feats = scorer.build_constraint_features([pos_cs], device)

                pos_score = scorer(
                    q_emb,
                    pos_d_emb,
                    pos_kg_embed,
                    q_meta if use_entity_signal else None,
                    q_meta if use_entity_signal else None,
                    pos_cs_feats,
                )
                pos_scores_list.append(pos_score.squeeze(0))

                # Generate CHAP negative
                negs = sampler.sample(pos_kg, n_negatives=1)
                if not negs:
                    negs = [apply_chap_e(pos_kg)]

                for neg in negs:
                    neg_kg = build_kg_from_markdown(neg.table_md)
                    neg_kg_embed = gat_encoder.encode_graph(neg_kg).unsqueeze(0)
                    neg_d_emb = text_encoder.encode_single(neg.table_md[:512]).unsqueeze(0)

                    neg_cs = compute_constraint_score(neg_kg, version=contr_version, relative_tolerance=rel_tol)
                    neg_cs_feats = scorer.build_constraint_features([neg_cs], device)

                    neg_score = scorer(
                        q_emb,
                        neg_d_emb,
                        neg_kg_embed,
                        q_meta if use_entity_signal else None,
                        q_meta if use_entity_signal else None,
                        neg_cs_feats,
                    )
                    neg_scores_list.append(neg_score.squeeze(0))
                    violates_list.append(1.0 if neg.is_violated else 0.0)

            pos_scores_t = torch.stack(pos_scores_list)
            neg_scores_t = torch.stack(neg_scores_list)
            violates_t = torch.tensor(violates_list, dtype=torch.float32, device=device)

            # Align for triplet
            neg_for_triplet = neg_scores_t[:B] if len(neg_scores_t) >= B else neg_scores_t
            if len(neg_for_triplet) < B:
                pad = torch.zeros(B - len(neg_for_triplet), device=device)
                neg_for_triplet = torch.cat([neg_for_triplet, pad])

            losses = criterion(pos_scores_t, neg_for_triplet, violates_t)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(text_encoder.parameters()) + list(scorer.parameters()) + list(gat_encoder.parameters()),
                max_norm=1.0,
            )
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
