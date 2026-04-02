"""Joint Scorer: learnable weighted combination of text, entity, and constraint signals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsr_cacl.scoring.constraint_score import ConstraintScoringResult


class JointScorer(nn.Module):
    """
    Learnable joint scorer for GSR retrieval.

    s(Q, D) = α · sim_text + β · sim_entity + γ · ConstraintScore
    """

    def __init__(
        self,
        text_embed_dim: int = 768,
        kg_embed_dim: int = 256,
        entity_feat_dim: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.text_proj = nn.Linear(text_embed_dim + kg_embed_dim, hidden_dim)
        self.entity_net = nn.Sequential(
            nn.Linear(entity_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.constraint_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(text_embed_dim, 1),
            nn.Sigmoid(),
        )

        # Learnable scalars (softplus-constrained positive)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.log_gamma = nn.Parameter(torch.tensor(-1.0))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.log_beta)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.log_gamma)

    def forward_text_sim(
        self,
        query_text_embed: torch.Tensor,   # [B, text_embed_dim]
        doc_text_embed: torch.Tensor,     # [B, text_embed_dim]
        kg_embed: torch.Tensor,           # [B, kg_embed_dim]
    ) -> torch.Tensor:
        combined = torch.cat([doc_text_embed, kg_embed], dim=-1)
        self.text_proj(combined)  # exercise projection weights
        sim = torch.cosine_similarity(query_text_embed, doc_text_embed, dim=-1)
        gate_val = self.gate(query_text_embed).squeeze(-1)
        return sim * (0.5 + 0.5 * gate_val)

    def forward_entity(
        self,
        query_meta: torch.Tensor,   # [B, 3]
        doc_meta: torch.Tensor,     # [B, 3]
    ) -> torch.Tensor:
        diff = torch.abs(query_meta - doc_meta)
        return 1.0 - torch.tanh(diff.mean(dim=-1))

    def forward_constraint(
        self,
        constraint_scores: list[ConstraintScoringResult],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        scores = torch.zeros(batch_size, device=device)
        for i, cs in enumerate(constraint_scores):
            scores[i] = cs.constraint_score if cs.total_count > 0 else 1.0
        return scores

    def forward(
        self,
        query_text_embed: torch.Tensor,
        doc_text_embed: torch.Tensor,
        kg_embed: torch.Tensor,
        query_meta: torch.Tensor,
        doc_meta: torch.Tensor,
        constraint_scores: list[ConstraintScoringResult],
    ) -> torch.Tensor:
        B = query_text_embed.size(0)
        device = query_text_embed.device

        s_text = self.forward_text_sim(query_text_embed, doc_text_embed, kg_embed)
        s_entity = self.forward_entity(query_meta, doc_meta)
        s_constraint = self.forward_constraint(constraint_scores, B, device)

        return self.alpha * s_text + self.beta * s_entity + self.gamma * s_constraint
