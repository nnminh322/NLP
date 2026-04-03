"""Joint Scorer: learnable weighted combination of text, entity, and constraint signals.

Architecture (architecture.md §4):
    s(Q, D) = α · sim_text(Q, D, KG) + β · sim_entity(Q, D) + γ · CS(G_D)

    where sim_text fuses text cosine similarity with KG-enriched projection,
    α/β/γ are softplus-constrained learnable scalars.
"""

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
        self.text_embed_dim = text_embed_dim
        self.kg_embed_dim = kg_embed_dim

        # KG-enriched text projection: [doc_text ⊕ kg_embed] → hidden → score adjustment
        self.text_kg_proj = nn.Sequential(
            nn.Linear(text_embed_dim + kg_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # Query gate: modulates text similarity based on query complexity
        self.gate = nn.Sequential(
            nn.Linear(text_embed_dim, 1),
            nn.Sigmoid(),
        )

        # Constraint score projection: [raw_cs, violated_ratio, edge_count_norm] → refined score
        self.constraint_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
        """Text similarity enriched by KG structural info.

        sim = cosine(q, d) * gate(q) + kg_adjustment(d, kg)
        """
        # Base cosine similarity
        sim = torch.cosine_similarity(query_text_embed, doc_text_embed, dim=-1)  # [B]

        # Query-dependent gating
        gate_val = self.gate(query_text_embed).squeeze(-1)  # [B]
        gated_sim = sim * (0.5 + 0.5 * gate_val)

        # KG-enriched adjustment: projects [doc_text ⊕ kg_embed] to a small correction
        combined = torch.cat([doc_text_embed, kg_embed], dim=-1)  # [B, text+kg]
        kg_adjustment = self.text_kg_proj(combined).squeeze(-1)  # [B], range [-1, 1]

        # Final text score: gated cosine + small KG adjustment (max ±0.2)
        return gated_sim + 0.2 * kg_adjustment

    def forward_entity(
        self,
        query_meta: torch.Tensor,   # [B, 3]  (company, year, sector as float)
        doc_meta: torch.Tensor,     # [B, 3]
    ) -> torch.Tensor:
        """Entity matching score: exact match fraction."""
        match = (query_meta == doc_meta).float()  # [B, 3]
        return match.mean(dim=-1)  # [B]

    def forward_constraint(
        self,
        constraint_features: torch.Tensor,  # [B, 3]: [raw_cs, violated_ratio, edge_count_norm]
    ) -> torch.Tensor:
        """Learnable constraint score refinement.

        Args:
            constraint_features: [B, 3] tensor with:
                - col 0: raw constraint score (0-1)
                - col 1: violated ratio (0-1)
                - col 2: normalized edge count (0-1)
        """
        return self.constraint_proj(constraint_features).squeeze(-1)  # [B]

    @staticmethod
    def build_constraint_features(
        constraint_results: list[ConstraintScoringResult],
        device: torch.device,
    ) -> torch.Tensor:
        """Convert ConstraintScoringResult list to [B, 3] tensor for forward_constraint."""
        feats = []
        for cs in constraint_results:
            raw_score = cs.constraint_score if cs.total_count > 0 else 1.0
            violated_ratio = cs.violated_count / max(cs.total_count, 1)
            edge_norm = min(cs.total_count / 20.0, 1.0)  # normalize by ~max expected edges
            feats.append([raw_score, violated_ratio, edge_norm])
        return torch.tensor(feats, dtype=torch.float32, device=device)

    def forward(
        self,
        query_text_embed: torch.Tensor,
        doc_text_embed: torch.Tensor,
        kg_embed: torch.Tensor,
        query_meta: torch.Tensor,
        doc_meta: torch.Tensor,
        constraint_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint score s(Q, D) = α·s_text + β·s_entity + γ·s_constraint.

        Args:
            query_text_embed: [B, text_embed_dim]
            doc_text_embed: [B, text_embed_dim]
            kg_embed: [B, kg_embed_dim]
            query_meta: [B, 3]
            doc_meta: [B, 3]
            constraint_features: [B, 3]
        """
        s_text = self.forward_text_sim(query_text_embed, doc_text_embed, kg_embed)
        s_entity = self.forward_entity(query_meta, doc_meta)
        s_constraint = self.forward_constraint(constraint_features)

        return self.alpha * s_text + self.beta * s_entity + self.gamma * s_constraint

    def score_single(
        self,
        query_text_embed: torch.Tensor,   # [text_embed_dim]
        doc_text_embed: torch.Tensor,     # [text_embed_dim]
        kg_embed: torch.Tensor,           # [kg_embed_dim]
        entity_score: float,
        constraint_result: ConstraintScoringResult,
    ) -> float:
        """Score a single (query, document) pair. Used in inference."""
        device = next(self.parameters()).device

        q = query_text_embed.unsqueeze(0).to(device)
        d = doc_text_embed.unsqueeze(0).to(device)
        kg = kg_embed.unsqueeze(0).to(device)

        # Text similarity (enriched by KG)
        s_text = self.forward_text_sim(q, d, kg).item()

        # Constraint score (pre-computed)
        cs_feats = self.build_constraint_features([constraint_result], device)
        s_constraint = self.forward_constraint(cs_feats).item()

        # Weighted combination
        return float(
            self.alpha.item() * s_text
            + self.beta.item() * entity_score
            + self.gamma.item() * s_constraint
        )
