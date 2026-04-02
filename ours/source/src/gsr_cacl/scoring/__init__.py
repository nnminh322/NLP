"""Constraint-Aware Scoring module.

Implements the three-component joint scoring function from the paper:

  s(Q, D) = α · sim_text(Q, D)
           + β · sim_entity(Q, G_D)
           + γ · ConstraintScore(G_D, Q)

Where:
  - sim_text: standard dense retrieval similarity (BGE)
  - sim_entity: metadata matching score (company + year + sector)
  - ConstraintScore: ε-tolerance accounting consistency score

Constraint Score (ε-tolerance, differentiable):
  CS(G, Q) = (1/|Ec|) Σ_{(u,v)∈Ec} exp(−|ω_uv·v_u − v_v| / max(|v_v|, ε))

Reference:
  overall_idea.md §2.7 — Component 1c: Constraint-Aware Scoring
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsr_cacl.kg import ConstraintKG, KGEdge


# ----------------------------------------------------------------------
# Constraint Score
# ----------------------------------------------------------------------

@dataclass
class ConstraintScoringResult:
    """Results from constraint scoring."""
    constraint_score: float        # 0–1 (higher = more consistent)
    violated_count: int            # number of violated constraints
    total_count: int              # total accounting edges evaluated
    per_constraint_scores: list[float]


def compute_constraint_score(
    kg: ConstraintKG,
    epsilon: float = 1e-4,
) -> ConstraintScoringResult:
    """
    Compute constraint satisfaction score for a KG.

    For each accounting edge (u → v, ω):
        residual = |ω · v_u − v_v|
        denominator = max(|v_v|, ε)
        score = exp(− residual / denominator)

    Returns average score over all accounting edges.

    Args:
        kg:      ConstraintKG to score
        epsilon: tolerance for numerical stability
    Returns:
        ConstraintScoringResult with score and diagnostics
    """
    acc_edges = kg.accounting_edges
    if not acc_edges:
        return ConstraintScoringResult(
            constraint_score=1.0,  # no constraints = trivially satisfied
            violated_count=0,
            total_count=0,
            per_constraint_scores=[],
        )

    node_map = {n.id: n for n in kg.nodes}
    scores = []
    violated = 0

    for edge in acc_edges:
        src_node = node_map.get(edge.src)
        tgt_node = node_map.get(edge.tgt)

        if src_node is None or tgt_node is None:
            continue
        if src_node.value is None or tgt_node.value is None:
            continue

        # Compute constraint residual
        residual = abs(edge.omega * src_node.value - tgt_node.value)
        denom = max(abs(tgt_node.value), epsilon)
        edge_score = math.exp(-residual / denom)

        scores.append(edge_score)
        if edge_score < 0.5:   # threshold for "violated"
            violated += 1

    if not scores:
        return ConstraintScoringResult(
            constraint_score=1.0,
            violated_count=0,
            total_count=0,
            per_constraint_scores=[],
        )

    avg_score = sum(scores) / len(scores)
    return ConstraintScoringResult(
        constraint_score=avg_score,
        violated_count=violated,
        total_count=len(scores),
        per_constraint_scores=scores,
    )


# ----------------------------------------------------------------------
# Entity Matching Score
# ----------------------------------------------------------------------

def compute_entity_score(
    query_meta: dict,
    doc_meta: dict,
) -> float:
    """
    Compute entity matching score between query and document metadata.

    Scores based on:
      - company_name exact match (or fuzzy)
      - report_year exact match
      - company_sector overlap

    Returns float in [0, 1].
    """
    score = 0.0
    n_checks = 0

    # Company name
    q_company = str(query_meta.get("company_name", "")).lower().strip()
    d_company = str(doc_meta.get("company_name", "")).lower().strip()
    if q_company and d_company:
        n_checks += 1
        if q_company == d_company:
            score += 1.0
        elif q_company in d_company or d_company in q_company:
            score += 0.5

    # Report year
    q_year = str(query_meta.get("report_year", "")).strip()
    d_year = str(doc_meta.get("report_year", "")).strip()
    if q_year and d_year:
        n_checks += 1
        if q_year == d_year:
            score += 1.0
        elif abs(int(q_year or 0) - int(d_year or 0)) <= 1:
            score += 0.5

    # Sector
    q_sector = str(query_meta.get("company_sector", "")).lower().strip()
    d_sector = str(doc_meta.get("company_sector", "")).lower().strip()
    if q_sector and d_sector:
        n_checks += 1
        if q_sector == d_sector:
            score += 1.0
        elif q_sector in d_sector or d_sector in q_sector:
            score += 0.5

    if n_checks == 0:
        return 0.5  # neutral when no metadata available
    return score / n_checks


# ----------------------------------------------------------------------
# Joint Scoring Module (learnable)
# ----------------------------------------------------------------------

class JointScorer(nn.Module):
    """
    Learnable joint scorer for GSR retrieval.

    s(Q, D) = α · sim_text(Q, D)
             + β · sim_entity(Q, G_D)
             + γ · ConstraintScore(G_D, Q)

    All three components are combined with learnable scalar weights.
    """

    def __init__(
        self,
        text_embed_dim: int = 768,
        kg_embed_dim: int = 256,
        entity_feat_dim: int = 3,  # company + year + sector → 3 dims
        hidden_dim: int = 64,
    ):
        super().__init__()

        # Text similarity projection
        self.text_proj = nn.Linear(text_embed_dim + kg_embed_dim, hidden_dim)

        # Entity matching network
        self.entity_net = nn.Sequential(
            nn.Linear(entity_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Constraint score projection (input = 1 scalar + 2 diagnostic ints → hidden)
        self.constraint_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),  # score, violated_ratio, edge_count_norm
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Gating for text vs structured signals
        self.gate = nn.Sequential(
            nn.Linear(text_embed_dim, 1),
            nn.Sigmoid(),
        )

        # Learnable scalars (initialised to 1.0, constrained positive)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))   # text weight
        self.log_beta = nn.Parameter(torch.tensor(0.0))    # entity weight
        self.log_gamma = nn.Parameter(torch.tensor(-1.0))  # constraint weight

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)   # positive

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
        """
        Compute text + kg similarity score.
        Returns [B] scores.
        """
        # Concatenate text + kg embeddings
        combined = torch.cat([doc_text_embed, kg_embed], dim=-1)   # [B, text+kg]
        x = self.text_proj(combined)                               # [B, hidden]
        x = F.relu(x)
        sim = torch.cosine_similarity(query_text_embed, doc_text_embed, dim=-1)  # [B]

        # Gate: modulate text similarity by structural alignment
        gate_val = self.gate(query_text_embed).squeeze(-1)        # [B]
        return sim * (0.5 + 0.5 * gate_val)

    def forward_entity(
        self,
        query_meta: torch.Tensor,   # [B, 3]
        doc_meta: torch.Tensor,     # [B, 3]
    ) -> torch.Tensor:
        """Entity matching score [B]."""
        diff = torch.abs(query_meta - doc_meta)                   # [B, 3]
        # Soft similarity: high when differences are small
        return 1.0 - torch.tanh(diff.mean(dim=-1))                # [B]

    def forward_constraint(
        self,
        constraint_scores: list[ConstraintScoringResult],  # per-document
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Constraint scores as tensor [B].
        """
        scores = torch.zeros(batch_size, device=device)
        for i, cs in enumerate(constraint_scores):
            if cs.total_count == 0:
                scores[i] = 1.0
            else:
                scores[i] = cs.constraint_score
        return scores

    def forward(
        self,
        query_text_embed: torch.Tensor,   # [B, text_embed_dim]
        doc_text_embed: torch.Tensor,     # [B, text_embed_dim]
        kg_embed: torch.Tensor,            # [B, kg_embed_dim]
        query_meta: torch.Tensor,          # [B, 3]
        doc_meta: torch.Tensor,           # [B, 3]
        constraint_scores: list[ConstraintScoringResult],
    ) -> torch.Tensor:
        """
        Full joint scoring forward pass.

        Returns:
            final_scores: [B] combined relevance scores
        """
        B = query_text_embed.size(0)
        device = query_text_embed.device

        # Component 1: text similarity
        s_text = self.forward_text_sim(query_text_embed, doc_text_embed, kg_embed)

        # Component 2: entity matching
        s_entity = self.forward_entity(query_meta, doc_meta)

        # Component 3: constraint score
        s_constraint = self.forward_constraint(constraint_scores, B, device)

        # Weighted combination
        final = (
            self.alpha * s_text
            + self.beta * s_entity
            + self.gamma * s_constraint
        )
        return final
