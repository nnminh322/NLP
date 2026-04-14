"""Joint Scorer: learnable weighted combination of text, entity, and constraint signals.

Implements §4.6 of the GSR-CACL proposal:
    s(Q, D) = α · sim_text(Q, D, KG) + β · s_entity(Q, D) + γ · CS(G_D)

Key improvement over original code:
    s_entity = cos(e_Q, e_D) from learned EntityEncoder
    (was: exact string match with no gradient)

Architecture (Eq.15 in proposal):
    s(Q, D) = α · s_text + β · s_entity + γ · CS(G_D)

where:
    s_text(Q, D) = cos(q_text, d_text) × gate(q_text) + kg_adjustment
    s_entity(Q, D) = cos(e_Q, e_D)        ← learned (was: exact match)
    CS(G_D) = (1/|E_c|) Σ exp(-|ω·v_u - v_v| / max(|v_v|, ε))
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

    Key features:
        - Text similarity enriched by KG structural info (Eq.16)
        - Entity similarity from learned EntityEncoder (Eq.17) ← NEW
        - Constraint score from KG scoring (Eq.10)
        - Learnable α/β/γ weights via softplus (Eq.15)

    Training:
        JointScorer receives gradients from both triplet loss (text/constraint)
        AND entity SupCon loss (entity similarity). This is why α, β, γ
        are learnable — they adapt to the relative importance of each signal.
    """

    def __init__(
        self,
        text_embed_dim: int = 768,
        kg_embed_dim: int = 256,
        entity_embed_dim: int = 256,  # NEW: entity embedding dimension
        hidden_dim: int = 64,
    ):
        """
        Args:
            text_embed_dim: BGE text embedding dimension (d = 1024 for bge-large)
            kg_embed_dim: GAT graph embedding dimension (h_dim = 256)
            entity_embed_dim: Entity embedding dimension (d_e = 256) ← NEW
            hidden_dim: Hidden dimension for intermediate projections
        """
        super().__init__()
        self.text_embed_dim = text_embed_dim
        self.kg_embed_dim = kg_embed_dim
        self.entity_embed_dim = entity_embed_dim

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
        self.log_alpha = nn.Parameter(torch.tensor(0.0))   # α
        self.log_beta = nn.Parameter(torch.tensor(0.0))    # β
        self.log_gamma = nn.Parameter(torch.tensor(-1.0))  # γ

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

        sim = cosine(q, d) × gate(q) + kg_adjustment
        """
        # Base cosine similarity
        sim = torch.cosine_similarity(query_text_embed, doc_text_embed, dim=-1)  # [B]

        # Query-dependent gating
        gate_val = self.gate(query_text_embed).squeeze(-1)  # [B]
        gated_sim = sim * (0.5 + 0.5 * gate_val)

        # KG-enriched adjustment
        combined = torch.cat([doc_text_embed, kg_embed], dim=-1)  # [B, text+kg]
        kg_adjustment = self.text_kg_proj(combined).squeeze(-1)  # [B]

        return gated_sim + 0.2 * kg_adjustment

    def forward_entity_sim(
        self,
        query_entity_embed: torch.Tensor,  # [B, entity_embed_dim]
        doc_entity_embed: torch.Tensor,    # [B, entity_embed_dim]
    ) -> torch.Tensor:
        """
        Entity similarity score from learned EntityEncoder (Eq.17 in proposal).

        s_entity = cos(e_Q, e_D)

        This replaces the old exact-match scoring.
        Gradients flow back through the EntityEncoder → shared BGE backbone.

        Why cosine similarity?
            Entity embeddings are L2-normalized → dot product = cosine similarity.
            This is equivalent to the formulation in Eq.17.
        """
        return torch.cosine_similarity(
            query_entity_embed, doc_entity_embed, dim=-1
        )  # [B]

    def forward_entity_exact(
        self,
        query_meta: torch.Tensor,   # [B, 3] — (company_hash, year_hash, sector_hash)
        doc_meta: torch.Tensor,     # [B, 3]
    ) -> torch.Tensor:
        """
        Legacy entity matching: exact match fraction.

        DEPRECATED: Use forward_entity_sim with learned embeddings instead.
        Kept for backwards compatibility with inference code.
        """
        match = (query_meta == doc_meta).float()
        return match.mean(dim=-1)

    def forward_constraint(
        self,
        constraint_features: torch.Tensor,  # [B, 3]: [raw_cs, violated_ratio, edge_count_norm]
    ) -> torch.Tensor:
        """Learnable constraint score refinement."""
        return self.constraint_proj(constraint_features).squeeze(-1)

    @staticmethod
    def build_constraint_features(
        constraint_results: list[ConstraintScoringResult],
        device: torch.device,
    ) -> torch.Tensor:
        """Convert ConstraintScoringResult list to [B, 3] tensor."""
        feats = []
        for cs in constraint_results:
            raw_score = cs.constraint_score if cs.total_count > 0 else 1.0
            violated_ratio = cs.violated_count / max(cs.total_count, 1)
            edge_norm = min(cs.total_count / 20.0, 1.0)
            feats.append([raw_score, violated_ratio, edge_norm])
        return torch.tensor(feats, dtype=torch.float32, device=device)

    def forward(
        self,
        query_text_embed: torch.Tensor,     # [B, text_embed_dim]
        doc_text_embed: torch.Tensor,        # [B, text_embed_dim]
        kg_embed: torch.Tensor,              # [B, kg_embed_dim]
        query_entity_embed: torch.Tensor,    # [B, entity_embed_dim] ← NEW
        doc_entity_embed: torch.Tensor,     # [B, entity_embed_dim] ← NEW
        constraint_features: torch.Tensor,     # [B, 3]
    ) -> torch.Tensor:
        """
        Full joint scoring (Eq.15 in proposal).

        s(Q, D) = α · s_text + β · s_entity + γ · s_constraint

        Args:
            query_text_embed: [B, text_embed_dim]
            doc_text_embed: [B, text_embed_dim]
            kg_embed: [B, kg_embed_dim]
            query_entity_embed: [B, entity_embed_dim] ← NEW
            doc_entity_embed: [B, entity_embed_dim] ← NEW
            constraint_features: [B, 3]
        """
        s_text = self.forward_text_sim(query_text_embed, doc_text_embed, kg_embed)
        s_entity = self.forward_entity_sim(query_entity_embed, doc_entity_embed)
        s_constraint = self.forward_constraint(constraint_features)

        return self.alpha * s_text + self.beta * s_entity + self.gamma * s_constraint

    def score_single(
        self,
        query_text_embed: torch.Tensor,   # [text_embed_dim]
        doc_text_embed: torch.Tensor,     # [text_embed_dim]
        kg_embed: torch.Tensor,           # [kg_embed_dim]
        entity_score: float | torch.Tensor,  # float (exact) or learned tensor
        constraint_result: ConstraintScoringResult,
    ) -> float:
        """
        Score a single (query, document) pair. Used in inference.

        For backwards compatibility, accepts a float entity_score (from exact match).
        If entity embeddings are available, use score_single_learned() instead.
        """
        device = next(self.parameters()).device

        q = query_text_embed.unsqueeze(0).to(device)
        d = doc_text_embed.unsqueeze(0).to(device)
        kg = kg_embed.unsqueeze(0).to(device)

        # Text similarity
        s_text = self.forward_text_sim(q, d, kg).item()

        # Constraint score
        cs_feats = self.build_constraint_features([constraint_result], device)
        s_constraint = self.forward_constraint(cs_feats).item()

        # Entity score (backwards-compatible: exact float)
        if isinstance(entity_score, torch.Tensor):
            s_entity = entity_score.item()
        else:
            s_entity = float(entity_score)

        return float(
            self.alpha.item() * s_text
            + self.beta.item() * s_entity
            + self.gamma.item() * s_constraint
        )

    def score_single_learned(
        self,
        query_text_embed: torch.Tensor,
        doc_text_embed: torch.Tensor,
        kg_embed: torch.Tensor,
        query_entity_embed: torch.Tensor,
        doc_entity_embed: torch.Tensor,
        constraint_result: ConstraintScoringResult,
    ) -> float:
        """
        Score a single pair using learned entity embeddings.

        Preferred method during/after EntitySupCon training.
        """
        device = next(self.parameters()).device

        q_t = query_text_embed.unsqueeze(0).to(device)
        d_t = doc_text_embed.unsqueeze(0).to(device)
        kg = kg_embed.unsqueeze(0).to(device)
        q_e = query_entity_embed.unsqueeze(0).to(device)
        d_e = doc_entity_embed.unsqueeze(0).to(device)

        cs_feats = self.build_constraint_features([constraint_result], device)

        score_tensor = self.forward(q_t, d_t, kg, q_e, d_e, cs_feats)
        return score_tensor.item()
