"""GAT Layer: single Graph Attention layer with edge-aware message passing.

Implements the Edge-Aware Multi-Head Attention from §4.3 of the GSR-CACL proposal.

Architecture (Eq.6–8 in proposal):
    e_uv^{(k)} = <W_q h_u, W_k h_v> / √d_k
                  + Proj(ω_uv)                    ← edge-aware bias
                  + EntitySim(e_u, e_v)           ← entity-aware attention

    α_uv^{(k)} = softmax_j exp(e_uj^{(k)}) / Σ_j exp(e_uj^{(k)})

    h_v^{(l+1)} = W_o [ ‖_k Σ_{u∈N(v)} α_uv^{(k)} · ω_uv · W_v h_u^{(l)} ]
                  + h_v^{(l)}                      ← residual connection

Key differences from standard GAT:
    1. Edge-aware bias: ω_uv (accounting constraint weight) modifies attention
    2. Entity similarity: entity embeddings guide attention between related cells
    3. Residual connection: original h_v added to output (helps gradient flow)

Why edge-aware attention matters:
    In a Balance Sheet table, "Total Assets" should attend strongly to
    "Current Assets" and "Non-Current Assets" (ω=+1 additive edge),
    but NOT to "Revenue" (no edge in this template).
    Without edge-aware bias, standard GAT treats all neighbors equally.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single Graph Attention Layer with edge-aware message passing.

    Implements Eq.6–8 from the proposal:
        e_uv = <Q_u, K_v> / √d_k + Proj(ω_uv) + EntitySim(e_u, e_v)
        α_uv = softmax_j exp(e_uj) / Σ_j exp(e_uj)
        h_v^{(l+1)} = W_o [ ⊕_k Σ α_uv · ω_uv · W_v h_u ] + h_v^{(l)}  (residual)

    Features:
        - Multi-head attention (H heads, F_h features each)
        - Edge-aware bias via learnable projection of ω ∈ {+1, −1, 0}
        - Optional entity similarity term (set e_u=None to disable)
        - Residual connection h_v^{(l)} → h_v^{(l+1)} (from proposal Eq.8)
        - Pre-LayerNorm architecture for better training stability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        leakyrate: float = 0.2,
        add_residual: bool = True,
        add_entity_sim: bool = False,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
            leakyrate: LeakyReLU negative slope
            add_residual: If True, add residual connection h_v^{(l)} → h_v^{(l+1)}
            add_entity_sim: If True, add entity similarity term to attention
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_features = out_features // num_heads
        self.add_residual = add_residual
        self.add_entity_sim = add_entity_sim

        assert out_features % num_heads == 0, (
            f"out_features={out_features} must be divisible by num_heads={num_heads}"
        )

        # Query, Key, Value projections
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.W_o = nn.Linear(out_features, out_features)

        # Edge projection: ω ∈ {+1, −1, 0} → bias per head
        # ω encodes accounting semantics: +1=additive, −1=subtractive, 0=positional
        self.edge_proj = nn.Linear(1, num_heads)

        # FIX ISSUE 8: EntitySim scale — learnable instead of fixed 0.5
        # Proposal Eq.6 doesn't specify scale; making it learnable lets the model
        # adapt how much entity similarity should influence attention.
        self.log_entity_sim_scale = nn.Parameter(torch.tensor(0.0))  # σ^{-1}(0.0) ≈ 0.5

        self.norm = nn.LayerNorm(out_features)

        self.dropout = nn.Dropout(dropout)
        self.leakyrate = leakyrate
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,            # [V, in_features]
        edge_index: torch.Tensor,    # [2, E]
        edge_weight: torch.Tensor,   # [E] — ω values
        entity_embeddings: torch.Tensor | None = None,  # [V, d_e] entity embeds
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with edge-aware attention + optional entity similarity.

        Args:
            h: Node features [V, in_features]
            edge_index: [2, E] source→target edges
            edge_weight: [E] omega values (+1/−1/0)
            entity_embeddings: [V, d_e] entity embeddings for EntitySim term
            return_attention: If True, return attention weights too
        Returns:
            Updated node features [V, out_features]
            ( Optionally: attention weights [V, H] )
        """
        V = h.size(0)
        H = self.num_heads
        F_h = self.head_features

        # Step 1: Project to Q, K, V
        Q = self.W_q(h).view(V, H, F_h)   # [V, H, F_h]
        K = self.W_k(h).view(V, H, F_h)
        V_out = self.W_v(h).view(V, H, F_h)

        # Step 2: Edge-aware bias (Eq.6 component 1)
        # ω is the accounting constraint weight: +1=additive, −1=subtractive, 0=positional
        edge_weight_expanded = edge_weight.unsqueeze(-1)   # [E, 1]
        edge_bias = self.edge_proj(edge_weight_expanded)   # [E, H]
        edge_bias = F.leaky_relu(edge_bias, negative_slope=self.leakyrate)

        # Step 3: Entity similarity term (Eq.6 component 2 — optional)
        # This term makes cells with the same entity attend to each other
        if self.add_entity_sim and entity_embeddings is not None:
            # Entity similarity: cosine similarity between entity embeddings
            # For normalized entity embeddings, cos ≈ dot product
            entity_sim = torch.matmul(
                F.normalize(entity_embeddings, p=2, dim=-1),
                F.normalize(entity_embeddings, p=2, dim=-1).T,
            )  # [V, V]
            # Map entity similarity to edge attention bias
            src_idx = edge_index[0]
            tgt_idx = edge_index[1]
            ent_bias = entity_sim[src_idx, tgt_idx].unsqueeze(-1)  # [E, 1]
            # FIX ISSUE 8: Use learnable scale instead of fixed 0.5
            scale = F.softplus(self.log_entity_sim_scale)  # positive scale
            ent_bias = F.leaky_relu(ent_bias * scale, negative_slope=self.leakyrate)
        else:
            ent_bias = torch.zeros_like(edge_bias)

        # Step 4: Compute attention scores (Eq.6)
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        q_src = Q[src_idx]   # [E, H, F_h]
        k_tgt = K[tgt_idx]   # [E, H, F_h]

        attn_raw = torch.sum(q_src * k_tgt, dim=-1) / math.sqrt(F_h)  # [E, H]
        attn_raw = attn_raw + edge_bias + ent_bias

        # Step 5: Normalize attention scores (Eq.7 — softmax)
        attn_exp = torch.exp(attn_raw)

        # Aggregate at target nodes: Σ_j exp(e_ij) for each target
        target_deg = torch.zeros(V, H, device=h.device)
        target_deg.index_add_(0, tgt_idx, attn_exp)
        target_deg = target_deg + 1e-16

        attn_norm = attn_exp / target_deg[tgt_idx]  # [E, H]
        attn_norm = self.dropout(attn_norm)

        # Step 6: Message passing (Eq.8: α · ω · W_v h_u)
        alpha_vu = attn_norm.unsqueeze(-1)    # [E, H, 1]
        w_vu = V_out[src_idx]                  # [E, H, F_h]
        # ω is encoded in edge_weight; multiplied into message
        msg = w_vu * alpha_vu * edge_weight_expanded.unsqueeze(-1)

        # Aggregate messages at target nodes
        out = torch.zeros(V, H, F_h, device=h.device, dtype=h.dtype)
        out.index_add_(0, tgt_idx, msg)
        out = out.view(V, H * F_h)

        # Step 7: Output projection + residual connection (Eq.8: + h_v^{(l)})
        out = self.W_o(out)
        if self.add_residual:
            out = out + h  # Residual: h_v^{(l)} → h_v^{(l+1)}

        out = self.norm(out)
        out = F.leaky_relu(out, negative_slope=self.leakyrate)

        if return_attention:
            attn_avg = attn_norm.mean(dim=1)  # [V] average over heads
            return out, attn_avg
        return out
