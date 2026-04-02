"""GAT Layer: single Graph Attention layer with edge-aware message passing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single Graph Attention Layer with edge-aware message passing.

    h_v^{(l+1)} = LeakyReLU( Σ_{u∈N(v)} α_{vu} · ω_{vu} · W · h_u^{(l)} )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        leakyrate: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_features = out_features // num_heads
        assert out_features % num_heads == 0

        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.W_o = nn.Linear(out_features, out_features)
        self.edge_proj = nn.Linear(1, num_heads)

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
        h: torch.Tensor,           # [V, in_features]
        edge_index: torch.Tensor,   # [2, E]
        edge_weight: torch.Tensor,  # [E]
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        V = h.size(0)
        H = self.num_heads
        F_h = self.head_features

        Q = self.W_q(h).view(V, H, F_h)
        K = self.W_k(h).view(V, H, F_h)
        V_out = self.W_v(h).view(V, H, F_h)

        edge_weight_expanded = edge_weight.unsqueeze(-1)   # [E, 1]
        edge_bias = self.edge_proj(edge_weight_expanded)   # [E, H]
        edge_bias = F.leaky_relu(edge_bias, negative_slope=self.leakyrate)

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        q_src = Q[src_idx]     # [E, H, F_h]
        k_tgt = K[tgt_idx]     # [E, H, F_h]

        attn_raw = torch.sum(q_src * k_tgt, dim=-1) / math.sqrt(F_h)  # [E, H]
        attn_raw = attn_raw + edge_bias

        attn_exp = torch.exp(attn_raw)
        target_deg = torch.zeros(V, H, device=h.device)
        target_deg.index_add_(0, tgt_idx, attn_exp)
        target_deg = target_deg + 1e-16

        attn_norm = attn_exp / target_deg[tgt_idx]
        attn_norm = self.dropout(attn_norm)

        alpha_vu = attn_norm.unsqueeze(-1)                    # [E, H, 1]
        w_v = V_out[src_idx]                                   # [E, H, F_h]
        msg = w_v * alpha_vu * edge_weight_expanded.unsqueeze(-1)

        out = torch.zeros(V, H, F_h, device=h.device, dtype=h.dtype)
        out.index_add_(0, tgt_idx, msg)
        out = out.view(V, H * F_h)

        out = self.W_o(out)
        out = F.leaky_relu(out, negative_slope=self.leakyrate)

        if return_attention:
            attn_avg = attn_norm.mean(dim=1)
            return out, attn_avg
        return out
