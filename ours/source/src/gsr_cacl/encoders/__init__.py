"""GAT Encoder: Graph Attention Network for financial table cells.

Edge-aware message passing:
  h_v^{(l+1)} = LeakyReLU( Σ_{u∈N(v)} α_{vu} · ω_{vu} · W^{(l)} · h_u^{(l)} )

Where:
  - ω ∈ {+1, −1, 0}: accounting edge weight (+1 additive, −1 subtractive, 0 positional)
  - α_{vu}: attention coefficient (softmax over neighbours)
  - W^{(l)}: learnable weight matrix

Node embedding:
  h_v^{(0)} = [BGE(cell_text); PosEnc(row_idx); PosEnc(col_idx)]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsr_cacl.kg import ConstraintKG, KGNode, KGEdge


# ----------------------------------------------------------------------
# Positional Encoding
# ----------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for row/column indices."""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: LongTensor of shape [batch_size, seq_len] (row/col indices)
        Returns:
            positional embeddings of shape [batch_size, seq_len, d_model]
        """
        return self.pe[:, indices, :]


# ----------------------------------------------------------------------
# GAT Layer
# ----------------------------------------------------------------------

class GATLayer(nn.Module):
    """Single Graph Attention Layer with edge-aware message passing."""

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
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"

        # Query-Key-Value projections
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.W_o = nn.Linear(out_features, out_features)

        # Edge weight projection (ω ∈ {+1, −1, 0})
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
        h: torch.Tensor,       # [V, in_features]
        edge_index: torch.Tensor,   # [2, E]
        edge_weight: torch.Tensor,   # [E]  omega values (+1, -1, 0)
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:           node features [V, in_features]
            edge_index:  [2, E] source→target edges
            edge_weight: [E] omega values
            return_attention: also return attention weights
        Returns:
            out: [V, out_features]
            (optional) attn: [V, H] averaged attention coefficients
        """
        V = h.size(0)
        H = self.num_heads
        F_h = self.head_features

        # Project to Q, K, V
        Q = self.W_q(h).view(V, H, F_h)        # [V, H, F_h]
        K = self.W_k(h).view(V, H, F_h)
        V_out = self.W_v(h).view(V, H, F_h)

        # Edge weight → attention bias [E, H]
        edge_weight_expanded = edge_weight.unsqueeze(-1)   # [E, 1]
        edge_bias = self.edge_proj(edge_weight_expanded)  # [E, H]
        edge_bias = F.leaky_relu(edge_bias, negative_slope=self.leakyrate)

        # Compute attention scores per edge
        # score(e_{vu}) = (Q_u · K_v) / sqrt(F_h) + edge_bias
        src_idx = edge_index[0]   # source nodes
        tgt_idx = edge_index[1]   # target nodes

        # Q for source, K for target
        q_src = Q[src_idx]     # [E, H, F_h]
        k_tgt = K[tgt_idx]     # [E, H, F_h]

        attn_raw = torch.sum(q_src * k_tgt, dim=-1) / math.sqrt(F_h)  # [E, H]
        attn_raw = attn_raw + edge_bias                                      # [E, H]

        # --- Neighbourhood softmax ---
        # For each target node v, softmax over incoming neighbours u
        # attn_vu = exp(attn_raw) / Σ_{w∈N(v)} exp(attn_raw)
        attn_exp = torch.exp(attn_raw)   # [E, H]

        # Build softmax denominator per target
        target_deg = torch.zeros(V, H, device=h.device)
        target_deg.index_add_(0, tgt_idx, attn_exp)

        # Avoid division by zero
        target_deg = target_deg + 1e-16

        # Normalise
        attn_norm = attn_exp / target_deg[tgt_idx]   # [E, H]
        attn_norm = self.dropout(attn_norm)

        # --- Aggregate values ---
        alpha_vu = attn_norm.unsqueeze(-1)           # [E, H, 1]
        w_v = V_out[src_idx]                          # [E, H, F_h]
        msg = w_v * alpha_vu * edge_weight_expanded.unsqueeze(-1)  # [E, H, F_h]

        # Scatter add: sum messages grouped by target
        out = torch.zeros(V, H, F_h, device=h.device, dtype=h.dtype)
        out.index_add_(0, tgt_idx, msg)
        out = out.view(V, H * F_h)                   # [V, out_features]

        out = self.W_o(out)
        out = F.leaky_relu(out, negative_slope=self.leakyrate)

        if return_attention:
            attn_avg = attn_norm.mean(dim=1)         # [E] → average over heads
            return out, attn_avg
        return out


# ----------------------------------------------------------------------
# Full GAT Encoder
# ----------------------------------------------------------------------

class GATEncoder(nn.Module):
    """
    Two-layer GAT encoder for financial table cells.

    Architecture:
        Input:  [BGE(cell_text) ⊕ PosEnc(row) ⊕ PosEnc(col)]
        Layer 1: GAT → LeakyReLU
        Layer 2: GAT → LeakyReLU
        Output: node embeddings [V, hidden_dim]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Positional encodings
        self.row_pe = SinusoidalPositionalEncoding(embed_dim // 4, max_len=256)
        self.col_pe = SinusoidalPositionalEncoding(embed_dim // 4, max_len=64)

        # Input projection: BGE_embed ⊕ row_pe ⊕ col_pe → hidden_dim
        in_proj_dim = embed_dim + 2 * (embed_dim // 4)
        self.input_proj = nn.Sequential(
            nn.Linear(in_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim if i > 0 else hidden_dim
            out_d = hidden_dim
            self.gat_layers.append(
                GATLayer(in_d, out_d, num_heads=num_heads, dropout=dropout)
            )

    def forward(self, kg: ConstraintKG) -> torch.Tensor:
        """
        Encode all nodes of a knowledge graph.

        Args:
            kg: ConstraintKG with pre-computed nodes
        Returns:
            node_embeddings: [V, hidden_dim]
        """
        if not kg.nodes:
            return torch.empty(0, self.hidden_dim)

        # --- Build node features ---
        # h_0 = [cell_embed ⊕ row_pos ⊕ col_pos]
        V = len(kg.nodes)
        device = next(self.parameters()).device

        # Placeholder: cell embeddings would come from BGE in practice.
        # Here we create a zero tensor as placeholder — replace with real BGE
        # embeddings in the full pipeline.
        cell_embeds = torch.zeros(V, self.embed_dim, device=device)

        row_indices = torch.tensor(
            [n.row_idx for n in kg.nodes], dtype=torch.long, device=device
        )
        col_indices = torch.tensor(
            [n.col_idx for n in kg.nodes], dtype=torch.long, device=device
        )

        row_pos = self.row_pe(row_indices.unsqueeze(0)).squeeze(0)   # [V, embed_dim/4]
        col_pos = self.col_pe(col_indices.unsqueeze(0)).squeeze(0)   # [V, embed_dim/4]

        h = torch.cat([cell_embeds, row_pos, col_pos], dim=-1)     # [V, embed_dim + embed_dim/2]
        h = self.input_proj(h)                                      # [V, hidden_dim]

        # --- Message passing ---
        edge_index = self._build_edge_index(kg, device)
        edge_weight = self._build_edge_weight(kg, device)

        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_weight)

        return h  # [V, hidden_dim]

    def _build_edge_index(self, kg: ConstraintKG, device: torch.device) -> torch.Tensor:
        """Convert KGEdges to [2, E] tensor."""
        if not kg.edges:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        src_ids = [e.src for e in kg.edges]
        tgt_ids = [e.tgt for e in kg.edges]

        # Build node_id → index lookup
        node_id_to_idx = {n.id: i for i, n in enumerate(kg.nodes)}

        src_idx = torch.tensor(
            [node_id_to_idx.get(sid, 0) for sid in src_ids], dtype=torch.long, device=device
        )
        tgt_idx = torch.tensor(
            [node_id_to_idx.get(tid, 0) for tid in tgt_ids], dtype=torch.long, device=device
        )
        return torch.stack([src_idx, tgt_idx], dim=0)

    def _build_edge_weight(self, kg: ConstraintKG, device: torch.device) -> torch.Tensor:
        """Extract omega values as edge weights."""
        if not kg.edges:
            return torch.empty(0, dtype=torch.float32, device=device)
        return torch.tensor(
            [float(e.omega) for e in kg.edges],
            dtype=torch.float32,
            device=device,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim
