"""Full GAT Encoder for financial table cells."""

from __future__ import annotations

import torch
import torch.nn as nn

from gsr_cacl.kg.data_structures import ConstraintKG
from gsr_cacl.encoders.positional import SinusoidalPositionalEncoding
from gsr_cacl.encoders.gat_layer import GATLayer


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

        self.row_pe = SinusoidalPositionalEncoding(embed_dim // 4, max_len=256)
        self.col_pe = SinusoidalPositionalEncoding(embed_dim // 4, max_len=64)

        in_proj_dim = embed_dim + 2 * (embed_dim // 4)
        self.input_proj = nn.Sequential(
            nn.Linear(in_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

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

        V = len(kg.nodes)
        device = next(self.parameters()).device

        # Placeholder cell embeddings — replace with real BGE in full pipeline
        cell_embeds = torch.zeros(V, self.embed_dim, device=device)

        row_indices = torch.tensor(
            [n.row_idx for n in kg.nodes], dtype=torch.long, device=device
        )
        col_indices = torch.tensor(
            [n.col_idx for n in kg.nodes], dtype=torch.long, device=device
        )

        row_pos = self.row_pe(row_indices.unsqueeze(0)).squeeze(0)
        col_pos = self.col_pe(col_indices.unsqueeze(0)).squeeze(0)

        h = torch.cat([cell_embeds, row_pos, col_pos], dim=-1)
        h = self.input_proj(h)

        edge_index = self._build_edge_index(kg, device)
        edge_weight = self._build_edge_weight(kg, device)

        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_weight)

        return h

    def _build_edge_index(self, kg: ConstraintKG, device: torch.device) -> torch.Tensor:
        if not kg.edges:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        node_id_to_idx = {n.id: i for i, n in enumerate(kg.nodes)}
        src_idx = torch.tensor(
            [node_id_to_idx.get(e.src, 0) for e in kg.edges],
            dtype=torch.long, device=device,
        )
        tgt_idx = torch.tensor(
            [node_id_to_idx.get(e.tgt, 0) for e in kg.edges],
            dtype=torch.long, device=device,
        )
        return torch.stack([src_idx, tgt_idx], dim=0)

    def _build_edge_weight(self, kg: ConstraintKG, device: torch.device) -> torch.Tensor:
        if not kg.edges:
            return torch.empty(0, dtype=torch.float32, device=device)
        return torch.tensor(
            [float(e.omega) for e in kg.edges],
            dtype=torch.float32, device=device,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim
