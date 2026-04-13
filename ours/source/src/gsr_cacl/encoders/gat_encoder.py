"""Full GAT Encoder for financial table cells.

Architecture (architecture.md §3):
    Input:  [CellEmbed(value, header) ⊕ PosEnc(row) ⊕ PosEnc(col)]
    Layer 1: GAT → LeakyReLU
    Layer 2: GAT → LeakyReLU
    Output: node embeddings [V, hidden_dim]

Cell embedding strategy:
    - Numeric value: log-scale encoding projected to embed_dim
    - Header text: learnable hash-based embedding (lightweight, no BGE needed at train time)
    - External BGE embeddings can be injected via encode_with_external_embeds()
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from gsr_cacl.kg.data_structures import ConstraintKG
from gsr_cacl.encoders.positional import SinusoidalPositionalEncoding
from gsr_cacl.encoders.gat_layer import GATLayer


def _numeric_features(value: Optional[float], device: torch.device) -> torch.Tensor:
    """Encode a numeric value into a small feature vector [4].

    Features: [log|v|, sign, is_zero, magnitude_bucket]
    """
    if value is None or math.isnan(value) or math.isinf(value):
        return torch.zeros(4, device=device)
    sign = 1.0 if value >= 0 else -1.0
    is_zero = 1.0 if abs(value) < 1e-8 else 0.0
    log_abs = math.log1p(abs(value))
    # magnitude bucket: 0=tiny, 1=small, 2=medium, 3=large, 4=huge
    if abs(value) < 1:
        bucket = 0.0
    elif abs(value) < 1e3:
        bucket = 1.0
    elif abs(value) < 1e6:
        bucket = 2.0
    elif abs(value) < 1e9:
        bucket = 3.0
    else:
        bucket = 4.0
    return torch.tensor([log_abs, sign, is_zero, bucket / 4.0], device=device)


def _header_hash(text: str, n_buckets: int = 512) -> int:
    """Deterministic hash of header text to bucket index."""
    h = 0
    for c in text.lower().strip():
        h = (h * 31 + ord(c)) % n_buckets
    return h


class GATEncoder(nn.Module):
    """
    Two-layer GAT encoder for financial table cells.

    Architecture:
        Input:  [CellEmbed ⊕ PosEnc(row) ⊕ PosEnc(col)]
        Layer 1: GAT → LeakyReLU
        Layer 2: GAT → LeakyReLU
        Output: node embeddings [V, hidden_dim]

    Cell embedding (learnable, no external model needed):
        - header_embed: learnable embedding table indexed by header hash
        - numeric_proj / numeric_encoder: encodes numeric values

    Numeric encoding versions (--contr1 flag):
        v1: numeric_proj over [log|v|, sign, is_zero, mag_bucket] → embed_dim
        v2: ScaleAwareNumericEncoder (magnitude bin + mantissa + unit) → embed_dim
    """

    HEADER_BUCKETS = 512

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        numeric_encoder: nn.Module | None = None,
        numeric_version: str = "v1",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.numeric_version = numeric_version

        # Cell embedding components
        self.header_embed = nn.Embedding(self.HEADER_BUCKETS, embed_dim)

        # V1: original log-scale numeric projection
        self.numeric_proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
        )

        # V2: ScaleAwareNumericEncoder (passed in as module)
        self.numeric_encoder_module = numeric_encoder

        self.cell_norm = nn.LayerNorm(embed_dim)

        # Positional encodings for row/col
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

    def _build_cell_embeddings(self, kg: ConstraintKG, device: torch.device) -> torch.Tensor:
        """Build learnable cell embeddings from header text + numeric value.

        Returns: [V, embed_dim]

        V1 (default): hash → header_embed + [log|v|, sign, zero, bucket] → numeric_proj
        V2:           hash → header_embed + ScaleAwareNumericEncoder (magnitude+mantissa+unit)
        """
        V = len(kg.nodes)

        # Header hash → embedding
        header_indices = torch.tensor(
            [_header_hash(n.header, self.HEADER_BUCKETS) for n in kg.nodes],
            dtype=torch.long, device=device,
        )
        h_embed = self.header_embed(header_indices)  # [V, embed_dim]

        # Numeric features → projection (v1) or ScaleAwareNumericEncoder (v2)
        if self.numeric_encoder_module is not None:
            # V2: multi-resolution encoding
            values = [n.value for n in kg.nodes]
            headers = [n.header for n in kg.nodes]
            cell_texts = [n.text for n in kg.nodes]
            n_embed = self.numeric_encoder_module.forward(values, headers, cell_texts, device)
        else:
            # V1: original log-scale encoding
            num_feats = torch.stack(
                [_numeric_features(n.value, device) for n in kg.nodes], dim=0
            )  # [V, 4]
            n_embed = self.numeric_proj(num_feats)  # [V, embed_dim]

        # Combine: additive fusion + layer norm
        cell_embed = self.cell_norm(h_embed + n_embed)
        return cell_embed

    def forward(
        self,
        kg: ConstraintKG,
        external_cell_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode all nodes of a knowledge graph.

        Args:
            kg: ConstraintKG with pre-computed nodes
            external_cell_embeds: Optional [V, embed_dim] tensor (e.g. from BGE).
                If provided, uses these instead of learnable cell embeddings.
        Returns:
            node_embeddings: [V, hidden_dim]
        """
        if not kg.nodes:
            return torch.empty(0, self.hidden_dim)

        V = len(kg.nodes)
        device = next(self.parameters()).device

        # Cell embeddings: external (BGE) or learnable
        if external_cell_embeds is not None:
            cell_embeds = external_cell_embeds.to(device)
        else:
            cell_embeds = self._build_cell_embeddings(kg, device)

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

    def encode_graph(self, kg: ConstraintKG, external_cell_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a KG and return graph-level embedding via mean pooling.

        Returns: [hidden_dim]
        """
        node_embeds = self.forward(kg, external_cell_embeds=external_cell_embeds)
        if node_embeds.numel() == 0:
            return torch.zeros(self.hidden_dim, device=next(self.parameters()).device)
        return node_embeds.mean(dim=0)

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
