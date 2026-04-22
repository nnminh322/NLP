"""Full GAT Encoder for financial table cells.

Implements §4.3.3–4.3.4 of the GSR-CACL proposal:
    - Node feature construction: [BGE(cell_text) ⊕ PE(row) ⊕ PE(col) ⊕ e_D]
    - Edge-Aware GAT with EntitySim and residual connection
    - Graph-level representation via mean pooling

Node Features (Eq.4 in proposal):
    x_v = f_θ(cell_text_v) ⊕ PE_row(r) ⊕ PE_col(c) ⊕ e_D
        = BGE(cell_text) ⊕ positional_encoding ⊕ entity_embedding

FIX BUG 4: Entity embeddings are NOW properly concatenated into node features.
FIX BUG 8: EntitySim scale is now a learnable parameter.

GAT Attention (Eq.6 in proposal):
    e_uv = <Q_u, K_v> / √d_k + Proj(ω_uv) + σ(scale) · EntitySim(e_u, e_v)

GAT Update (Eq.8 in proposal):
    h_v^{(l+1)} = W_o [ ⊕_k Σ α_uv · ω_uv · W_v h_u^{(l)} ] + h_v^{(l)}
                   └─ residual ─┘  └─ edge-weighted message ─┘
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsr_cacl.kg.data_structures import ConstraintKG
from gsr_cacl.encoders.positional import SinusoidalPositionalEncoding
from gsr_cacl.encoders.gat_layer import GATLayer


# ---------------------------------------------------------------------------
# Numeric features (fallback when no BGE cell encoding)
# ---------------------------------------------------------------------------

def _numeric_features(value: Optional[float], device: torch.device) -> torch.Tensor:
    """Encode a numeric value into a small feature vector [4].

    Features: [log|v|, sign, is_zero, magnitude_bucket]
    Used as a fallback when BGE cell encoding is not available.
    """
    if value is None or math.isnan(value) or math.isinf(value):
        return torch.zeros(4, device=device)
    sign = 1.0 if value >= 0 else -1.0
    is_zero = 1.0 if abs(value) < 1e-8 else 0.0
    log_abs = math.log1p(abs(value))
    abs_v = abs(value)
    if abs_v < 1:
        bucket = 0.0
    elif abs_v < 1e3:
        bucket = 1.0
    elif abs_v < 1e6:
        bucket = 2.0
    elif abs_v < 1e9:
        bucket = 3.0
    else:
        bucket = 4.0
    return torch.tensor([log_abs, sign, is_zero, bucket / 4.0], device=device)


def _header_hash(text: str, n_buckets: int = 512) -> int:
    """Deterministic hash of header text to bucket index (fallback)."""
    h = 0
    for c in text.lower().strip():
        h = (h * 31 + ord(c)) % n_buckets
    return h


# ---------------------------------------------------------------------------
# GAT Encoder
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """
    Two-layer GAT encoder for financial table cells.

    Architecture (proposal §4.3.3):
        Input:  [CellEmbed ⊕ PosEnc(row) ⊕ PosEnc(col) ⊕ e_D]
        Layer 1: GAT → LeakyReLU  (edge-aware attention + EntitySim + residual)
        Layer 2: GAT → LeakyReLU  (edge-aware attention + EntitySim + residual)
        Output: node embeddings [V, hidden_dim]
        Graph representation: mean_pool(node_embeddings)

    FIX BUG 4: Entity embeddings (e_D) are now concatenated into node features
                per proposal Eq.4: x_v = [cell ⊕ PE(row) ⊕ PE(col) ⊕ e_D]

    FIX BUG 8: EntitySim scale is now a learnable parameter instead of fixed 0.5.
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
        entity_embed_dim: int = 256,
        add_entity_sim: bool = False,
        use_bge_cell_encoding: bool = True,
        cell_encoder: nn.Module | None = None,
    ):
        """
        Args:
            embed_dim: Text encoder output dimension (d = 1024 for bge-large)
            hidden_dim: GAT hidden dimension (h_dim = 256 in proposal)
            num_heads: Number of attention heads (H = 4 in proposal)
            num_layers: Number of GAT layers (L = 2 in proposal)
            dropout: Dropout rate
            numeric_encoder: ScaleAwareNumericEncoder (v2) or None (v1)
            numeric_version: "v1" (log-scale) or "v2" (magnitude+mantissa)
            entity_embed_dim: Entity embedding dimension (d_e = 256)
            add_entity_sim: If True, include EntitySim in GAT attention (Eq.6)
            use_bge_cell_encoding: If True, use BGE for cell text (Eq.4)
            cell_encoder: Pre-trained BGE encoder for cell text (optional)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.numeric_version = numeric_version
        self.entity_embed_dim = entity_embed_dim
        self.add_entity_sim = add_entity_sim
        self.use_bge_cell_encoding = use_bge_cell_encoding
        self.cell_encoder = cell_encoder
        self._has_entity = entity_embed_dim > 0

        # Cell embedding components
        self.header_embed = nn.Embedding(self.HEADER_BUCKETS, embed_dim)

        # V1: original log-scale numeric projection [4] → embed_dim
        self.numeric_proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
        )

        # V2: ScaleAwareNumericEncoder (passed in as module)
        self.numeric_encoder_module = numeric_encoder

        self.cell_norm = nn.LayerNorm(embed_dim)

        # Positional encodings for row/col (proposal Eq.4)
        self.row_pe = SinusoidalPositionalEncoding(embed_dim // 4, max_len=256)
        self.col_pe = SinusoidalPositionalEncoding(embed_dim // 4, max_len=64)

        # FIX BUG 4: Compute input dimension for Eq.4 node features
        # x_v = [cell ⊕ PE(row) ⊕ PE(col) ⊕ e_D]
        #   cell:       embed_dim
        #   PE(row):    embed_dim // 4
        #   PE(col):    embed_dim // 4
        #   e_D:        entity_embed_dim (if available)
        self._proj_entity_dim = embed_dim  # internal: project e_D → embed_dim
        in_proj_dim = embed_dim + 2 * (embed_dim // 4)
        if self._has_entity:
            in_proj_dim += self._proj_entity_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # FIX BUG 4: Project entity embeddings to match cell embedding dimension
        if self._has_entity:
            self.entity_proj = nn.Linear(entity_embed_dim, self._proj_entity_dim)

        # GAT layers with edge-aware attention + optional EntitySim + residual
        self.gat_layers = nn.ModuleList([
            GATLayer(
                hidden_dim, hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                add_residual=True,     # From proposal Eq.8: + h_v^{(l)}
                add_entity_sim=add_entity_sim,
            )
            for _ in range(num_layers)
        ])

    def _build_cell_embeddings(
        self,
        kg: ConstraintKG,
        device: torch.device,
        external_cell_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build cell embeddings from header text + numeric value.

        Strategy (proposal Eq.4 — priority order):
            1. BGE(cell_text) if external_cell_embeds provided
            2. BGE(header) + numeric_proj if cell_encoder available
            3. hash(header) + numeric_proj (legacy fallback)
        """
        V = len(kg.nodes)
        if external_cell_embeds is not None:
            return external_cell_embeds.to(device)

        if self.cell_encoder is not None:
            headers = [n.header for n in kg.nodes]
            with torch.no_grad():
                cell_embeds = self.cell_encoder(headers, device=device)
            return cell_embeds

        # Legacy fallback: hash(header) + numeric
        header_indices = torch.tensor(
            [_header_hash(n.header, self.HEADER_BUCKETS) for n in kg.nodes],
            dtype=torch.long, device=device,
        )
        h_embed = self.header_embed(header_indices)

        if self.numeric_encoder_module is not None:
            values = [n.value for n in kg.nodes]
            headers = [n.header for n in kg.nodes]
            cell_texts = [n.text for n in kg.nodes]
            n_embed = self.numeric_encoder_module.forward(values, headers, cell_texts, device)
        else:
            num_feats = torch.stack(
                [_numeric_features(n.value, device) for n in kg.nodes], dim=0
            )
            n_embed = self.numeric_proj(num_feats)

        return self.cell_norm(h_embed + n_embed)

    def forward(
        self,
        kg: ConstraintKG,
        external_cell_embeds: torch.Tensor | None = None,
        entity_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode all nodes of a knowledge graph.

        Args:
            kg: ConstraintKG with pre-computed nodes
            external_cell_embeds: Optional [V, embed_dim] BGE cell embeddings
            entity_embeddings: Optional [V, entity_embed_dim] entity embeddings
                              CONCATENATED into node features per Eq.4
                              Also used in EntitySim attention term

        Returns:
            node_embeddings: [V, hidden_dim]

        FIX BUG 4: entity_embeddings are now concatenated into node features.
        """
        if not kg.nodes:
            return torch.empty(0, self.hidden_dim, device=next(self.parameters()).device)

        V = len(kg.nodes)
        device = next(self.parameters()).device

        if entity_embeddings is not None:
            if entity_embeddings.dim() == 1:
                entity_embeddings = entity_embeddings.unsqueeze(0)
            if entity_embeddings.size(0) == 1 and V > 1:
                entity_embeddings = entity_embeddings.expand(V, -1)

        # Build cell embeddings (Eq.4)
        cell_embeds = self._build_cell_embeddings(kg, device, external_cell_embeds)

        # Positional encodings (Eq.4)
        row_indices = torch.tensor(
            [n.row_idx for n in kg.nodes], dtype=torch.long, device=device
        )
        col_indices = torch.tensor(
            [n.col_idx for n in kg.nodes], dtype=torch.long, device=device
        )
        row_pos = self.row_pe(row_indices.unsqueeze(0)).squeeze(0)   # [V, embed_dim//4]
        col_pos = self.col_pe(col_indices.unsqueeze(0)).squeeze(0)   # [V, embed_dim//4]

        # FIX BUG 4: Concat entity embeddings into node features (Eq.4)
        # x_v = [cell ⊕ PE(row) ⊕ PE(col) ⊕ e_D]
        if self._has_entity and entity_embeddings is not None:
            entity_proj = self.entity_proj(entity_embeddings)  # [V, _proj_entity_dim=embed_dim]
            h = torch.cat([cell_embeds, row_pos, col_pos, entity_proj], dim=-1)
        else:
            h = torch.cat([cell_embeds, row_pos, col_pos], dim=-1)

        h = self.input_proj(h)

        # Edge structures
        edge_index = self._build_edge_index(kg, device)
        edge_weight = self._build_edge_weight(kg, device)

        # GAT forward: entity-aware attention + residual connection
        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_weight, entity_embeddings=entity_embeddings)

        return h

    def encode_graph(
        self,
        kg: ConstraintKG,
        external_cell_embeds: torch.Tensor | None = None,
        entity_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode a KG and return graph-level embedding via mean pooling.

        Formula (proposal Eq.9):
            d_KG = (1/|V|) Σ_{v∈V} h_v^{(L)}
        """
        node_embeds = self.forward(kg, external_cell_embeds, entity_embeddings)
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
