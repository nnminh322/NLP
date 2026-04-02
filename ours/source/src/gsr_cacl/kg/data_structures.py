"""Data structures for the constraint knowledge graph."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KGNode:
    """A node in the constraint knowledge graph."""
    id: str
    row_idx: int
    col_idx: int
    value: float | None
    text: str               # raw cell text
    header: str             # column header
    header_canonical: str   # mapped to IFRS/GAAP canonical form
    is_total: bool = False  # True if this is a "Total" / parent cell

    def __repr__(self) -> str:
        val_str = f"{self.value:.2f}" if self.value is not None else "None"
        return f"Node({self.header_canonical}={val_str}, r={self.row_idx})"


@dataclass
class KGEdge:
    """An edge in the constraint knowledge graph."""
    src: str
    tgt: str
    omega: int          # +1 = additive parent, −1 = subtractive parent, 0 = positional
    edge_type: str      # "accounting" | "positional"
    constraint_name: str = ""


@dataclass
class ConstraintKG:
    """
    Full knowledge graph for a single financial table.

    Attributes:
        nodes:     list of KGNode
        edges:     list of KGEdge  (accounting + positional)
        template:  matched template or None
        template_confidence:  0.0–1.0
        table_md:  raw markdown table string (for fallback)
    """
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    template: Optional[object] = None  # AccountingTemplate (avoid circular import)
    template_confidence: float = 0.0
    table_md: str = ""

    # Convenience lookups
    def get_node(self, node_id: str) -> KGNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def get_outgoing(self, node_id: str) -> list[KGEdge]:
        return [e for e in self.edges if e.src == node_id]

    def get_incoming(self, node_id: str) -> list[KGEdge]:
        return [e for e in self.edges if e.tgt == node_id]

    @property
    def accounting_edges(self) -> list[KGEdge]:
        return [e for e in self.edges if e.edge_type == "accounting"]

    @property
    def total_nodes(self) -> list[KGNode]:
        return [n for n in self.nodes if n.is_total]
