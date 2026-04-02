"""KG Builder: Graph-Structured Retrieval Knowledge Graph construction.

Architecture:
  table_md ──parse──► Cell nodes + Header mapping
                     │
                     └──► Template Matching
                           │
                           └──► Constraint edges (ω ∈ {+1, -1})
                                 │
                                 └──► Partial KG (fallback: positional graph)

Key equations:
  Revenue − COGS = Gross Profit     (ω = −1, additive)
  CA + NCA = Total Assets          (ω = +1, additive)
  Revenue × Margin = Gross Profit  (ω = +1, multiplicative)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gsr_cacl.templates import (
    AccountingConstraint,
    AccountingTemplate,
    match_template,
    _normalize_header,
)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

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
    template: Optional[AccountingTemplate] = None
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


# ----------------------------------------------------------------------
# Markdown table parsing
# ----------------------------------------------------------------------

def parse_markdown_rows(table_md: str) -> list[list[str]]:
    """
    Parse a markdown table into rows of cell strings.
    Handles pipes '|' and whitespace padding.

    Example:
        | A | B | C |
        |---|---|---|
        | 1 | 2 | 3 |

    Returns:
        list of rows (each row = list of cell strings)
    """
    lines = table_md.strip().split("\n")
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip markdown separator lines like |---|---|
        if re.match(r"^\|[\s\-:]+\|$", line) or re.match(r"^[\s\-:\|]+$", line):
            continue
        # Split by '|', strip whitespace
        cells = [cell.strip() for cell in line.split("|")]
        # Remove leading/trailing empty cells (from leading/trailing pipes)
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)
    return rows


def parse_number(cell: str) -> float | None:
    """
    Parse a numeric value from a cell string.
    Handles: "1,234", "$1,234", "(123)", "-123", "12.3%", "1.2B", "500M"
    Returns None if not numeric.
    """
    if not cell or not cell.strip():
        return None
    s = cell.strip()

    # Negative in parentheses: (123) → -123
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    # Strip currency symbols and commas
    s = re.sub(r"[\$€£¥,]", "", s)

    # Strip percentage → multiply by 0.01
    pct = False
    if s.endswith("%"):
        pct = True
        s = s[:-1]

    # Scale suffixes: B, M, K
    scale = 1.0
    if s.endswith("B") or s.endswith("b"):
        scale = 1e9
        s = s[:-1]
    elif s.endswith("M") or s.endswith("m"):
        scale = 1e6
        s = s[:-1]
    elif s.endswith("K") or s.endswith("k"):
        scale = 1e3
        s = s[:-1]

    try:
        val = float(s)
        if neg:
            val = -val
        val *= scale
        if pct:
            val *= 0.01
        return val
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Core KG Builder
# ----------------------------------------------------------------------

_TOTAL_KEYWORDS = {
    "total", "totals", "sum", "grand total", "net total",
    "total assets", "total liabilities", "total equity",
    "total revenue", "total debt", "net income",
    "net profit", "net cash flow", "annual",
}


def _is_total_header(h: str) -> bool:
    """True if header suggests a total / parent row."""
    h_lower = h.lower().strip()
    for kw in _TOTAL_KEYWORDS:
        if kw in h_lower:
            return True
    return False


def _is_total_cell(cell_text: str) -> bool:
    """True if a cell value looks like a total (contains 'total' keyword or is labelled)."""
    t = cell_text.lower().strip()
    if not t:
        return False
    return t in _TOTAL_KEYWORDS or any(kw in t for kw in _TOTAL_KEYWORDS)


def build_constraint_kg(
    table_md: str,
    headers: list[str],
    cell_values: list[list[str | float | None]],
    epsilon: float = 1e-4,
) -> ConstraintKG:
    """
    Build a constraint knowledge graph from a parsed markdown table.

    Args:
        table_md:    raw markdown table string (used for fallback)
        headers:     list of column header strings
        cell_values: 2D list of cell strings (row 0 = headers, rows 1+ = data)
                     Each cell may be a string or a pre-parsed float.
        epsilon:     tolerance for ε-tolerance check (default 1e-4)

    Returns:
        ConstraintKG with nodes, edges, matched template.
    """
    if not headers or len(cell_values) < 2:
        # Empty or header-only table → return empty KG
        return ConstraintKG(table_md=table_md)

    # --- 1. Template matching ----------------------------------------
    canonical_headers = [_normalize_header(h) for h in headers]
    template, conf = match_template(canonical_headers)

    # --- 2. Build nodes ---------------------------------------------
    nodes: list[KGNode] = []
    row_count = len(cell_values)

    for row_idx, row in enumerate(cell_values):
        for col_idx, raw_cell in enumerate(row):
            # Convert to string
            if raw_cell is None:
                cell_text = ""
            elif isinstance(raw_cell, (int, float)):
                cell_text = str(raw_cell)
            else:
                cell_text = str(raw_cell).strip()

            # Parse numeric value
            num_val: float | None = None
            if isinstance(raw_cell, (int, float)):
                num_val = float(raw_cell)
            elif isinstance(raw_cell, str):
                num_val = parse_number(raw_cell)

            # Header for this column
            col_header = headers[col_idx] if col_idx < len(headers) else ""
            canonical = _normalize_header(col_header) if col_header else ""

            # Is this a total/parent node?
            is_total = _is_total_header(col_header) or _is_total_cell(cell_text)

            node = KGNode(
                id=f"v_{row_idx}_{col_idx}",
                row_idx=row_idx,
                col_idx=col_idx,
                value=num_val,
                text=cell_text,
                header=col_header,
                header_canonical=canonical,
                is_total=is_total,
            )
            nodes.append(node)

    # --- 3. Build edges ----------------------------------------------
    edges: list[KGEdge] = []

    if template is not None and conf >= 0.5:
        # === Template-based (accounting) edges ===
        edges = _build_template_edges(nodes, headers, cell_values, template)

    if not edges or conf < 0.7:
        # === Fallback: positional graph ===
        edges = _build_positional_edges(nodes, row_count, len(headers))

    return ConstraintKG(
        nodes=nodes,
        edges=edges,
        template=template,
        template_confidence=conf,
        table_md=table_md,
    )


def _build_template_edges(
    nodes: list[KGNode],
    headers: list[str],
    cell_values: list[list[str | float | None]],
    template: AccountingTemplate,
) -> list[KGEdge]:
    """Build accounting constraint edges from a matched template."""
    edges: list[KGEdge] = []
    canonical_headers = [_normalize_header(h) for h in headers]

    for constraint in template.constraints:
        # Find column indices for LHS and RHS
        lhs_cols: list[int] = []
        for h in constraint.lhs:
            try:
                idx = canonical_headers.index(h)
                lhs_cols.append(idx)
            except ValueError:
                pass  # skip if header not found

        try:
            rhs_col = canonical_headers.index(constraint.rhs)
        except ValueError:
            continue  # skip if RHS header not found

        # Build edges: each LHS cell → RHS cell (parent)
        for row_idx in range(1, len(cell_values)):  # skip header row
            for lhs_col in lhs_cols:
                # LHS → RHS edge (child → parent)
                src_id = f"v_{row_idx}_{lhs_col}"
                tgt_id = f"v_{row_idx}_{rhs_col}"
                edges.append(KGEdge(
                    src=src_id,
                    tgt=tgt_id,
                    omega=constraint.omega,
                    edge_type="accounting",
                    constraint_name=constraint.name,
                ))

            # Also add sibling edges (same row, same constraint group)
            for i, ci in enumerate(lhs_cols):
                for cj in lhs_cols[i + 1:]:
                    si = f"v_{row_idx}_{ci}"
                    sj = f"v_{row_idx}_{cj}"
                    edges.append(KGEdge(
                        src=si, tgt=sj,
                        omega=0,
                        edge_type="positional",
                        constraint_name=f"sibling_{constraint.name}",
                    ))

    return edges


def _build_positional_edges(
    nodes: list[KGNode],
    n_rows: int,
    n_cols: int,
) -> list[KGEdge]:
    """
    Fallback: build a sparse positional graph.
    Edges connect cells in the same row or same column.
    """
    edges: list[KGEdge] = []

    for row_idx in range(1, n_rows):          # skip header row
        for col_idx in range(n_cols):
            node_id = f"v_{row_idx}_{col_idx}"

            # Same-row edges
            for other_col in range(col_idx + 1, n_cols):
                other_id = f"v_{row_idx}_{other_col}"
                edges.append(KGEdge(
                    src=node_id, tgt=other_id,
                    omega=0,
                    edge_type="positional",
                    constraint_name="same_row",
                ))

            # Same-column edges
            for other_row in range(row_idx + 1, n_rows):
                other_id = f"v_{other_row}_{col_idx}"
                edges.append(KGEdge(
                    src=node_id, tgt=other_id,
                    omega=0,
                    edge_type="positional",
                    constraint_name="same_col",
                ))

    return edges


# ----------------------------------------------------------------------
# High-level convenience function (mirrors overall_idea.md pseudo-code)
# ----------------------------------------------------------------------

def build_kg_from_markdown(
    table_md: str,
    epsilon: float = 1e-4,
) -> ConstraintKG:
    """
    One-shot KG construction from a markdown table string.

    Usage:
        kg = build_kg_from_markdown(table_md)
        print(f"Template: {kg.template.name if kg.template else 'None'}, "
              f"Confidence: {kg.template_confidence:.2f}")
        print(f"Nodes: {len(kg.nodes)}, Edges: {len(kg.edges)}")
    """
    rows = parse_markdown_rows(table_md)
    if len(rows) < 2:
        return ConstraintKG(table_md=table_md)

    headers = rows[0]
    data_rows = rows[1:]
    return build_constraint_kg(table_md, headers, data_rows, epsilon=epsilon)
