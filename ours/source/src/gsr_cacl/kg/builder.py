"""Core KG construction from parsed markdown tables."""

from __future__ import annotations

import re
from typing import Optional

from gsr_cacl.kg.data_structures import ConstraintKG, KGNode, KGEdge
from gsr_cacl.kg.parser import parse_markdown_rows, parse_number
from gsr_cacl.templates.data_structures import AccountingTemplate
from gsr_cacl.templates.matching import match_template, normalize_header


# ----------------------------------------------------------------------
# Total-detection heuristics
# ----------------------------------------------------------------------

_TOTAL_KEYWORDS = {
    "total", "totals", "sum", "grand total", "net total",
    "total assets", "total liabilities", "total equity",
    "total revenue", "total debt", "net income",
    "net profit", "net cash flow", "annual",
}


def _is_total_header(h: str) -> bool:
    h_lower = h.lower().strip()
    return any(kw in h_lower for kw in _TOTAL_KEYWORDS)


def _is_total_cell(cell_text: str) -> bool:
    t = cell_text.lower().strip()
    if not t:
        return False
    return t in _TOTAL_KEYWORDS or any(kw in t for kw in _TOTAL_KEYWORDS)


# ----------------------------------------------------------------------
# Core builder
# ----------------------------------------------------------------------

def build_constraint_kg(
    table_md: str,
    headers: list[str],
    cell_values: list[list[str | float | None]],
    epsilon: float = 1e-4,
) -> ConstraintKG:
    """
    Build a constraint knowledge graph from a parsed markdown table.

    Args:
        table_md:    raw markdown table string
        headers:     list of column header strings
        cell_values: 2D list of cell values (rows × cols)
        epsilon:     tolerance for numerical checks
    Returns:
        ConstraintKG with nodes, edges, matched template.
    """
    if not headers or len(cell_values) < 1:
        return ConstraintKG(table_md=table_md)

    # --- 1. Template matching ---
    canonical_headers = [normalize_header(h) for h in headers]
    template, conf = match_template(canonical_headers)

    # --- 2. Build nodes ---
    nodes: list[KGNode] = []

    for row_idx, row in enumerate(cell_values):
        for col_idx, raw_cell in enumerate(row):
            if raw_cell is None:
                cell_text = ""
            elif isinstance(raw_cell, (int, float)):
                cell_text = str(raw_cell)
            else:
                cell_text = str(raw_cell).strip()

            num_val: float | None = None
            if isinstance(raw_cell, (int, float)):
                num_val = float(raw_cell)
            elif isinstance(raw_cell, str):
                num_val = parse_number(raw_cell)

            col_header = headers[col_idx] if col_idx < len(headers) else ""
            canonical = normalize_header(col_header) if col_header else ""
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

    # --- 3. Build edges ---
    edges: list[KGEdge] = []

    if template is not None and conf >= 0.5:
        edges = _build_template_edges(nodes, headers, cell_values, template)

    if not edges or conf < 0.7:
        edges = _build_positional_edges(nodes, len(cell_values), len(headers))

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
    cell_values: list[list],
    template: AccountingTemplate,
) -> list[KGEdge]:
    """Build accounting constraint edges from a matched template."""
    edges: list[KGEdge] = []
    canonical_headers = [normalize_header(h) for h in headers]

    for constraint in template.constraints:
        lhs_cols: list[int] = []
        for h in constraint.lhs:
            try:
                idx = canonical_headers.index(h)
                lhs_cols.append(idx)
            except ValueError:
                pass

        try:
            rhs_col = canonical_headers.index(constraint.rhs)
        except ValueError:
            continue

        for row_idx in range(len(cell_values)):
            for lhs_col in lhs_cols:
                src_id = f"v_{row_idx}_{lhs_col}"
                tgt_id = f"v_{row_idx}_{rhs_col}"
                edges.append(KGEdge(
                    src=src_id,
                    tgt=tgt_id,
                    omega=constraint.omega,
                    edge_type="accounting",
                    constraint_name=constraint.name,
                ))

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
    """Fallback: build a sparse positional graph (same-row + same-col edges)."""
    edges: list[KGEdge] = []

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            node_id = f"v_{row_idx}_{col_idx}"

            for other_col in range(col_idx + 1, n_cols):
                other_id = f"v_{row_idx}_{other_col}"
                edges.append(KGEdge(
                    src=node_id, tgt=other_id,
                    omega=0, edge_type="positional",
                    constraint_name="same_row",
                ))

            for other_row in range(row_idx + 1, n_rows):
                other_id = f"v_{other_row}_{col_idx}"
                edges.append(KGEdge(
                    src=node_id, tgt=other_id,
                    omega=0, edge_type="positional",
                    constraint_name="same_col",
                ))

    return edges


# ----------------------------------------------------------------------
# Convenience: one-shot from markdown string
# ----------------------------------------------------------------------

def build_kg_from_markdown(
    table_md: str,
    epsilon: float = 1e-4,
) -> ConstraintKG:
    """
    Build a KG directly from a raw markdown table string.

    Usage:
        kg = build_kg_from_markdown(table_md)
    """
    rows = parse_markdown_rows(table_md)
    if len(rows) < 2:
        return ConstraintKG(table_md=table_md)

    headers = rows[0]
    data_rows = rows[1:]
    return build_constraint_kg(table_md, headers, data_rows, epsilon=epsilon)
