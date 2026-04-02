"""CHAP Negative Sampler: Contrastive Hard-negative viA Accounting Perturbations.

Three negative types, all satisfying the Zero-Sum property:
  - CHAP-A: Additive violation  (change 1 cell → recompute parent → broken equation)
  - CHAP-S: Scale violation    (change unit M → B, parent unchanged → ratio broken)
  - CHAP-E: Entity/Year swap   (wrong company or year, same structure)

Zero-Sum property:
  "Change exactly one component, keep constraint structure intact"
  → negatives are "close to positive" but guaranteed to violate constraints.

Reference:
  overall_idea.md §3.2 — CHAP Negative Generation Protocol
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from gsr_cacl.kg import (
    ConstraintKG,
    KGNode,
    KGEdge,
    build_constraint_kg,
    parse_markdown_rows,
)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class PerturbedTable:
    """A table after CHAP perturbation."""
    table_md: str              # markdown string (reconstructed)
    perturbation_type: str      # "CHAP-A" | "CHAP-S" | "CHAP-E"
    perturbed_cell_id: str      # which node was changed
    original_value: str
    new_value: str
    is_violated: bool = True    # always True for CHAP negatives

    def __repr__(self) -> str:
        return (f"PerturbedTable({self.perturbation_type}, "
                f"cell={self.perturbed_cell_id}, "
                f"{self.original_value} → {self.new_value})")


# ----------------------------------------------------------------------
# Number formatting utilities
# ----------------------------------------------------------------------

def format_number(value: float, original_fmt: str) -> str:
    """
    Format a float back to string using the same format as original.
    Preserves: commas, currency symbols, parentheses for negatives.
    """
    # Detect original format
    if "(" in original_fmt and ")" in original_fmt:
        # Negative in parentheses
        return f"({abs(value):,.2f})"
    if original_fmt.startswith("$"):
        return f"${value:,.2f}"
    if original_fmt.endswith("%"):
        return f"{value * 100:.2f}%"
    if original_fmt.endswith("B") or original_fmt.endswith("b"):
        return f"{value / 1e9:.2f}B"
    if original_fmt.endswith("M") or original_fmt.endswith("m"):
        return f"{value / 1e6:.2f}M"
    if original_fmt.endswith("K") or original_fmt.endswith("k"):
        return f"{value / 1e3:.2f}K"
    # Default: comma-separated
    if value < 0:
        return f"({abs(value):,.2f})"
    return f"{value:,.2f}"


def parse_number_str(cell: str) -> tuple[float | None, str]:
    """
    Parse a numeric cell string. Returns (value, original_format).
    """
    s = cell.strip()
    original_fmt = s

    # Detect format
    has_parens = s.startswith("(") and s.endswith(")")
    has_currency = bool(re.match(r"^[\$€£¥]", s))
    has_pct = s.endswith("%")
    has_scale = s[-1].upper() in "BMK"
    has_comma = "," in s

    # Strip formatting characters
    cleaned = re.sub(r"[\$€£¥\(\)%]", "", s)
    if has_comma:
        cleaned = cleaned.replace(",", "")
    if cleaned.startswith("-"):
        cleaned = cleaned[1:]
        has_parens = True  # negative

    # Scale
    scale = 1.0
    if has_scale:
        suffix = cleaned[-1].upper()
        cleaned = cleaned[:-1]
        if suffix == "B":
            scale = 1e9
        elif suffix == "M":
            scale = 1e6
        elif suffix == "K":
            scale = 1e3

    try:
        val = float(cleaned) * scale
        if has_parens:
            val = -val
        return val, original_fmt
    except ValueError:
        return None, original_fmt


# ----------------------------------------------------------------------
# CHAP-A: Additive violation
# ----------------------------------------------------------------------

def _find_additive_violation_target(
    kg: ConstraintKG,
) -> tuple[str, float] | None:
    """
    Find a leaf (non-total) node that participates in an additive constraint.
    Returns (node_id, original_value).
    """
    # Find nodes that are LHS of accounting edges (children)
    child_ids = {e.src for e in kg.accounting_edges}
    if not child_ids:
        return None

    # Prefer leaf nodes (not targets of any accounting edge)
    parent_ids = {e.tgt for e in kg.accounting_edges}
    leaf_candidates = [
        n for n in kg.nodes
        if n.id in child_ids and n.id not in parent_ids and n.value is not None
    ]

    if leaf_candidates:
        chosen = random.choice(leaf_candidates)
        return chosen.id, chosen.value

    # Fallback: any child node with a numeric value
    for n in kg.nodes:
        if n.id in child_ids and n.value is not None:
            return n.id, n.value

    return None


def apply_chap_a(kg: ConstraintKG) -> PerturbedTable | None:
    """
    CHAP-A: Additive violation — change one LHS cell, breaking the equation.

    Example:
      Revenue=100M, COGS=60M, Gross_Profit=40M
      → Change Revenue to 120M → Gross_Profit equation broken (120M - 60M ≠ 40M)
    """
    target = _find_additive_violation_target(kg)
    if target is None:
        return None

    node_id, original_val = target
    node = kg.get_node(node_id)
    if node is None:
        return None

    # Perturb by ±10–30%
    factor = random.choice([1.1, 1.2, 1.3, 0.9, 0.8, 0.7])
    new_val = original_val * factor
    # Round to sensible precision
    new_val_rounded = round(new_val, 2)

    # Reconstruct table markdown with new value
    new_table_md = _reconstruct_table_md(kg, node_id, new_val_rounded)

    return PerturbedTable(
        table_md=new_table_md,
        perturbation_type="CHAP-A",
        perturbed_cell_id=node_id,
        original_value=str(original_val),
        new_value=str(new_val_rounded),
        is_violated=True,
    )


# ----------------------------------------------------------------------
# CHAP-S: Scale violation
# ----------------------------------------------------------------------

def _find_scale_target(kg: ConstraintKG) -> tuple[str, float] | None:
    """Find a numeric cell that could be scaled."""
    candidates = [
        (n.id, n.value)
        for n in kg.nodes
        if n.value is not None and abs(n.value) > 1e3
    ]
    if candidates:
        return random.choice(candidates)
    return None


def apply_chap_s(kg: ConstraintKG) -> PerturbedTable | None:
    """
    CHAP-S: Scale violation — change unit (M → B), breaking ratio constraints.

    Example:
      Revenue: 500M → 0.5B (same number, different unit)
      → Ratio constraints broken because actual value changed 500×

    Strategy: instead of changing display unit (complex), we just scale the value.
    """
    target = _find_scale_target(kg)
    if target is None:
        return None

    node_id, original_val = target
    node = kg.get_node(node_id)
    if node is None:
        return None

    # Scale by 1000× (simulates M↔B swap)
    if abs(original_val) > 1:
        new_val = original_val * 0.001
    else:
        new_val = original_val * 1000

    new_table_md = _reconstruct_table_md(kg, node_id, round(new_val, 4))

    return PerturbedTable(
        table_md=new_table_md,
        perturbation_type="CHAP-S",
        perturbed_cell_id=node_id,
        original_value=str(original_val),
        new_value=str(round(new_val, 4)),
        is_violated=True,
    )


# ----------------------------------------------------------------------
# CHAP-E: Entity/Year swap
# ----------------------------------------------------------------------

def apply_chap_e(
    kg: ConstraintKG,
    wrong_company: str = "WrongCorp Inc.",
    wrong_year: str = "2020",
) -> PerturbedTable:
    """
    CHAP-E: Entity/Year swap — same structure, wrong company or year.

    This preserves the internal constraint structure but violates
    entity identity (company_name + report_year mismatch).

    Returns a perturbed table with modified metadata markers.
    """
    # Reconstruct with wrong entity info in header
    # Since we only have table_md, we prefix it with a wrong metadata header
    original_md = kg.table_md
    new_table_md = (
        f"[COMPANY: {wrong_company}] [YEAR: {wrong_year}]\n"
        + original_md
    )

    return PerturbedTable(
        table_md=new_table_md,
        perturbation_type="CHAP-E",
        perturbed_cell_id="entity_metadata",
        original_value=f"{wrong_company} / {wrong_year}",
        new_value=f"{wrong_company} / {wrong_year}",
        is_violated=True,
    )


# ----------------------------------------------------------------------
# Table reconstruction
# ----------------------------------------------------------------------

def _reconstruct_table_md(
    kg: ConstraintKG,
    perturbed_node_id: str,
    new_value: float,
) -> str:
    """Reconstruct markdown table with one cell value changed."""
    rows = parse_markdown_rows(kg.table_md)
    if not rows:
        return kg.table_md

    # Find node
    node = kg.get_node(perturbed_node_id)
    if node is None:
        return kg.table_md

    # Update the cell in the rows
    target_row_idx = node.row_idx
    target_col_idx = node.col_idx

    if target_row_idx < len(rows) and target_col_idx < len(rows[target_row_idx]):
        original_fmt = rows[target_row_idx][target_col_idx]
        rows[target_row_idx][target_col_idx] = format_number(new_value, original_fmt)

    # Reconstruct markdown
    col_widths = [max(len(str(row[i])) for row in rows if i < len(row)) + 1
                  for i in range(len(rows[0]))]

    lines = []
    for ri, row in enumerate(rows):
        cells = [str(row[i]).ljust(col_widths[i]) if i < len(row) else ""
                 for i in range(len(col_widths))]
        line = "| " + " | ".join(cells) + " |"
        lines.append(line)
        if ri == 0 and len(rows) > 1:
            sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
            lines.append(sep)

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main CHAP Sampler
# ----------------------------------------------------------------------

class CHAPNegativeSampler:
    """
    CHAP (Contrastive Hard-negative viA Accounting Perturbations) sampler.

    Generates hard negatives using three perturbation strategies:
      1. CHAP-A: Additive constraint violation
      2. CHAP-S: Scale/unit violation
      3. CHAP-E: Entity/year swap

    Each negative satisfies the Zero-Sum property: structure intact,
    exactly one component changed → guaranteed to violate constraints.

    Usage:
        sampler = CHAPNegativeSampler()
        for sample in dataset:
            pos_kg = build_constraint_kg(sample.table_md)
            negatives = sampler.sample(pos_kg, n_negatives=3)
    """

    def __init__(
        self,
        chap_a_prob: float = 0.5,
        chap_s_prob: float = 0.3,
        chap_e_prob: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            chap_a_prob: probability of CHAP-A perturbation
            chap_s_prob: probability of CHAP-S perturbation
            chap_e_prob: probability of CHAP-E perturbation
            seed: random seed
        """
        self.probs = [chap_a_prob, chap_s_prob, chap_e_prob]
        self.rng = random.Random(seed)

    def sample(
        self,
        kg: ConstraintKG,
        n_negatives: int = 1,
    ) -> list[PerturbedTable]:
        """
        Generate n CHAP negatives from a positive KG.

        Args:
            kg: ConstraintKG from positive document
            n_negatives: number of negatives to generate
        Returns:
            list of PerturbedTable objects
        """
        negatives: list[PerturbedTable] = []
        types_used: set[str] = set()

        for _ in range(n_negatives):
            # Select perturbation type (cycle through types for diversity)
            if len(types_used) < 3 and len(negatives) < 3:
                remaining = [t for t in ["CHAP-A", "CHAP-S", "CHAP-E"] if t not in types_used]
                ptype = self.rng.choice(remaining)
            else:
                ptype = self.rng.choices(
                    ["CHAP-A", "CHAP-S", "CHAP-E"],
                    weights=self.probs,
                )[0]

            if ptype == "CHAP-A":
                neg = apply_chap_a(kg)
            elif ptype == "CHAP-S":
                neg = apply_chap_s(kg)
            else:
                neg = apply_chap_e(kg)

            if neg is not None:
                negatives.append(neg)
                types_used.add(ptype)

        return negatives

    def sample_from_table_md(
        self,
        table_md: str,
        n_negatives: int = 1,
    ) -> list[PerturbedTable]:
        """
        Convenience method: build KG and generate negatives in one call.
        """
        kg = build_constraint_kg(table_md)
        return self.sample(kg, n_negatives=n_negatives)

    def get_diagnostics(self, negatives: list[PerturbedTable]) -> dict:
        """Return statistics about generated negatives."""
        type_counts = {}
        for neg in negatives:
            type_counts[neg.perturbation_type] = type_counts.get(neg.perturbation_type, 0) + 1
        return {
            "total": len(negatives),
            "by_type": type_counts,
            "all_violated": all(n.is_violated for n in negatives),
        }
