"""CHAP Negative Sampler: Contrastive Hard-negative via Accounting Perturbations.

Three negative types (Zero-Sum property):
  - CHAP-A: Additive violation
  - CHAP-S: Scale violation
  - CHAP-E: Entity/Year swap
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from gsr_cacl.kg.data_structures import ConstraintKG
from gsr_cacl.kg.parser import parse_markdown_rows


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class PerturbedTable:
    """A table after CHAP perturbation."""
    table_md: str
    perturbation_type: str      # "CHAP-A" | "CHAP-S" | "CHAP-E"
    perturbed_cell_id: str
    original_value: str
    new_value: str
    is_violated: bool = True

    def __repr__(self) -> str:
        return (f"PerturbedTable({self.perturbation_type}, "
                f"cell={self.perturbed_cell_id}, "
                f"{self.original_value} → {self.new_value})")


# ----------------------------------------------------------------------
# Number formatting
# ----------------------------------------------------------------------

def format_number(value: float, original_fmt: str) -> str:
    """Format a float back to string using the same format as original."""
    if "(" in original_fmt and ")" in original_fmt:
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
    if value < 0:
        return f"({abs(value):,.2f})"
    return f"{value:,.2f}"


# ----------------------------------------------------------------------
# CHAP-A: Additive violation
# ----------------------------------------------------------------------

def _find_additive_violation_target(kg: ConstraintKG) -> tuple[str, float] | None:
    child_ids = {e.src for e in kg.accounting_edges}
    if not child_ids:
        return None
    parent_ids = {e.tgt for e in kg.accounting_edges}
    leaf_candidates = [
        n for n in kg.nodes
        if n.id in child_ids and n.id not in parent_ids and n.value is not None
    ]
    if leaf_candidates:
        chosen = random.choice(leaf_candidates)
        return chosen.id, chosen.value
    for n in kg.nodes:
        if n.id in child_ids and n.value is not None:
            return n.id, n.value
    return None


def apply_chap_a(kg: ConstraintKG) -> PerturbedTable | None:
    """CHAP-A: change one LHS cell, breaking the equation."""
    target = _find_additive_violation_target(kg)
    if target is None:
        return None
    node_id, original_val = target
    node = kg.get_node(node_id)
    if node is None:
        return None
    factor = random.choice([1.1, 1.2, 1.3, 0.9, 0.8, 0.7])
    new_val = round(original_val * factor, 2)
    new_table_md = _reconstruct_table_md(kg, node_id, new_val)
    return PerturbedTable(
        table_md=new_table_md,
        perturbation_type="CHAP-A",
        perturbed_cell_id=node_id,
        original_value=str(original_val),
        new_value=str(new_val),
    )


# ----------------------------------------------------------------------
# CHAP-S: Scale violation
# ----------------------------------------------------------------------

def apply_chap_s(kg: ConstraintKG) -> PerturbedTable | None:
    """CHAP-S: change unit (M → B), breaking ratio constraints."""
    candidates = [(n.id, n.value) for n in kg.nodes if n.value is not None and abs(n.value) > 1e3]
    if not candidates:
        return None
    node_id, original_val = random.choice(candidates)
    node = kg.get_node(node_id)
    if node is None:
        return None
    new_val = original_val * 0.001 if abs(original_val) > 1 else original_val * 1000
    new_val = round(new_val, 4)
    new_table_md = _reconstruct_table_md(kg, node_id, new_val)
    return PerturbedTable(
        table_md=new_table_md,
        perturbation_type="CHAP-S",
        perturbed_cell_id=node_id,
        original_value=str(original_val),
        new_value=str(new_val),
    )


# ----------------------------------------------------------------------
# CHAP-E: Entity/Year swap
# ----------------------------------------------------------------------

def apply_chap_e(
    kg: ConstraintKG,
    wrong_company: str = "WrongCorp Inc.",
    wrong_year: str = "2020",
) -> PerturbedTable:
    """CHAP-E: same structure, wrong company or year."""
    new_table_md = (
        f"[COMPANY: {wrong_company}] [YEAR: {wrong_year}]\n"
        + kg.table_md
    )
    return PerturbedTable(
        table_md=new_table_md,
        perturbation_type="CHAP-E",
        perturbed_cell_id="entity_metadata",
        original_value="original_entity",
        new_value=f"{wrong_company} / {wrong_year}",
    )


# ----------------------------------------------------------------------
# Table reconstruction helper
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
    node = kg.get_node(perturbed_node_id)
    if node is None:
        return kg.table_md

    target_row_idx = node.row_idx
    target_col_idx = node.col_idx

    # +1 offset because parse_markdown_rows includes headers as row 0
    # but node row_idx is 0-based for data rows
    actual_row = target_row_idx + 1 if target_row_idx + 1 < len(rows) else target_row_idx

    if actual_row < len(rows) and target_col_idx < len(rows[actual_row]):
        original_fmt = rows[actual_row][target_col_idx]
        rows[actual_row][target_col_idx] = format_number(new_value, original_fmt)

    col_widths = [
        max(len(str(row[i])) for row in rows if i < len(row)) + 1
        for i in range(len(rows[0]))
    ]

    lines = []
    for ri, row in enumerate(rows):
        cells = [
            str(row[i]).ljust(col_widths[i]) if i < len(row) else ""
            for i in range(len(col_widths))
        ]
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
    Generate hard negatives using three CHAP perturbation strategies.

    Usage:
        sampler = CHAPNegativeSampler()
        negatives = sampler.sample(kg, n_negatives=3)
    """

    def __init__(
        self,
        chap_a_prob: float = 0.5,
        chap_s_prob: float = 0.3,
        chap_e_prob: float = 0.2,
        seed: int = 42,
    ):
        self.probs = [chap_a_prob, chap_s_prob, chap_e_prob]
        self.rng = random.Random(seed)

    def sample(
        self,
        kg: ConstraintKG,
        n_negatives: int = 1,
    ) -> list[PerturbedTable]:
        negatives: list[PerturbedTable] = []
        types_used: set[str] = set()

        for _ in range(n_negatives):
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

    def sample_from_table_md(self, table_md: str, n_negatives: int = 1) -> list[PerturbedTable]:
        """Build KG from markdown and generate negatives in one call."""
        from gsr_cacl.kg.builder import build_kg_from_markdown
        kg = build_kg_from_markdown(table_md)
        return self.sample(kg, n_negatives=n_negatives)

    def get_diagnostics(self, negatives: list[PerturbedTable]) -> dict:
        type_counts: dict[str, int] = {}
        for neg in negatives:
            type_counts[neg.perturbation_type] = type_counts.get(neg.perturbation_type, 0) + 1
        return {
            "total": len(negatives),
            "by_type": type_counts,
            "all_violated": all(n.is_violated for n in negatives),
        }
