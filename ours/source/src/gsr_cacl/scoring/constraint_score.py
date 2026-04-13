"""Constraint-Aware Scoring: constraint score + entity score.

Two versions for constraint scoring:
    v1 (default): fixed epsilon tolerance  score = exp(-residual / max(|v|, ε))
    v2:           relative tolerance        score = exp(-residual / (|v| * rel_tol + ε))

P2 improvement: v1 has a critical bug with large-value tables.
For |v| = 1e12 (e.g. Apple's total assets), a residual of 1e8 produces:
    score_v1 = exp(-1e8 / 1e12) = exp(-1e-4) ≈ 0.9999
This is nearly perfect despite a $100 billion discrepancy — the fixed ε=1e-4
is orders of magnitude too small for large financial figures.

v2 uses RELATIVE tolerance: residual is measured relative to the target value.
    score_v2 = exp(-1e8 / (1e12 * 1e-4)) = exp(-0.001) ≈ 0.999
This is still lenient (correct: large companies round to billions), but the
tolerance scales with scale. For a small company with |v|=1e6:
    score_v2 = exp(-1e8 / (1e6 * 1e-4)) = exp(-1000) ≈ 0
→ The same absolute error of $100M is correctly identified as a severe violation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from gsr_cacl.kg.data_structures import ConstraintKG


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConstraintScoringResult:
    """Results from constraint scoring."""
    constraint_score: float        # 0–1 (higher = more consistent)
    violated_count: int
    total_count: int
    per_constraint_scores: list[float]


# ---------------------------------------------------------------------------
# V1: Fixed epsilon (original)
# ---------------------------------------------------------------------------

def compute_constraint_score_v1(
    kg: ConstraintKG,
    epsilon: float = 1e-4,
) -> ConstraintScoringResult:
    """
    V1: Fixed epsilon tolerance.

    score = exp(-residual / max(|tgt|, ε))

    Limitation: For large |tgt| values (e.g. |tgt|=1e12), the denominator
    max(|tgt|, ε) ≈ |tgt|, so the ratio residual/|tgt| is tiny even for
    large absolute errors. A $100B discrepancy in a $300B total is penalized
    almost nothing.

    For small |tgt| values (e.g. |tgt|=1e3), the same residual produces
    severe penalties. This asymmetry is not accounting-grounded.
    """
    acc_edges = kg.accounting_edges
    if not acc_edges:
        return ConstraintScoringResult(
            constraint_score=1.0,
            violated_count=0,
            total_count=0,
            per_constraint_scores=[],
        )

    node_map = {n.id: n for n in kg.nodes}
    scores = []
    violated = 0

    for edge in acc_edges:
        src_node = node_map.get(edge.src)
        tgt_node = node_map.get(edge.tgt)

        if src_node is None or tgt_node is None:
            continue
        if src_node.value is None or tgt_node.value is None:
            continue

        residual = abs(edge.omega * src_node.value - tgt_node.value)
        denom = max(abs(tgt_node.value), epsilon)
        edge_score = math.exp(-residual / denom)

        scores.append(edge_score)
        if edge_score < 0.5:
            violated += 1

    if not scores:
        return ConstraintScoringResult(
            constraint_score=1.0,
            violated_count=0,
            total_count=0,
            per_constraint_scores=[],
        )

    return ConstraintScoringResult(
        constraint_score=sum(scores) / len(scores),
        violated_count=violated,
        total_count=len(scores),
        per_constraint_scores=scores,
    )


# ---------------------------------------------------------------------------
# V2: Relative tolerance (P2 improvement)
# ---------------------------------------------------------------------------

def compute_constraint_score_v2(
    kg: ConstraintKG,
    relative_tolerance: float = 1e-4,
    epsilon: float = 1e-10,
) -> ConstraintScoringResult:
    """
    V2: Relative tolerance.

    score = exp(-residual / (|tgt| * relative_tolerance + ε))

    Rationale: In financial reporting, "small" vs "large" errors are relative
    to the magnitude of the target value:
        - A $1M error in a $300B revenue is negligible (< 0.1%)
        - A $1M error in a $5M revenue is catastrophic (> 20%)

    This matches how accountants think about materiality thresholds.
    relative_tolerance=1e-4 means: 0.01% relative error → exp(-1) ≈ 0.37
                                   0.10% relative error → exp(-10) ≈ 4.5e-5

    For ratio constraints (margin, EPS), the target value |tgt| is already
    normalized, so relative tolerance works naturally.
    For absolute constraints (Revenue = A + B), |tgt| is the total,
    which scales correctly with company size.

    Edge cases:
        - |tgt| = 0: falls back to ε (pure absolute tolerance)
        - |tgt| is very small (< 1): the ε term dominates → behaves like v1
    """
    acc_edges = kg.accounting_edges
    if not acc_edges:
        return ConstraintScoringResult(
            constraint_score=1.0,
            violated_count=0,
            total_count=0,
            per_constraint_scores=[],
        )

    node_map = {n.id: n for n in kg.nodes}
    scores = []
    violated = 0

    for edge in acc_edges:
        src_node = node_map.get(edge.src)
        tgt_node = node_map.get(edge.tgt)

        if src_node is None or tgt_node is None:
            continue
        if src_node.value is None or tgt_node.value is None:
            continue

        residual = abs(edge.omega * src_node.value - tgt_node.value)
        abs_tgt = abs(tgt_node.value)

        # Relative tolerance: penalize proportionally to target magnitude
        denom = abs_tgt * relative_tolerance + epsilon
        edge_score = math.exp(-residual / denom)

        scores.append(edge_score)
        if edge_score < 0.5:
            violated += 1

    if not scores:
        return ConstraintScoringResult(
            constraint_score=1.0,
            violated_count=0,
            total_count=0,
            per_constraint_scores=[],
        )

    return ConstraintScoringResult(
        constraint_score=sum(scores) / len(scores),
        violated_count=violated,
        total_count=len(scores),
        per_constraint_scores=scores,
    )


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

ConstraintScoringVersion = Literal["v1", "v2"]

# Default relative tolerance: 1e-4 = 0.01%
# This means a 0.01% relative error → exp(-1) ≈ 0.37 (borderline)
# and a 0.10% relative error → exp(-10) ≈ 0 (definitely violated)
DEFAULT_RELATIVE_TOLERANCE = 1e-3  # 0.1% relative tolerance


def compute_constraint_score(
    kg: ConstraintKG,
    epsilon: float = 1e-4,
    version: ConstraintScoringVersion = "v1",
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> ConstraintScoringResult:
    """
    Unified constraint scoring with version switch.

    Args:
        kg:                the constraint knowledge graph
        epsilon:           floor tolerance for v1 (ignored in v2)
        version:           "v1" = fixed epsilon, "v2" = relative tolerance
        relative_tolerance: relative tolerance for v2 (relative_tol=1e-4 → 0.01%)
    """
    if version == "v1":
        return compute_constraint_score_v1(kg, epsilon=epsilon)
    elif version == "v2":
        return compute_constraint_score_v2(kg, relative_tolerance=relative_tolerance, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown constraint scoring version: {version}. Choose 'v1' or 'v2'.")


# ---------------------------------------------------------------------------
# Entity scoring (unchanged)
# ---------------------------------------------------------------------------

def compute_entity_score(
    query_meta: dict,
    doc_meta: dict,
) -> float:
    """
    Compute entity matching score between query and document metadata.
    Returns float in [0, 1].
    """
    score = 0.0
    n_checks = 0

    q_company = str(query_meta.get("company_name", "")).lower().strip()
    d_company = str(doc_meta.get("company_name", "")).lower().strip()
    if q_company and d_company:
        n_checks += 1
        if q_company == d_company:
            score += 1.0
        elif q_company in d_company or d_company in q_company:
            score += 0.5

    q_year = str(query_meta.get("report_year", "")).strip()
    d_year = str(doc_meta.get("report_year", "")).strip()
    if q_year and d_year:
        n_checks += 1
        if q_year == d_year:
            score += 1.0
        else:
            try:
                if abs(int(q_year) - int(d_year)) <= 1:
                    score += 0.5
            except ValueError:
                pass

    q_sector = str(query_meta.get("company_sector", "")).lower().strip()
    d_sector = str(doc_meta.get("company_sector", "")).lower().strip()
    if q_sector and d_sector:
        n_checks += 1
        if q_sector == d_sector:
            score += 1.0
        elif q_sector in d_sector or d_sector in q_sector:
            score += 0.5

    if n_checks == 0:
        return 0.5
    return score / n_checks
