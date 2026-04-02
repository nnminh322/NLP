"""Constraint-Aware Scoring: constraint score + entity score."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gsr_cacl.kg.data_structures import ConstraintKG


@dataclass
class ConstraintScoringResult:
    """Results from constraint scoring."""
    constraint_score: float        # 0–1 (higher = more consistent)
    violated_count: int
    total_count: int
    per_constraint_scores: list[float]


def compute_constraint_score(
    kg: ConstraintKG,
    epsilon: float = 1e-4,
) -> ConstraintScoringResult:
    """
    Compute constraint satisfaction score for a KG.

    For each accounting edge (u → v, ω):
        residual = |ω · v_u − v_v|
        score = exp(− residual / max(|v_v|, ε))

    Returns average score over all accounting edges.
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
