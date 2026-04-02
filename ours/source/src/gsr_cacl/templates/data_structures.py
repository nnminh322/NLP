"""Data structures for accounting templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class AccountingConstraint:
    """A single accounting identity: LHS op RHS = 0."""
    name: str
    lhs: list[str]          # header names on the left-hand side
    rhs: str                # single header on the right-hand side (total)
    omega: int              # +1 or -1 (direction of constraint)
    op: str = "add"         # "add" | "sub" | "div"

    def __post_init__(self):
        assert self.omega in (+1, -1), "omega must be +1 or -1"


@dataclass
class AccountingTemplate:
    """Represents one IFRS/GAAP accounting template pattern."""
    name: str
    description: str
    headers: list[str]                     # ordered headers (row-based reading)
    constraints: list[AccountingConstraint] = field(default_factory=list)
    confidence_threshold: float = 0.7
    detector: Optional[Callable[[list[str]], float]] = None

    def __repr__(self) -> str:
        return f"Template({self.name})"
