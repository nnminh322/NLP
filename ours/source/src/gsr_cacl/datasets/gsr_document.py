"""GSR-enriched document representation.

Wraps a Document with pre-computed Constraint-KG metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gsr_cacl.core import Document
from gsr_cacl.kg.builder import build_kg_from_markdown
from gsr_cacl.kg.data_structures import ConstraintKG


@dataclass
class GSRDocument:
    """Extended document with GSR-specific metadata."""

    base: Document
    kg: ConstraintKG
    template_name: str = ""
    template_confidence: float = 0.0
    n_constraint_edges: int = 0
    n_positional_edges: int = 0
    n_cells: int = 0

    @classmethod
    def from_document(cls, doc: Document) -> "GSRDocument":
        """Build GSRDocument from a g4k Document."""
        table_md = extract_table(doc.page_content)
        kg = build_kg_from_markdown(table_md)

        return cls(
            base=doc,
            kg=kg,
            template_name=kg.template.name if kg.template else "none",
            template_confidence=kg.template_confidence,
            n_constraint_edges=len([e for e in kg.edges if e.edge_type == "accounting"]),
            n_positional_edges=len([e for e in kg.edges if e.edge_type == "positional"]),
            n_cells=len(kg.nodes),
        )


def extract_table(content: str) -> str:
    """Extract the first markdown table from page_content."""
    lines = content.split("\n")
    table_lines: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if "|" in stripped:
            in_table = True
            table_lines.append(stripped)
        elif in_table:
            break
    if len(table_lines) < 2:
        return ""
    return "\n".join(table_lines)
