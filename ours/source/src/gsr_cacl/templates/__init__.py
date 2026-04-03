"""Accounting template library for GSR-CACL."""

from gsr_cacl.templates.data_structures import AccountingConstraint, AccountingTemplate
from gsr_cacl.templates.library import TEMPLATES
from gsr_cacl.templates.matching import match_template, normalize_header

# Backwards-compatible aliases
TEMPLATE_REGISTRY = TEMPLATES
_normalize_header = normalize_header

def get_all_templates() -> list[AccountingTemplate]:
    return list(TEMPLATES.values())

__all__ = [
    "AccountingConstraint",
    "AccountingTemplate",
    "TEMPLATES",
    "TEMPLATE_REGISTRY",
    "get_all_templates",
    "match_template",
    "normalize_header",
    "_normalize_header",
]
