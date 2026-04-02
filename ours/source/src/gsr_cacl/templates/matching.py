"""Template matching: header normalization and fuzzy matching."""

from __future__ import annotations

from gsr_cacl.templates.data_structures import AccountingTemplate
from gsr_cacl.templates.library import TEMPLATES


# ----------------------------------------------------------------------
# Header synonym mapping (case-insensitive)
# ----------------------------------------------------------------------

_HEADER_SYNONYMS: dict[str, str] = {
    # Revenue variants
    "revenue": "Revenue", "total revenue": "Revenue",
    "net revenue": "Revenue", "sales": "Revenue",
    "net sales": "Revenue", "total sales": "Revenue",
    # COGS
    "cogs": "COGS", "cost of revenue": "COGS",
    "cost of goods sold": "COGS", "cost of sales": "COGS",
    "operating costs": "Operating Expenses",
    # Gross Profit
    "gross profit": "Gross Profit", "gross income": "Gross Profit",
    # Operating Expenses
    "operating expenses": "Operating Expenses", "opex": "Operating Expenses",
    "sga": "SG&A", "sg&a": "SG&A",
    "selling general and administrative": "SG&A",
    # EBIT / Operating Income
    "operating income": "Operating Income", "ebit": "Operating Income",
    "income from operations": "Operating Income",
    "operating profit": "Operating Income",
    # Tax
    "income tax": "Income Tax", "tax expense": "Income Tax", "tax": "Income Tax",
    # Net Income
    "net income": "Net Income", "net profit": "Net Income",
    "net earnings": "Net Income", "profit for the period": "Net Income",
    # Assets
    "current assets": "Current Assets",
    "non-current assets": "Non-Current Assets",
    "total assets": "Total Assets",
    "property plant equipment": "Non-Current Assets",
    "ppe": "Non-Current Assets",
    "goodwill": "Non-Current Assets",
    "intangible assets": "Non-Current Assets",
    # Liabilities
    "current liabilities": "Current Liabilities",
    "non-current liabilities": "Non-Current Liabilities",
    "total liabilities": "Total Liabilities",
    "long-term debt": "Long-Term Debt", "lt debt": "Long-Term Debt",
    "short-term debt": "Short-Term Debt", "st debt": "Short-Term Debt",
    "total debt": "Total Debt",
    # Equity
    "total equity": "Total Equity",
    "shareholders equity": "Total Equity",
    "stockholders equity": "Total Equity",
    "common stock": "Common Stock",
    "preferred stock": "Preferred Stock",
    "retained earnings": "Retained Earnings",
    # Cash Flow
    "operating cash flow": "Operating Cash Flow",
    "cash from operations": "Operating Cash Flow",
    "ocf": "Operating Cash Flow",
    "investing cash flow": "Investing Cash Flow",
    "financing cash flow": "Financing Cash Flow",
    "net cash flow": "Net Cash Flow",
    "net increase in cash": "Net Cash Flow",
    # EBITDA
    "ebitda": "EBITDA",
    "da": "Depreciation & Amortization",
    "d&a": "Depreciation & Amortization",
    "depreciation": "Depreciation & Amortization",
    "depreciation and amortization": "Depreciation & Amortization",
    # EPS
    "net income per share": "EPS", "earnings per share": "EPS",
    "eps": "EPS", "diluted eps": "EPS",
    "shares outstanding": "Shares Outstanding",
    "weighted average shares": "Shares Outstanding",
    # Margins / Ratios
    "gross margin": "Gross Margin",
    "operating margin": "Operating Margin",
    "net margin": "Net Profit Margin",
    "net profit margin": "Net Profit Margin",
    "current ratio": "Current Ratio",
    # Quarterly
    "q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4",
    "annual": "Annual", "fiscal year": "Annual", "full year": "Annual",
    # YoY
    "yoy change": "YoY Change",
}


def normalize_header(h: str) -> str:
    """Map a header to its canonical form."""
    h_lower = h.lower().strip()
    if h_lower in _HEADER_SYNONYMS:
        return _HEADER_SYNONYMS[h_lower]
    return " ".join(w.capitalize() for w in h.split())


def _fuzzy_match_headers(
    table_headers: list[str],
    template: AccountingTemplate,
) -> float:
    """
    Compute a 0-1 match score between table_headers and template.headers.
    Score = (# canonical matches) / max(len(table), len(template))
    """
    if not template.headers:
        return 0.0
    table_canonical = [normalize_header(h) for h in table_headers]
    matched = sum(1 for th in template.headers if th in table_canonical)
    denom = max(len(table_headers), len(template.headers))
    return matched / denom if denom > 0 else 0.0


def match_template(
    headers: list[str],
) -> tuple[AccountingTemplate | None, float]:
    """
    Find the best-matching accounting template for a given list of column headers.
    Returns (template, confidence). If confidence < 0.5, returns (None, 0.0).
    """
    best_tpl: AccountingTemplate | None = None
    best_score = 0.0

    for tpl in TEMPLATES.values():
        score = _fuzzy_match_headers(headers, tpl)
        if score > best_score:
            best_score = score
            best_tpl = tpl

    if best_score < 0.5:
        return None, 0.0
    return best_tpl, best_score


def get_all_template_names() -> list[str]:
    """Return list of all registered template names."""
    return list(TEMPLATES.keys())
