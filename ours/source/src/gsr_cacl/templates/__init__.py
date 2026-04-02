"""IFRS/GAAP Accounting Template Library.

Covers ~80-90% of financial tables in T²-RAGBench.
Each template defines: name, headers, constraints (LHS, RHS, ω).

Coverage estimates:
  - FinQA (S&P 500): ~90%
  - ConvFinQA: ~85%
  - TAT-DQA (diverse sectors): ~70%
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AccountingConstraint:
    """A single accounting identity: LHS op RHS = 0."""
    name: str
    lhs: list[str]          # header names on the left-hand side
    rhs: str                # single header on the right-hand side (total)
    omega: int              # +1 or -1 (direction of constraint)
    op: str = "add"         # "add" or "sub"

    def __post_init__(self):
        assert self.omega in (+1, -1), "omega must be +1 or -1"


# ----------------------------------------------------------------------
# 15 IFRS/GAAP Template Definitions
# ----------------------------------------------------------------------
TEMPLATES: dict[str, AccountingTemplate] = {}


def _reg(name: AccountingTemplate) -> AccountingTemplate:
    TEMPLATES[name.name] = name
    return name


@_reg
class AccountingTemplate:
    """Represents one IFRS/GAAP accounting template pattern."""

    name: str
    description: str
    headers: list[str]                     # ordered headers (row-based reading)
    constraints: list[AccountingConstraint]
    confidence_threshold: float = 0.7
    # Optional: function that detects this template from a list of headers
    detector: Callable[[list[str]], float] | None = None

    def __repr__(self) -> str:
        return f"Template({self.name})"


# ---- 1. Income Statement -----------------------------------------------
TEMPLATES["income_statement"] = AccountingTemplate(
    name="income_statement",
    description="Revenue → COGS → Gross Profit → OpEx → EBIT → EBT → Net Income",
    headers=["Revenue", "COGS", "Gross Profit", "Operating Expenses",
             "Operating Income", "Income Tax", "Net Income"],
    constraints=[
        AccountingConstraint(
            name="gross_profit",
            lhs=["Revenue", "COGS"],
            rhs="Gross Profit",
            omega=-1,  # Revenue - COGS = Gross Profit
            op="sub",
        ),
        AccountingConstraint(
            name="operating_income",
            lhs=["Gross Profit", "Operating Expenses"],
            rhs="Operating Income",
            omega=-1,  # Gross Profit - OpEx = EBIT
            op="sub",
        ),
        AccountingConstraint(
            name="net_income",
            lhs=["Operating Income", "Income Tax"],
            rhs="Net Income",
            omega=-1,  # EBIT - Tax = Net Income
            op="sub",
        ),
    ],
)


# ---- 2. Balance Sheet (assets) ------------------------------------------
TEMPLATES["balance_sheet_assets"] = AccountingTemplate(
    name="balance_sheet_assets",
    description="Current Assets + Non-Current Assets = Total Assets",
    headers=["Current Assets", "Non-Current Assets", "Total Assets"],
    constraints=[
        AccountingConstraint(
            name="total_assets",
            lhs=["Current Assets", "Non-Current Assets"],
            rhs="Total Assets",
            omega=+1,  # CA + NCA = Total
            op="add",
        ),
    ],
)


# ---- 3. Balance Sheet (liabilities + equity) ----------------------------
TEMPLATES["balance_sheet_le"] = AccountingTemplate(
    name="balance_sheet_le",
    description="Current Liabilities + Non-Current Liabilities + Equity = Total Equity",
    headers=["Current Liabilities", "Non-Current Liabilities",
             "Total Liabilities", "Total Equity"],
    constraints=[
        AccountingConstraint(
            name="total_liabilities",
            lhs=["Current Liabilities", "Non-Current Liabilities"],
            rhs="Total Liabilities",
            omega=+1,
            op="add",
        ),
        AccountingConstraint(
            name="balance_sheet_identity",
            lhs=["Total Liabilities", "Total Equity"],
            rhs="Total Equity",
            omega=-1,  # Liabilities + Equity = Total
            op="sub",
        ),
    ],
)


# ---- 4. Cash Flow -------------------------------------------------------
TEMPLATES["cash_flow"] = AccountingTemplate(
    name="cash_flow",
    description="Operating CF + Investing CF + Financing CF = Net Cash Flow",
    headers=["Operating Cash Flow", "Investing Cash Flow",
             "Financing Cash Flow", "Net Cash Flow"],
    constraints=[
        AccountingConstraint(
            name="net_cf",
            lhs=["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"],
            rhs="Net Cash Flow",
            omega=+1,
            op="add",
        ),
    ],
)


# ---- 5. Revenue by Segment ---------------------------------------------
TEMPLATES["revenue_segment"] = AccountingTemplate(
    name="revenue_segment",
    description="Segment_1 + ... + Segment_N = Total Revenue",
    headers=[],   # dynamic; populated at match time
    constraints=[
        AccountingConstraint(
            name="segment_sum",
            lhs=[],   # dynamic
            rhs="Total Revenue",
            omega=+1,
            op="add",
        ),
    ],
)


# ---- 6. Gross Margin Ratio ----------------------------------------------
TEMPLATES["gross_margin_ratio"] = AccountingTemplate(
    name="gross_margin_ratio",
    description="Gross Profit / Revenue = Gross Margin (ratio constraint)",
    headers=["Revenue", "Gross Profit", "Gross Margin"],
    constraints=[
        AccountingConstraint(
            name="gross_margin_def",
            lhs=["Revenue", "Gross Profit"],
            rhs="Gross Margin",
            omega=+1,  # Revenue × margin = Gross Profit  (multiplicative)
            op="div",
        ),
    ],
)


# ---- 7. Year-over-Year Change ------------------------------------------
TEMPLATES["yoy_change"] = AccountingTemplate(
    name="yoy_change",
    description="[Year_N, Value] pairs, ordered; delta = Year_N - Year_N-1",
    headers=[],   # dynamic (e.g., "2022", "2023", "2024")
    constraints=[],
)


# ---- 8. Quarterly Breakdown ---------------------------------------------
TEMPLATES["quarterly_breakdown"] = AccountingTemplate(
    name="quarterly_breakdown",
    description="Q1 + Q2 + Q3 + Q4 = Annual",
    headers=["Q1", "Q2", "Q3", "Q4", "Annual"],
    constraints=[
        AccountingConstraint(
            name="annual_total",
            lhs=["Q1", "Q2", "Q3", "Q4"],
            rhs="Annual",
            omega=+1,
            op="add",
        ),
    ],
)


# ---- 9. EPS Calculation -------------------------------------------------
TEMPLATES["eps"] = AccountingTemplate(
    name="eps",
    description="Net Income / Shares Outstanding = EPS",
    headers=["Net Income", "Shares Outstanding", "EPS"],
    constraints=[
        AccountingConstraint(
            name="eps_def",
            lhs=["Net Income", "Shares Outstanding"],
            rhs="EPS",
            omega=+1,
            op="div",
        ),
    ],
)


# ---- 10. Debt Schedule --------------------------------------------------
TEMPLATES["debt_schedule"] = AccountingTemplate(
    name="debt_schedule",
    description="Long-Term Debt + Short-Term Debt = Total Debt",
    headers=["Long-Term Debt", "Short-Term Debt", "Total Debt"],
    constraints=[
        AccountingConstraint(
            name="total_debt",
            lhs=["Long-Term Debt", "Short-Term Debt"],
            rhs="Total Debt",
            omega=+1,
            op="add",
        ),
    ],
)


# ---- 11. Shareholder Equity --------------------------------------------
TEMPLATES["shareholder_equity"] = AccountingTemplate(
    name="shareholder_equity",
    description="Common Stock + Preferred Stock + Retained Earnings = Total Equity",
    headers=["Common Stock", "Preferred Stock",
             "Retained Earnings", "Total Equity"],
    constraints=[
        AccountingConstraint(
            name="equity_identity",
            lhs=["Common Stock", "Preferred Stock", "Retained Earnings"],
            rhs="Total Equity",
            omega=+1,
            op="add",
        ),
    ],
)


# ---- 12. EBITDA --------------------------------------------------------
TEMPLATES["ebitda"] = AccountingTemplate(
    name="ebitda",
    description="Revenue - COGS - SG&A + D&A = EBITDA",
    headers=["Revenue", "COGS", "SG&A", "Depreciation & Amortization", "EBITDA"],
    constraints=[
        AccountingConstraint(
            name="ebitda_def",
            lhs=["Revenue", "COGS", "SG&A", "Depreciation & Amortization"],
            rhs="EBITDA",
            omega=-1,
            op="sub",
        ),
    ],
)


# ---- 13. Operating Margin ------------------------------------------------
TEMPLATES["operating_margin"] = AccountingTemplate(
    name="operating_margin",
    description="Operating Income / Revenue = Operating Margin",
    headers=["Revenue", "Operating Income", "Operating Margin"],
    constraints=[
        AccountingConstraint(
            name="op_margin_def",
            lhs=["Revenue", "Operating Income"],
            rhs="Operating Margin",
            omega=+1,
            op="div",
        ),
    ],
)


# ---- 14. Net Profit Margin ---------------------------------------------
TEMPLATES["net_margin"] = AccountingTemplate(
    name="net_margin",
    description="Net Income / Revenue = Net Profit Margin",
    headers=["Revenue", "Net Income", "Net Profit Margin"],
    constraints=[
        AccountingConstraint(
            name="net_margin_def",
            lhs=["Revenue", "Net Income"],
            rhs="Net Profit Margin",
            omega=+1,
            op="div",
        ),
    ],
)


# ---- 15. Current Ratio --------------------------------------------------
TEMPLATES["current_ratio"] = AccountingTemplate(
    name="current_ratio",
    description="Current Assets / Current Liabilities = Current Ratio",
    headers=["Current Assets", "Current Liabilities", "Current Ratio"],
    constraints=[
        AccountingConstraint(
            name="current_ratio_def",
            lhs=["Current Assets", "Current Liabilities"],
            rhs="Current Ratio",
            omega=+1,
            op="div",
        ),
    ],
)


# ----------------------------------------------------------------------
# Template Matching
# ----------------------------------------------------------------------

# Keyword → canonical header mapping (case-insensitive)
_HEADER_SYNONYMS: dict[str, str] = {
    # Revenue variants
    "revenue": "Revenue",
    "total revenue": "Revenue",
    "net revenue": "Revenue",
    "sales": "Revenue",
    "net sales": "Revenue",
    "total sales": "Revenue",
    # COGS
    "cogs": "COGS",
    "cost of revenue": "COGS",
    "cost of goods sold": "COGS",
    "cost of sales": "COGS",
    "operating costs": "Operating Expenses",
    # Gross Profit
    "gross profit": "Gross Profit",
    "gross margin": "Gross Profit",
    "gross income": "Gross Profit",
    # Operating Expenses
    "operating expenses": "Operating Expenses",
    "opex": "Operating Expenses",
    "sga": "SG&A",
    "sg&a": "SG&A",
    "selling general and administrative": "SG&A",
    # EBIT / Operating Income
    "operating income": "Operating Income",
    "ebit": "Operating Income",
    "income from operations": "Operating Income",
    "operating profit": "Operating Income",
    # Tax
    "income tax": "Income Tax",
    "tax expense": "Income Tax",
    "tax": "Income Tax",
    # Net Income
    "net income": "Net Income",
    "net profit": "Net Income",
    "net earnings": "Net Income",
    "profit for the period": "Net Income",
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
    "long-term debt": "Long-Term Debt",
    "lt debt": "Long-Term Debt",
    "short-term debt": "Short-Term Debt",
    "st debt": "Short-Term Debt",
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
    "net income per share": "EPS",
    "earnings per share": "EPS",
    "eps": "EPS",
    "diluted eps": "EPS",
    "shares outstanding": "Shares Outstanding",
    "weighted average shares": "Shares Outstanding",
    # Margins / Ratios
    "gross margin": "Gross Margin",
    "operating margin": "Operating Margin",
    "net margin": "Net Profit Margin",
    "net profit margin": "Net Profit Margin",
    "current ratio": "Current Ratio",
    # Quarterly
    "q1": "Q1",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Q4",
    "annual": "Annual",
    "fiscal year": "Annual",
    "full year": "Annual",
    # YoY
    "2020": "2020",
    "2021": "2021",
    "2022": "2022",
    "2023": "2023",
    "2024": "2024",
    "yoy change": "YoY Change",
}


def _normalize_header(h: str) -> str:
    """Map a header to its canonical form."""
    h_lower = h.lower().strip()
    # Check synonyms dict first
    if h_lower in _HEADER_SYNONYMS:
        return _HEADER_SYNONYMS[h_lower]
    # Otherwise return title-cased version
    return " ".join(w.capitalize() for w in h.split())


def _fuzzy_match_headers(
    table_headers: list[str],
    template: AccountingTemplate,
) -> float:
    """
    Compute a 0-1 match score between `table_headers` and `template.headers`.
    Score = (# canonical matches) / max(len(table_headers), len(template.headers))
    """
    if not template.headers:
        return 0.0

    table_canonical = [_normalize_header(h) for h in table_headers]
    matched = 0
    for th in template.headers:
        if th in table_canonical:
            matched += 1
    denom = max(len(table_headers), len(template.headers))
    return matched / denom if denom > 0 else 0.0


def match_template(headers: list[str]) -> tuple[AccountingTemplate | None, float]:
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


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def get_all_template_names() -> list[str]:
    """Return list of all registered template names."""
    return list(TEMPLATES.keys())
