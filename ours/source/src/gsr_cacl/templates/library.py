"""15 IFRS/GAAP Accounting Template Definitions.

Covers ~80-90% of financial tables in T²-RAGBench.
"""

from __future__ import annotations

from gsr_cacl.templates.data_structures import AccountingConstraint, AccountingTemplate


# Global registry
TEMPLATES: dict[str, AccountingTemplate] = {}


def _register(tpl: AccountingTemplate) -> AccountingTemplate:
    TEMPLATES[tpl.name] = tpl
    return tpl


# ---- 1. Income Statement ------------------------------------------------
_register(AccountingTemplate(
    name="income_statement",
    description="Revenue → COGS → Gross Profit → OpEx → EBIT → EBT → Net Income",
    headers=["Revenue", "COGS", "Gross Profit", "Operating Expenses",
             "Operating Income", "Income Tax", "Net Income"],
    constraints=[
        AccountingConstraint("gross_profit",
                             lhs=["Revenue", "COGS"], rhs="Gross Profit",
                             omega=-1, op="sub"),
        AccountingConstraint("operating_income",
                             lhs=["Gross Profit", "Operating Expenses"], rhs="Operating Income",
                             omega=-1, op="sub"),
        AccountingConstraint("net_income",
                             lhs=["Operating Income", "Income Tax"], rhs="Net Income",
                             omega=-1, op="sub"),
    ],
))

# ---- 2. Balance Sheet (assets) ------------------------------------------
_register(AccountingTemplate(
    name="balance_sheet_assets",
    description="Current Assets + Non-Current Assets = Total Assets",
    headers=["Current Assets", "Non-Current Assets", "Total Assets"],
    constraints=[
        AccountingConstraint("total_assets",
                             lhs=["Current Assets", "Non-Current Assets"], rhs="Total Assets",
                             omega=+1, op="add"),
    ],
))

# ---- 3. Balance Sheet (liabilities + equity) ----------------------------
_register(AccountingTemplate(
    name="balance_sheet_le",
    description="Total Liabilities + Total Equity = Total (Assets = L + E)",
    headers=["Current Liabilities", "Non-Current Liabilities",
             "Total Liabilities", "Total Equity"],
    constraints=[
        AccountingConstraint("total_liabilities",
                             lhs=["Current Liabilities", "Non-Current Liabilities"],
                             rhs="Total Liabilities", omega=+1, op="add"),
    ],
))

# ---- 4. Cash Flow -------------------------------------------------------
_register(AccountingTemplate(
    name="cash_flow",
    description="Operating CF + Investing CF + Financing CF = Net Cash Flow",
    headers=["Operating Cash Flow", "Investing Cash Flow",
             "Financing Cash Flow", "Net Cash Flow"],
    constraints=[
        AccountingConstraint("net_cf",
                             lhs=["Operating Cash Flow", "Investing Cash Flow",
                                  "Financing Cash Flow"],
                             rhs="Net Cash Flow", omega=+1, op="add"),
    ],
))

# ---- 5. Revenue by Segment ----------------------------------------------
_register(AccountingTemplate(
    name="revenue_segment",
    description="Segment_1 + ... + Segment_N = Total Revenue",
    headers=[],   # dynamic; populated at match time
    constraints=[
        AccountingConstraint("segment_sum", lhs=[], rhs="Total Revenue",
                             omega=+1, op="add"),
    ],
))

# ---- 6. Gross Margin Ratio ----------------------------------------------
_register(AccountingTemplate(
    name="gross_margin_ratio",
    description="Gross Profit / Revenue = Gross Margin",
    headers=["Revenue", "Gross Profit", "Gross Margin"],
    constraints=[
        AccountingConstraint("gross_margin_def",
                             lhs=["Revenue", "Gross Profit"], rhs="Gross Margin",
                             omega=+1, op="div"),
    ],
))

# ---- 7. Year-over-Year Change -------------------------------------------
_register(AccountingTemplate(
    name="yoy_change",
    description="[Year_N, Value] pairs, ordered",
    headers=[],
    constraints=[],
))

# ---- 8. Quarterly Breakdown ---------------------------------------------
_register(AccountingTemplate(
    name="quarterly_breakdown",
    description="Q1 + Q2 + Q3 + Q4 = Annual",
    headers=["Q1", "Q2", "Q3", "Q4", "Annual"],
    constraints=[
        AccountingConstraint("annual_total",
                             lhs=["Q1", "Q2", "Q3", "Q4"], rhs="Annual",
                             omega=+1, op="add"),
    ],
))

# ---- 9. EPS Calculation -------------------------------------------------
_register(AccountingTemplate(
    name="eps",
    description="Net Income / Shares Outstanding = EPS",
    headers=["Net Income", "Shares Outstanding", "EPS"],
    constraints=[
        AccountingConstraint("eps_def",
                             lhs=["Net Income", "Shares Outstanding"], rhs="EPS",
                             omega=+1, op="div"),
    ],
))

# ---- 10. Debt Schedule --------------------------------------------------
_register(AccountingTemplate(
    name="debt_schedule",
    description="Long-Term Debt + Short-Term Debt = Total Debt",
    headers=["Long-Term Debt", "Short-Term Debt", "Total Debt"],
    constraints=[
        AccountingConstraint("total_debt",
                             lhs=["Long-Term Debt", "Short-Term Debt"], rhs="Total Debt",
                             omega=+1, op="add"),
    ],
))

# ---- 11. Shareholder Equity ---------------------------------------------
_register(AccountingTemplate(
    name="shareholder_equity",
    description="Common Stock + Preferred Stock + Retained Earnings = Total Equity",
    headers=["Common Stock", "Preferred Stock", "Retained Earnings", "Total Equity"],
    constraints=[
        AccountingConstraint("equity_identity",
                             lhs=["Common Stock", "Preferred Stock", "Retained Earnings"],
                             rhs="Total Equity", omega=+1, op="add"),
    ],
))

# ---- 12. EBITDA ---------------------------------------------------------
_register(AccountingTemplate(
    name="ebitda",
    description="Revenue - COGS - SG&A + D&A = EBITDA",
    headers=["Revenue", "COGS", "SG&A", "Depreciation & Amortization", "EBITDA"],
    constraints=[
        AccountingConstraint("ebitda_def",
                             lhs=["Revenue", "COGS", "SG&A", "Depreciation & Amortization"],
                             rhs="EBITDA", omega=-1, op="sub"),
    ],
))

# ---- 13. Operating Margin -----------------------------------------------
_register(AccountingTemplate(
    name="operating_margin",
    description="Operating Income / Revenue = Operating Margin",
    headers=["Revenue", "Operating Income", "Operating Margin"],
    constraints=[
        AccountingConstraint("op_margin_def",
                             lhs=["Revenue", "Operating Income"], rhs="Operating Margin",
                             omega=+1, op="div"),
    ],
))

# ---- 14. Net Profit Margin ----------------------------------------------
_register(AccountingTemplate(
    name="net_margin",
    description="Net Income / Revenue = Net Profit Margin",
    headers=["Revenue", "Net Income", "Net Profit Margin"],
    constraints=[
        AccountingConstraint("net_margin_def",
                             lhs=["Revenue", "Net Income"], rhs="Net Profit Margin",
                             omega=+1, op="div"),
    ],
))

# ---- 15. Current Ratio --------------------------------------------------
_register(AccountingTemplate(
    name="current_ratio",
    description="Current Assets / Current Liabilities = Current Ratio",
    headers=["Current Assets", "Current Liabilities", "Current Ratio"],
    constraints=[
        AccountingConstraint("current_ratio_def",
                             lhs=["Current Assets", "Current Liabilities"], rhs="Current Ratio",
                             omega=+1, op="div"),
    ],
))
