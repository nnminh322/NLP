"""Complete IFRS/GAAP Accounting Template Library.

Covers ~95% of financial table patterns in T²-RAGBench.
Organised into 15 accounting domains:

    1.  Income Statement (Step-by-step subtractive chain)
    2.  Balance Sheet — Assets
    3.  Balance Sheet — Liabilities & Equity
    4.  Cash Flow Statement
    5.  EPS & Per-Share Metrics
    6.  Segment Reporting
    7.  Working Capital & Liquidity Ratios
    8.  Debt & Leverage
    9.  Revenue & COGS Breakdown
    10. Deferred Tax
    11. Goodwill & Business Combinations
    12. Leases (IFRS 16 / ASC 842)
    13. Inventory
    14. Equity & Dividends
    15. Other Comprehensive Income (OCI)

Constraint semantics
--------------------
    AccountingConstraint(lhs=[...], rhs="Total", omega=+1, op="add")
        → check: Σ(lhs) − rhs = 0  (additive sum, e.g. CA + NCA = Total Assets)

    AccountingConstraint(lhs=[...], rhs="Result", omega=-1, op="sub")
        → check: first(LHS) − Σ(rest(LHS)) − rhs = 0  (subtractive chain,
          e.g. Revenue − COGS − Gross Profit = 0)

Ratio / soft templates (headers only, no hard constraint)
    Templates with empty `constraints` list are detected by header matching
    but do not generate arithmetic checks — useful for ratio tables and
    period comparisons.
"""

from __future__ import annotations

from gsr_cacl.templates.data_structures import AccountingConstraint, AccountingTemplate

# ─── Global registry ───────────────────────────────────────────────────────────

TEMPLATES: dict[str, AccountingTemplate] = {}


def _r(tpl: AccountingTemplate) -> AccountingTemplate:
    TEMPLATES[tpl.name] = tpl
    return tpl


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1 — INCOME STATEMENT
# Substractive chain (omega=-1, op="sub"): Revenue − COGS − GP = 0 etc.
# ══════════════════════════════════════════════════════════════════════════════

# ── 1a. Multi-step Income Statement (canonical 6-line P&L) ────────────────────
# Step 1: Revenue − COGS = Gross Profit
_r(AccountingTemplate(
    name="p&l_gross_profit",
    description="Step 1: Revenue − COGS = Gross Profit",
    headers=["Revenue", "COGS", "Gross Profit"],
    constraints=[
        AccountingConstraint("gross_profit",
                             lhs=["Revenue", "COGS"], rhs="Gross Profit",
                             omega=-1, op="sub"),
    ],
))

# Step 2: Gross Profit − Operating Expenses = Operating Income
_r(AccountingTemplate(
    name="p&l_operating_income",
    description="Step 2: Gross Profit − Operating Expenses = Operating Income (EBIT)",
    headers=["Gross Profit", "Operating Expenses", "Operating Income"],
    constraints=[
        AccountingConstraint("operating_income",
                             lhs=["Gross Profit", "Operating Expenses"],
                             rhs="Operating Income", omega=-1, op="sub"),
    ],
))

# Step 3: Operating Income +/− Other Income/Expense = EBT (Earnings Before Tax)
_r(AccountingTemplate(
    name="p&l_ebt",
    description="Step 3: Operating Income ± Other Income/Expense = EBT",
    headers=["Operating Income", "Interest Expense", "Other Income",
             "Earnings Before Tax"],
    constraints=[
        AccountingConstraint("ebt",
                             lhs=["Operating Income", "Interest Expense", "Other Income"],
                             rhs="Earnings Before Tax", omega=-1, op="sub"),
    ],
))

# Step 4: EBT − Income Tax = Net Income
_r(AccountingTemplate(
    name="p&l_net_income",
    description="Step 4: Earnings Before Tax − Income Tax = Net Income",
    headers=["Earnings Before Tax", "Income Tax", "Net Income"],
    constraints=[
        AccountingConstraint("net_income",
                             lhs=["Earnings Before Tax", "Income Tax"],
                             rhs="Net Income", omega=-1, op="sub"),
    ],
))

# ── 1b. Simple 3-line P&L ─────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="simple_income_statement",
    description="Revenue → COGS → Gross Profit → Net Income (condensed)",
    headers=["Revenue", "COGS", "Gross Profit", "Operating Expenses",
             "Operating Income", "Income Tax", "Net Income"],
    constraints=[
        AccountingConstraint("gp",
                             lhs=["Revenue", "COGS"], rhs="Gross Profit",
                             omega=-1, op="sub"),
        AccountingConstraint("oi",
                             lhs=["Gross Profit", "Operating Expenses"],
                             rhs="Operating Income", omega=-1, op="sub"),
        AccountingConstraint("ni",
                             lhs=["Operating Income", "Income Tax"],
                             rhs="Net Income", omega=-1, op="sub"),
    ],
))

# ── 1c. P&L with Interest Income (banking / financial institutions) ────────────
_r(AccountingTemplate(
    name="p&l_with_interest",
    description="P&L including Interest Income and Interest Expense",
    headers=["Interest Income", "Interest Expense", "Net Interest Income",
             "Revenue", "Operating Expenses", "Operating Income",
             "Income Tax", "Net Income"],
    constraints=[
        AccountingConstraint("net_interest",
                             lhs=["Interest Income", "Interest Expense"],
                             rhs="Net Interest Income", omega=-1, op="sub"),
        AccountingConstraint("gp",
                             lhs=["Revenue", "COGS"], rhs="Gross Profit",
                             omega=-1, op="sub"),
        AccountingConstraint("oi",
                             lhs=["Gross Profit", "Operating Expenses"],
                             rhs="Operating Income", omega=-1, op="sub"),
        AccountingConstraint("ni",
                             lhs=["Operating Income", "Income Tax"],
                             rhs="Net Income", omega=-1, op="sub"),
    ],
))

# ── 1d. P&L with EBITDA ───────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="p&l_ebitda",
    description="P&L with explicit EBITDA line (Revenue − COGS − SG&A + D&A)",
    headers=["Revenue", "COGS", "Gross Profit", "SG&A", "Depreciation & Amortization",
             "EBITDA", "Operating Income", "Income Tax", "Net Income"],
    constraints=[
        AccountingConstraint("gp",
                             lhs=["Revenue", "COGS"], rhs="Gross Profit",
                             omega=-1, op="sub"),
        AccountingConstraint("ebitda",
                             lhs=["Gross Profit", "SG&A", "Depreciation & Amortization"],
                             rhs="EBITDA", omega=-1, op="sub"),
        AccountingConstraint("oi",
                             lhs=["EBITDA", "Depreciation & Amortization"],
                             rhs="Operating Income", omega=-1, op="sub"),
        AccountingConstraint("ni",
                             lhs=["Operating Income", "Income Tax"],
                             rhs="Net Income", omega=-1, op="sub"),
    ],
))

# ── 1e. P&L with Non-Controlling Interest ─────────────────────────────────────
_r(AccountingTemplate(
    name="p&l_nci",
    description="P&L with Non-Controlling Interest and attributable profit",
    headers=["Revenue", "COGS", "Operating Income", "Net Income",
             "Non-Controlling Interest", "Net Income Attributable to Parent"],
    constraints=[
        AccountingConstraint("gp",
                             lhs=["Revenue", "COGS"], rhs="Gross Profit",
                             omega=-1, op="sub"),
        AccountingConstraint("ni",
                             lhs=["Operating Income", "Income Tax"],
                             rhs="Net Income", omega=-1, op="sub"),
    ],
))

# ── 1f. P&L with Discontinued Operations ───────────────────────────────────────
_r(AccountingTemplate(
    name="p&l_discontinued",
    description="P&L including Discontinued Operations (after-tax gain/loss)",
    headers=["Operating Income", "Income Tax", "Income from Discontinued Operations",
             "Net Income", "Net Income Attributable to Parent"],
    constraints=[
        AccountingConstraint("ni_cont",
                             lhs=["Operating Income", "Income Tax"],
                             rhs="Net Income Before Disc Ops", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2 — BALANCE SHEET: ASSETS
# Additive constraints (omega=+1, op="add"): CA + NCA = Total Assets
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. Total Assets = Current Assets + Non-Current Assets ────────────────────
_r(AccountingTemplate(
    name="total_assets",
    description="Fundamental: Current Assets + Non-Current Assets = Total Assets",
    headers=["Current Assets", "Non-Current Assets", "Total Assets"],
    constraints=[
        AccountingConstraint("total_assets",
                             lhs=["Current Assets", "Non-Current Assets"],
                             rhs="Total Assets", omega=+1, op="add"),
    ],
))

# ── 2b. Current Assets breakdown ─────────────────────────────────────────────
_r(AccountingTemplate(
    name="current_assets",
    description="Cash + Receivables + Inventory + Prepaid = Current Assets",
    headers=["Cash and Cash Equivalents", "Accounts Receivable", "Inventory",
             "Prepaid Expenses", "Current Assets"],
    constraints=[
        AccountingConstraint("ca_sum",
                             lhs=["Cash and Cash Equivalents", "Accounts Receivable",
                                  "Inventory", "Prepaid Expenses"],
                             rhs="Current Assets", omega=+1, op="add"),
    ],
))

# ── 2c. Non-Current Assets (PP&E, intangibles, investments) ───────────────────
_r(AccountingTemplate(
    name="non_current_assets",
    description="PP&E + Intangible Assets + Goodwill + Long-Term Investments "
                "+ Other Non-Current Assets = Non-Current Assets",
    headers=["Property Plant and Equipment", "Intangible Assets", "Goodwill",
             "Long-Term Investments", "Other Non-Current Assets",
             "Non-Current Assets"],
    constraints=[
        AccountingConstraint("nca_sum",
                             lhs=["Property Plant and Equipment", "Intangible Assets",
                                  "Goodwill", "Long-Term Investments",
                                  "Other Non-Current Assets"],
                             rhs="Non-Current Assets", omega=+1, op="add"),
    ],
))

# ── 2d. Property, Plant & Equipment (PP&E) roll-forward ────────────────────────
_r(AccountingTemplate(
    name="ppe_schedule",
    description="Gross PP&E − Accumulated Depreciation = Net PP&E",
    headers=["Gross Property Plant and Equipment", "Accumulated Depreciation",
             "Net Property Plant and Equipment"],
    constraints=[
        AccountingConstraint("net_ppe",
                             lhs=["Gross Property Plant and Equipment",
                                  "Accumulated Depreciation"],
                             rhs="Net Property Plant and Equipment",
                             omega=-1, op="sub"),
    ],
))

# ── 2e. Intangible Assets (Gross − Amortisation = Net) ───────────────────────
_r(AccountingTemplate(
    name="intangible_assets_schedule",
    description="Gross Intangibles − Accumulated Amortisation = Net Intangibles",
    headers=["Gross Intangible Assets", "Accumulated Amortisation",
             "Net Intangible Assets"],
    constraints=[
        AccountingConstraint("net_intangibles",
                             lhs=["Gross Intangible Assets", "Accumulated Amortisation"],
                             rhs="Net Intangible Assets", omega=-1, op="sub"),
    ],
))

# ── 2f. Right-of-Use Assets (IFRS 16 / ASC 842) ───────────────────────────────
_r(AccountingTemplate(
    name="rou_assets",
    description="ROU Assets from operating leases (IFRS 16 / ASC 842)",
    headers=["ROU Assets - Operating Leases", "ROU Assets - Finance Leases",
             "Accumulated Depreciation - ROU", "Net ROU Assets"],
    constraints=[
        AccountingConstraint("net_rou",
                             lhs=["ROU Assets - Operating Leases",
                                  "ROU Assets - Finance Leases",
                                  "Accumulated Depreciation - ROU"],
                             rhs="Net ROU Assets", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 3 — BALANCE SHEET: LIABILITIES & EQUITY
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. Total Liabilities = Current Liabilities + Non-Current Liabilities ────
_r(AccountingTemplate(
    name="total_liabilities",
    description="Current Liabilities + Non-Current Liabilities = Total Liabilities",
    headers=["Current Liabilities", "Non-Current Liabilities", "Total Liabilities"],
    constraints=[
        AccountingConstraint("total_liabilities",
                             lhs=["Current Liabilities", "Non-Current Liabilities"],
                             rhs="Total Liabilities", omega=+1, op="add"),
    ],
))

# ── 3b. Current Liabilities breakdown ───────────────────────────────────────
_r(AccountingTemplate(
    name="current_liabilities",
    description="Accounts Payable + Accrued Expenses + Short-Term Debt "
                "+ Current Tax Payable = Current Liabilities",
    headers=["Accounts Payable", "Accrued Expenses", "Short-Term Debt",
             "Current Tax Payable", "Current Liabilities"],
    constraints=[
        AccountingConstraint("cl_sum",
                             lhs=["Accounts Payable", "Accrued Expenses",
                                  "Short-Term Debt", "Current Tax Payable"],
                             rhs="Current Liabilities", omega=+1, op="add"),
    ],
))

# ── 3c. Non-Current Liabilities breakdown ────────────────────────────────────
_r(AccountingTemplate(
    name="non_current_liabilities",
    description="Long-Term Debt + Deferred Tax Liabilities + Other NCL = NCL",
    headers=["Long-Term Debt", "Deferred Tax Liabilities",
             "Other Non-Current Liabilities", "Non-Current Liabilities"],
    constraints=[
        AccountingConstraint("ncl_sum",
                             lhs=["Long-Term Debt", "Deferred Tax Liabilities",
                                  "Other Non-Current Liabilities"],
                             rhs="Non-Current Liabilities", omega=+1, op="add"),
    ],
))

# ── 3d. Lease Liabilities (IFRS 16 / ASC 842) ────────────────────────────────
_r(AccountingTemplate(
    name="lease_liabilities",
    description="Current Lease Liabilities + Non-Current Lease Liabilities = "
                "Total Lease Liabilities",
    headers=["Current Lease Liabilities", "Non-Current Lease Liabilities",
             "Total Lease Liabilities"],
    constraints=[
        AccountingConstraint("lease_liab",
                             lhs=["Current Lease Liabilities",
                                  "Non-Current Lease Liabilities"],
                             rhs="Total Lease Liabilities", omega=+1, op="add"),
    ],
))

# ── 3e. Total Debt (short-term + long-term) ───────────────────────────────────
_r(AccountingTemplate(
    name="total_debt",
    description="Short-Term Debt + Long-Term Debt = Total Debt",
    headers=["Short-Term Debt", "Long-Term Debt", "Total Debt"],
    constraints=[
        AccountingConstraint("total_debt",
                             lhs=["Short-Term Debt", "Long-Term Debt"],
                             rhs="Total Debt", omega=+1, op="add"),
    ],
))

# ── 3f. Shareholders' Equity (basic) ─────────────────────────────────────────
_r(AccountingTemplate(
    name="basic_equity",
    description="Common Stock + Additional Paid-In Capital + Retained Earnings "
                "= Total Equity",
    headers=["Common Stock", "Additional Paid-In Capital", "Retained Earnings",
             "Total Equity"],
    constraints=[
        AccountingConstraint("equity_sum",
                             lhs=["Common Stock", "Additional Paid-In Capital",
                                  "Retained Earnings"],
                             rhs="Total Equity", omega=+1, op="add"),
    ],
))

# ── 3g. Shareholders' Equity (full, with preferred stock & treasury) ──────────
_r(AccountingTemplate(
    name="full_equity",
    description="Common + Preferred + APIC + OCI + Treasury = Total Equity",
    headers=["Common Stock", "Preferred Stock", "Additional Paid-In Capital",
             "Retained Earnings", "Treasury Stock", "Accumulated Other Comprehensive Income",
             "Total Equity"],
    constraints=[
        AccountingConstraint("equity_components",
                             lhs=["Common Stock", "Preferred Stock",
                                  "Additional Paid-In Capital", "Retained Earnings",
                                  "Accumulated Other Comprehensive Income"],
                             rhs="Total Equity Before Treasury", omega=+1, op="add"),
        AccountingConstraint("treasury",
                             lhs=["Total Equity Before Treasury", "Treasury Stock"],
                             rhs="Total Equity", omega=-1, op="sub"),
    ],
))

# ── 3h. Balance Sheet Equation (the most fundamental identity) ────────────────
_r(AccountingTemplate(
    name="accounting_equation",
    description="Total Assets = Total Liabilities + Total Equity",
    headers=["Total Assets", "Total Liabilities", "Total Equity"],
    constraints=[
        AccountingConstraint("accounting_eq",
                             lhs=["Total Liabilities", "Total Equity"],
                             rhs="Total Assets", omega=+1, op="add"),
    ],
))

# ── 3i. Equity Reconciliation (year-over-year roll-forward) ──────────────────
_r(AccountingTemplate(
    name="equity_reconciliation",
    description="RE(t-1) + Net Income − Dividends = RE(t)",
    headers=["Retained Earnings Beginning of Year", "Net Income", "Dividends Paid",
             "Retained Earnings End of Year"],
    constraints=[
        AccountingConstraint("re_roll",
                             lhs=["Retained Earnings Beginning of Year",
                                  "Net Income", "Dividends Paid"],
                             rhs="Retained Earnings End of Year", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 4 — CASH FLOW STATEMENT
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. Cash Flow from Operations + Investing + Financing = Net Cash Flow ────
_r(AccountingTemplate(
    name="cash_flow_total",
    description="Operating CF + Investing CF + Financing CF = Net Cash Flow",
    headers=["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
             "Net Cash Flow"],
    constraints=[
        AccountingConstraint("net_cf",
                             lhs=["Operating Cash Flow", "Investing Cash Flow",
                                  "Financing Cash Flow"],
                             rhs="Net Cash Flow", omega=+1, op="add"),
    ],
))

# ── 4b. Operating Cash Flow (indirect method) ─────────────────────────────────
_r(AccountingTemplate(
    name="operating_cash_flow",
    description="Net Income + Non-Cash Charges + Working Capital Changes = OCF",
    headers=["Net Income", "Depreciation & Amortization", "Change in Accounts Receivable",
             "Change in Inventory", "Change in Accounts Payable",
             "Operating Cash Flow"],
    constraints=[
        AccountingConstraint("ocf_indirect",
                             lhs=["Net Income", "Depreciation & Amortization",
                                  "Change in Accounts Receivable", "Change in Inventory",
                                  "Change in Accounts Payable"],
                             rhs="Operating Cash Flow", omega=+1, op="add"),
    ],
))

# ── 4c. Free Cash Flow ────────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="free_cash_flow",
    description="Operating Cash Flow − Capital Expenditures = Free Cash Flow",
    headers=["Operating Cash Flow", "Capital Expenditures", "Free Cash Flow"],
    constraints=[
        AccountingConstraint("fcf",
                             lhs=["Operating Cash Flow", "Capital Expenditures"],
                             rhs="Free Cash Flow", omega=-1, op="sub"),
    ],
))

# ── 4d. Free Cash Flow to Firm (FCFF) ───────────────────────────────────────
_r(AccountingTemplate(
    name="fcff",
    description="EBIT − Taxes + D&A − ΔNWC − CapEx + Interest(1−t) = FCFF",
    headers=["EBIT", "Taxes on EBIT", "Depreciation & Amortization",
             "Change in Net Working Capital", "Capital Expenditures",
             "Interest Expense", "Free Cash Flow to Firm"],
    constraints=[
        AccountingConstraint("fcff_calc",
                             lhs=["EBIT", "Taxes on EBIT", "Depreciation & Amortization",
                                  "Change in Net Working Capital", "Capital Expenditures",
                                  "Interest Expense"],
                             rhs="Free Cash Flow to Firm", omega=-1, op="sub"),
    ],
))

# ── 4e. Cash reconciliation (beginning + net = ending) ───────────────────────
_r(AccountingTemplate(
    name="cash_reconciliation",
    description="Cash Beginning of Period + Net Cash Flow = Cash End of Period",
    headers=["Cash and Cash Equivalents Beginning of Period",
             "Net Cash Flow", "Cash and Cash Equivalents End of Period"],
    constraints=[
        AccountingConstraint("cash_recon",
                             lhs=["Cash and Cash Equivalents Beginning of Period",
                                  "Net Cash Flow"],
                             rhs="Cash and Cash Equivalents End of Period",
                             omega=+1, op="add"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 5 — EPS & PER-SHARE METRICS
# ══════════════════════════════════════════════════════════════════════════════

# ── 5a. Basic EPS ─────────────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="basic_eps",
    description="Net Income / Weighted Average Shares Outstanding = Basic EPS",
    headers=["Net Income", "Weighted Average Shares Outstanding", "Basic EPS"],
    constraints=[
        AccountingConstraint("basic_eps",
                             lhs=["Net Income", "Weighted Average Shares Outstanding"],
                             rhs="Basic EPS", omega=+1, op="div"),
    ],
))

# ── 5b. Diluted EPS ───────────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="diluted_eps",
    description="Net Income Attributable to Common / Diluted Shares = Diluted EPS",
    headers=["Net Income Attributable to Common Shareholders",
             "Diluted Weighted Average Shares", "Diluted EPS"],
    constraints=[
        AccountingConstraint("diluted_eps",
                             lhs=["Net Income Attributable to Common Shareholders",
                                  "Diluted Weighted Average Shares"],
                             rhs="Diluted EPS", omega=+1, op="div"),
    ],
))

# ── 5c. Book Value Per Share ─────────────────────────────────────────────────
_r(AccountingTemplate(
    name="book_value_per_share",
    description="Total Equity / Shares Outstanding = Book Value Per Share",
    headers=["Total Equity", "Shares Outstanding", "Book Value Per Share"],
    constraints=[
        AccountingConstraint("bvps",
                             lhs=["Total Equity", "Shares Outstanding"],
                             rhs="Book Value Per Share", omega=+1, op="div"),
    ],
))

# ── 5d. Tangible Book Value Per Share ───────────────────────────────────────
_r(AccountingTemplate(
    name="tangible_bvps",
    description="(Equity − Goodwill − Intangibles) / Shares = Tangible BVPS",
    headers=["Total Equity", "Goodwill", "Intangible Assets",
             "Tangible Book Value", "Shares Outstanding",
             "Tangible Book Value Per Share"],
    constraints=[
        AccountingConstraint("tbv",
                             lhs=["Total Equity", "Goodwill", "Intangible Assets"],
                             rhs="Tangible Book Value", omega=-1, op="sub"),
        AccountingConstraint("tbvps",
                             lhs=["Tangible Book Value", "Shares Outstanding"],
                             rhs="Tangible Book Value Per Share", omega=+1, op="div"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 6 — SEGMENT REPORTING
# ══════════════════════════════════════════════════════════════════════════════

# ── 6a. Revenue by Business Segment ─────────────────────────────────────────
_r(AccountingTemplate(
    name="segment_revenue",
    description="Segment_1 + ... + Segment_N = Total Revenue",
    headers=[],   # populated dynamically at match time
    constraints=[
        AccountingConstraint("segment_revenue_sum",
                             lhs=[], rhs="Total Revenue",
                             omega=+1, op="add"),
    ],
))

# ── 6b. Revenue by Geographic Segment ───────────────────────────────────────
_r(AccountingTemplate(
    name="geographic_revenue",
    description="Region_1 + ... + Region_N = Total Revenue (by geography)",
    headers=[],   # populated dynamically
    constraints=[
        AccountingConstraint("geo_revenue_sum",
                             lhs=[], rhs="Total Revenue",
                             omega=+1, op="add"),
    ],
))

# ── 6c. Segment Assets ────────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="segment_assets",
    description="Segment A Assets + Segment B Assets = Total Assets",
    headers=[],   # populated dynamically
    constraints=[
        AccountingConstraint("segment_assets_sum",
                             lhs=[], rhs="Total Assets",
                             omega=+1, op="add"),
    ],
))

# ── 6d. Segment Profitability ────────────────────────────────────────────────
_r(AccountingTemplate(
    name="segment_profit",
    description="Segment_1 Profit + ... + Segment_N Profit = Total Operating Profit",
    headers=[],   # populated dynamically
    constraints=[
        AccountingConstraint("segment_profit_sum",
                             lhs=[], rhs="Total Operating Profit",
                             omega=+1, op="add"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 7 — WORKING CAPITAL & LIQUIDITY RATIOS
# ══════════════════════════════════════════════════════════════════════════════

# ── 7a. Current Ratio ─────────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="current_ratio",
    description="Current Assets / Current Liabilities = Current Ratio",
    headers=["Current Assets", "Current Liabilities", "Current Ratio"],
    constraints=[
        AccountingConstraint("current_ratio",
                             lhs=["Current Assets", "Current Liabilities"],
                             rhs="Current Ratio", omega=+1, op="div"),
    ],
))

# ── 7b. Quick Ratio (Acid Test) ───────────────────────────────────────────────
_r(AccountingTemplate(
    name="quick_ratio",
    description="(Cash + Receivables) / Current Liabilities = Quick Ratio",
    headers=["Cash and Cash Equivalents", "Accounts Receivable",
             "Current Liabilities", "Quick Ratio"],
    constraints=[
        AccountingConstraint("quick_ratio",
                             lhs=["Cash and Cash Equivalents", "Accounts Receivable"],
                             rhs="Quick Ratio Numerator", omega=+1, op="add"),
    ],
))

# ── 7c. Working Capital ───────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="working_capital",
    description="Current Assets − Current Liabilities = Working Capital",
    headers=["Current Assets", "Current Liabilities", "Working Capital"],
    constraints=[
        AccountingConstraint("wc",
                             lhs=["Current Assets", "Current Liabilities"],
                             rhs="Working Capital", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 8 — DEBT & LEVERAGE RATIOS
# ══════════════════════════════════════════════════════════════════════════════

# ── 8a. Debt-to-Equity Ratio ─────────────────────────────────────────────────
_r(AccountingTemplate(
    name="debt_to_equity",
    description="Total Debt / Total Equity = Debt-to-Equity Ratio",
    headers=["Total Debt", "Total Equity", "Debt-to-Equity Ratio"],
    constraints=[
        AccountingConstraint("dte",
                             lhs=["Total Debt", "Total Equity"],
                             rhs="Debt-to-Equity Ratio", omega=+1, op="div"),
    ],
))

# ── 8b. Debt-to-Capital Ratio ────────────────────────────────────────────────
_r(AccountingTemplate(
    name="debt_to_capital",
    description="Total Debt / (Total Debt + Equity) = Debt-to-Capital",
    headers=["Total Debt", "Total Equity", "Total Capital", "Debt-to-Capital Ratio"],
    constraints=[
        AccountingConstraint("tdc",
                             lhs=["Total Debt", "Total Equity"],
                             rhs="Total Capital", omega=+1, op="add"),
        AccountingConstraint("dtc",
                             lhs=["Total Debt", "Total Capital"],
                             rhs="Debt-to-Capital Ratio", omega=+1, op="div"),
    ],
))

# ── 8c. Interest Coverage Ratio (ICR) ─────────────────────────────────────────
_r(AccountingTemplate(
    name="interest_coverage",
    description="EBIT / Interest Expense = Interest Coverage Ratio",
    headers=["EBIT", "Interest Expense", "Interest Coverage Ratio"],
    constraints=[
        AccountingConstraint("icr",
                             lhs=["EBIT", "Interest Expense"],
                             rhs="Interest Coverage Ratio", omega=+1, op="div"),
    ],
))

# ── 8d. Debt Schedule ────────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="debt_schedule",
    description="Beginning Balance + Issuances − Repayments = Ending Balance",
    headers=["Long-Term Debt Beginning of Period", "Long-Term Debt Issued",
             "Long-Term Debt Repaid", "Long-Term Debt End of Period"],
    constraints=[
        AccountingConstraint("debt_roll",
                             lhs=["Long-Term Debt Beginning of Period",
                                  "Long-Term Debt Issued", "Long-Term Debt Repaid"],
                             rhs="Long-Term Debt End of Period", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 9 — REVENUE & COGS BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

# ── 9a. Revenue by Product / Service Line ────────────────────────────────────
_r(AccountingTemplate(
    name="revenue_by_product",
    description="Product A Revenue + Product B Revenue = Total Revenue",
    headers=[],   # dynamic
    constraints=[
        AccountingConstraint("product_rev_sum",
                             lhs=[], rhs="Total Revenue",
                             omega=+1, op="add"),
    ],
))

# ── 9b. COGS breakdown ───────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="cogs_breakdown",
    description="Direct Materials + Direct Labor + Manufacturing Overhead = Total COGS",
    headers=["Direct Materials", "Direct Labor", "Manufacturing Overhead", "COGS"],
    constraints=[
        AccountingConstraint("cogs_sum",
                             lhs=["Direct Materials", "Direct Labor", "Manufacturing Overhead"],
                             rhs="COGS", omega=+1, op="add"),
    ],
))

# ── 9c. Revenue by Channel ────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="revenue_by_channel",
    description="Channel_1 Revenue + ... + Channel_N Revenue = Total Revenue",
    headers=[],
    constraints=[
        AccountingConstraint("channel_rev_sum",
                             lhs=[], rhs="Total Revenue",
                             omega=+1, op="add"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 10 — DEFERRED TAX
# ══════════════════════════════════════════════════════════════════════════════

# ── 10a. Total Income Tax Expense = Current Tax + Deferred Tax ───────────────
_r(AccountingTemplate(
    name="income_tax_expense",
    description="Current Tax Expense + Deferred Tax Expense = Total Income Tax Expense",
    headers=["Current Tax Expense", "Deferred Tax Expense",
             "Total Income Tax Expense"],
    constraints=[
        AccountingConstraint("tax_expense",
                             lhs=["Current Tax Expense", "Deferred Tax Expense"],
                             rhs="Total Income Tax Expense", omega=+1, op="add"),
    ],
))

# ── 10b. Effective Tax Rate reconciliation ─────────────────────────────────────
_r(AccountingTemplate(
    name="tax_rate_reconciliation",
    description="Tax at Statutory Rate ± Adjustments = Effective Tax Rate × EBT",
    headers=["Earnings Before Tax", "Tax at Statutory Rate",
             "Total Income Tax Expense", "Effective Tax Rate"],
    constraints=[
        AccountingConstraint("etr",
                             lhs=["Earnings Before Tax", "Effective Tax Rate"],
                             rhs="Total Income Tax Expense", omega=+1, op="div"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 11 — GOODWILL & BUSINESS COMBINATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 11a. Goodwill = Consideration Paid − Fair Value of Net Identifiable Assets ─
_r(AccountingTemplate(
    name="goodwill_calculation",
    description="Purchase Consideration − Fair Value Net Assets = Goodwill",
    headers=["Purchase Consideration", "Fair Value of Net Identifiable Assets",
             "Goodwill"],
    constraints=[
        AccountingConstraint("goodwill",
                             lhs=["Purchase Consideration",
                                  "Fair Value of Net Identifiable Assets"],
                             rhs="Goodwill", omega=-1, op="sub"),
    ],
))

# ── 11b. Purchase Price Allocation ───────────────────────────────────────────
_r(AccountingTemplate(
    name="purchase_price_allocation",
    description="Assets Acquired − Liabilities Assumed = Net Assets Acquired",
    headers=["Total Assets Acquired", "Total Liabilities Assumed",
             "Net Assets Acquired", "Goodwill", "Purchase Consideration Paid"],
    constraints=[
        AccountingConstraint("net_assets",
                             lhs=["Total Assets Acquired", "Total Liabilities Assumed"],
                             rhs="Net Assets Acquired", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 12 — LEASES (IFRS 16 / ASC 842)
# ══════════════════════════════════════════════════════════════════════════════

# ── 12a. Total Lease Liability = Current + Non-Current ───────────────────────
_r(AccountingTemplate(
    name="total_lease_liability",
    description="Current Lease Liabilities + Non-Current Lease Liabilities = Total Lease Liabilities",
    headers=["Current Lease Liabilities", "Non-Current Lease Liabilities",
             "Total Lease Liabilities"],
    constraints=[
        AccountingConstraint("total_lease_liab",
                             lhs=["Current Lease Liabilities",
                                  "Non-Current Lease Liabilities"],
                             rhs="Total Lease Liabilities", omega=+1, op="add"),
    ],
))

# ── 12b. ROU Asset schedule ───────────────────────────────────────────────────
_r(AccountingTemplate(
    name="rou_schedule",
    description="ROU Asset Beginning + New Leases − Depreciation = ROU Asset Ending",
    headers=["ROU Assets Beginning of Period", "New ROU Assets",
             "Depreciation of ROU Assets", "ROU Assets End of Period"],
    constraints=[
        AccountingConstraint("rou_roll",
                             lhs=["ROU Assets Beginning of Period", "New ROU Assets",
                                  "Depreciation of ROU Assets"],
                             rhs="ROU Assets End of Period", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 13 — INVENTORY
# ══════════════════════════════════════════════════════════════════════════════

# ── 13a. Manufacturing Inventory (Raw + WIP + FG) ─────────────────────────────
_r(AccountingTemplate(
    name="inventory_manufacturing",
    description="Raw Materials + Work in Progress + Finished Goods = Total Inventory",
    headers=["Raw Materials", "Work in Progress", "Finished Goods",
             "Total Inventory"],
    constraints=[
        AccountingConstraint("inv_manufacturing",
                             lhs=["Raw Materials", "Work in Progress", "Finished Goods"],
                             rhs="Total Inventory", omega=+1, op="add"),
    ],
))

# ── 13b. Inventory COGS relationship (periodic system) ───────────────────────
_r(AccountingTemplate(
    name="inventory_cogs_periodic",
    description="Beginning Inventory + Purchases − Ending Inventory = COGS",
    headers=["Beginning Inventory", "Purchases", "Ending Inventory", "COGS"],
    constraints=[
        AccountingConstraint("cogs_periodic",
                             lhs=["Beginning Inventory", "Purchases", "Ending Inventory"],
                             rhs="COGS", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 14 — EQUITY & DIVIDENDS
# ══════════════════════════════════════════════════════════════════════════════

# ── 14a. Dividends Paid reconciliation ────────────────────────────────────────
_r(AccountingTemplate(
    name="dividends_paid",
    description="Dividends Declared − Change in Dividends Payable = Cash Dividends Paid",
    headers=["Dividends Declared", "Change in Dividends Payable",
             "Cash Dividends Paid"],
    constraints=[
        AccountingConstraint("div_paid",
                             lhs=["Dividends Declared", "Change in Dividends Payable"],
                             rhs="Cash Dividends Paid", omega=-1, op="sub"),
    ],
))

# ── 14b. Share Repurchases ────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="share_repurchases",
    description="Shares Repurchased × Price = Total Repurchase Cost",
    headers=["Shares Repurchased", "Average Repurchase Price",
             "Total Share Repurchase Cost"],
    constraints=[
        AccountingConstraint("repurchase",
                             lhs=["Shares Repurchased", "Average Repurchase Price"],
                             rhs="Total Share Repurchase Cost", omega=+1, op="div"),
    ],
))

# ── 14c. Stock-Based Compensation ─────────────────────────────────────────────
_r(AccountingTemplate(
    name="stock_compensation",
    description="Stock Options Exercised × Exercise Price + RSUs Vested = Proceeds",
    headers=["Stock Options Exercised", "Exercise Price", "RSUs Vested",
             "Total Shares Issued", "Proceeds from Share Issuance"],
    constraints=[
        AccountingConstraint("sbc_proceeds",
                             lhs=["Stock Options Exercised", "Exercise Price"],
                             rhs="Option Proceeds", omega=+1, op="div"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 15 — OTHER COMPREHENSIVE INCOME (OCI)
# ══════════════════════════════════════════════════════════════════════════════

# ── 15a. Total Comprehensive Income = Net Income + OCI ────────────────────────
_r(AccountingTemplate(
    name="total_comprehensive_income",
    description="Net Income + Other Comprehensive Income = Total Comprehensive Income",
    headers=["Net Income", "Foreign Currency Translation Adjustment",
             "Unrealised Gain/Loss on Securities", "Pension Adjustment",
             "Cash Flow Hedge Adjustment", "Total Comprehensive Income"],
    constraints=[
        AccountingConstraint("tci",
                             lhs=["Net Income", "Foreign Currency Translation Adjustment",
                                  "Unrealised Gain/Loss on Securities",
                                  "Pension Adjustment", "Cash Flow Hedge Adjustment"],
                             rhs="Total Comprehensive Income", omega=+1, op="add"),
    ],
))

# ── 15b. Accumulated OCI roll-forward ─────────────────────────────────────────
_r(AccountingTemplate(
    name="accumulated_oci",
    description="Accumulated OCI(t-1) + Current Period OCI = Accumulated OCI(t)",
    headers=["Accumulated Other Comprehensive Income Beginning",
             "Current Period Other Comprehensive Income",
             "Accumulated Other Comprehensive Income End"],
    constraints=[
        AccountingConstraint("aoci_roll",
                             lhs=["Accumulated Other Comprehensive Income Beginning",
                                  "Current Period Other Comprehensive Income"],
                             rhs="Accumulated Other Comprehensive Income End",
                             omega=+1, op="add"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 16 — MARGIN RATIOS (soft — detected by header, no hard constraint)
# ══════════════════════════════════════════════════════════════════════════════

# ── 16a. Gross Margin ─────────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="gross_margin",
    description="Gross Profit / Revenue = Gross Margin",
    headers=["Revenue", "Gross Profit", "Gross Margin"],
    constraints=[
        AccountingConstraint("gross_margin_def",
                             lhs=["Gross Profit", "Revenue"],
                             rhs="Gross Margin", omega=+1, op="div"),
    ],
))

# ── 16b. Operating Margin ─────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="operating_margin",
    description="Operating Income / Revenue = Operating Margin",
    headers=["Revenue", "Operating Income", "Operating Margin"],
    constraints=[
        AccountingConstraint("op_margin_def",
                             lhs=["Operating Income", "Revenue"],
                             rhs="Operating Margin", omega=+1, op="div"),
    ],
))

# ── 16c. Net Profit Margin ────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="net_margin",
    description="Net Income / Revenue = Net Profit Margin",
    headers=["Revenue", "Net Income", "Net Profit Margin"],
    constraints=[
        AccountingConstraint("net_margin_def",
                             lhs=["Net Income", "Revenue"],
                             rhs="Net Profit Margin", omega=+1, op="div"),
    ],
))

# ── 16d. EBITDA Margin ─────────────────────────────────────────────────────────
_r(AccountingTemplate(
    name="ebitda_margin",
    description="EBITDA / Revenue = EBITDA Margin",
    headers=["Revenue", "EBITDA", "EBITDA Margin"],
    constraints=[
        AccountingConstraint("ebitda_margin_def",
                             lhs=["EBITDA", "Revenue"],
                             rhs="EBITDA Margin", omega=+1, op="div"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 17 — QUARTERLY & YEAR-OVER-YEAR (soft — period comparison tables)
# ══════════════════════════════════════════════════════════════════════════════

# ── 17a. Quarterly to Annual ───────────────────────────────────────────────────
_r(AccountingTemplate(
    name="quarterly_breakdown",
    description="Q1 + Q2 + Q3 + Q4 = Annual Total",
    headers=["Q1", "Q2", "Q3", "Q4", "Annual", "Full Year"],
    constraints=[
        AccountingConstraint("annual_total",
                             lhs=["Q1", "Q2", "Q3", "Q4"],
                             rhs="Annual", omega=+1, op="add"),
    ],
))

# ── 17b. Year-over-Year Change ────────────────────────────────────────────────
_r(AccountingTemplate(
    name="yoy_change",
    description="YoY change table: [Year_N, Value] pairs with % change",
    headers=[],   # dynamic: populated at match time
    constraints=[],
))

# ── 17c. Multi-Period Balance Sheet ───────────────────────────────────────────
_r(AccountingTemplate(
    name="multi_period_bs",
    description="Same balance sheet line items across 3 years",
    headers=[],
    constraints=[],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 18 — PENSION & DEFINED BENEFIT (IAS 19 / ASC 715)
# ══════════════════════════════════════════════════════════════════════════════

# ── 18a. Net Pension Liability / Asset ────────────────────────────────────────
_r(AccountingTemplate(
    name="pension_obligation",
    description="Projected Benefit Obligation − Plan Assets = Funded Status",
    headers=["Projected Benefit Obligation", "Plan Assets at Fair Value",
             "Funded Status", "Net Pension Liability"],
    constraints=[
        AccountingConstraint("pension_funded",
                             lhs=["Projected Benefit Obligation", "Plan Assets at Fair Value"],
                             rhs="Funded Status", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 19 — REVENUE RECOGNITION (IFRS 15 / ASC 606)
# ══════════════════════════════════════════════════════════════════════════════

# ── 19a. Contract Assets & Liabilities ────────────────────────────────────────
_r(AccountingTemplate(
    name="contract_balances",
    description="Contract Assets + Accounts Receivable = Total Billed & Unbilled",
    headers=["Contract Assets", "Accounts Receivable", "Contract Liabilities",
             "Net Contract Position"],
    constraints=[
        AccountingConstraint("contract_assets",
                             lhs=["Contract Assets", "Accounts Receivable"],
                             rhs="Total Billed and Unbilled", omega=+1, op="add"),
        AccountingConstraint("net_contract",
                             lhs=["Total Billed and Unbilled", "Contract Liabilities"],
                             rhs="Net Contract Position", omega=-1, op="sub"),
    ],
))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 20 — RELATED PARTY & OFF-BALANCE SHEET (soft, for matching)
# ══════════════════════════════════════════════════════════════════════════════

_r(AccountingTemplate(
    name="off_balance_sheet",
    description="Contingent liabilities, guarantees, commitments (off-B/S)",
    headers=["Contingent Liabilities", "Guarantees Outstanding",
             "Commitments under Contract", "Operating Lease Commitments"],
    constraints=[],
))

_r(AccountingTemplate(
    name="related_party",
    description="Related party transactions: key management compensation, affiliates",
    headers=["Key Management Compensation", "Transactions with Affiliates",
             "Related Party Receivables", "Related Party Payables"],
    constraints=[],
))

_r(AccountingTemplate(
    name="contingencies",
    description="Litigation reserves and contingent liabilities",
    headers=["Litigation Reserve Beginning", "Litigation Reserve Added",
             "Litigation Reserve Used", "Litigation Reserve Ending"],
    constraints=[
        AccountingConstraint("litigation_roll",
                             lhs=["Litigation Reserve Beginning",
                                  "Litigation Reserve Added",
                                  "Litigation Reserve Used"],
                             rhs="Litigation Reserve Ending", omega=-1, op="sub"),
    ],
))
