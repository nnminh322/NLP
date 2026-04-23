"""Template matching: header normalisation, fuzzy matching, and synonym tables.

Expands the synonym map to cover all headers in the complete IFRS/GAAP
template library (~75 templates, 250+ header variants).
"""

from __future__ import annotations

from gsr_cacl.templates.data_structures import AccountingTemplate
from gsr_cacl.templates.library import TEMPLATES


# ---------------------------------------------------------------------------
# Canonical header name → synonym variants (all lower-case keys)
# ---------------------------------------------------------------------------

_HEADER_SYNONYMS: dict[str, str] = {

    # ── Revenue ──────────────────────────────────────────────────────────────
    "revenue": "Revenue",
    "total revenue": "Revenue",
    "net revenue": "Revenue",
    "net sales": "Revenue",
    "total sales": "Revenue",
    "sales": "Revenue",
    "sales revenue": "Revenue",
    "operating revenue": "Revenue",
    "interest income": "Interest Income",
    "total interest income": "Interest Income",
    "other income": "Other Income",
    "other revenue": "Other Income",

    # ── COGS ──────────────────────────────────────────────────────────────────
    "cogs": "COGS",
    "cost of revenue": "COGS",
    "cost of goods sold": "COGS",
    "cost of sales": "COGS",
    "direct materials": "Direct Materials",
    "direct labour": "Direct Labor",
    "direct labor": "Direct Labor",
    "manufacturing overhead": "Manufacturing Overhead",
    "manufacturing overheads": "Manufacturing Overhead",
    "raw materials": "Raw Materials",
    "work in progress": "Work in Progress",
    "work-in-progress": "Work in Progress",
    "wip": "Work in Progress",
    "finished goods": "Finished Goods",
    "fg": "Finished Goods",
    "purchases": "Purchases",
    "beginning inventory": "Beginning Inventory",
    "ending inventory": "Ending Inventory",
    "inventory writedown": "Ending Inventory",

    # ── Gross Profit ──────────────────────────────────────────────────────────
    "gross profit": "Gross Profit",
    "gross income": "Gross Profit",
    "gross margin": "Gross Margin",

    # ── Operating Expenses ────────────────────────────────────────────────────
    "operating expenses": "Operating Expenses",
    "opex": "Operating Expenses",
    "sga": "SG&A",
    "sg&a": "SG&A",
    "sg and a": "SG&A",
    "selling general and administrative": "SG&A",
    "selling, general and administrative": "SG&A",
    "selling, general and administrative expenses": "SG&A",
    "research and development": "Operating Expenses",
    "r&d": "Operating Expenses",
    "rd expenses": "Operating Expenses",
    "marketing expenses": "Operating Expenses",
    "general and administrative": "Operating Expenses",
    "g&a": "Operating Expenses",

    # ── D&A ────────────────────────────────────────────────────────────────────
    "depreciation": "Depreciation & Amortization",
    "depreciation and amortization": "Depreciation & Amortization",
    "d&a": "Depreciation & Amortization",
    "da": "Depreciation & Amortization",
    "amortisation": "Depreciation & Amortization",
    "amortisation of intangible assets": "Depreciation & Amortization",
    "depreciation of pp&e": "Depreciation & Amortization",

    # ── EBIT / Operating Income ───────────────────────────────────────────────
    "operating income": "Operating Income",
    "operating profit": "Operating Income",
    "ebit": "Operating Income",
    "income from operations": "Operating Income",
    "earnings before interest and taxes": "Operating Income",
    "operating income - continuing operations": "Operating Income",
    "ebitda": "EBITDA",
    "ebitdat": "EBITDA",
    "earnings before interest, taxes, depreciation and amortisation": "EBITDA",

    # ── Interest ──────────────────────────────────────────────────────────────
    "interest expense": "Interest Expense",
    "interest income": "Interest Income",
    "interest income (expense)": "Interest Expense",
    "net interest expense": "Interest Expense",
    "net interest income": "Interest Income",
    "interest expense net of tax": "Interest Expense",

    # ── EBT / Pre-tax income ─────────────────────────────────────────────────
    "earnings before tax": "Earnings Before Tax",
    "ebt": "Earnings Before Tax",
    "pre-tax income": "Earnings Before Tax",
    "income before income tax": "Earnings Before Tax",
    "income before tax": "Earnings Before Tax",
    "profit before tax": "Earnings Before Tax",
    "profit before taxation": "Earnings Before Tax",

    # ── Tax ───────────────────────────────────────────────────────────────────
    "income tax": "Income Tax",
    "income tax expense": "Income Tax",
    "tax expense": "Income Tax",
    "tax": "Income Tax",
    "total income tax expense": "Total Income Tax Expense",
    "current tax expense": "Current Tax Expense",
    "current tax": "Current Tax Expense",
    "deferred tax expense": "Deferred Tax Expense",
    "deferred tax": "Deferred Tax Expense",
    "taxes on ebit": "Taxes on EBIT",
    "effective tax rate": "Effective Tax Rate",
    "tax at statutory rate": "Tax at Statutory Rate",
    "tax rate reconciliation": "Effective Tax Rate",

    # ── Net Income ────────────────────────────────────────────────────────────
    "net income": "Net Income",
    "net profit": "Net Income",
    "net earnings": "Net Income",
    "profit for the period": "Net Income",
    "profit for the year": "Net Income",
    "net income attributable to common shareholders": "Net Income Attributable to Common Shareholders",
    "net income attributable to parent": "Net Income Attributable to Parent",
    "net income attributable to equity holders": "Net Income Attributable to Parent",
    "income from discontinued operations": "Income from Discontinued Operations",
    "discontinued operations": "Income from Discontinued Operations",
    "net income before disc ops": "Net Income Before Disc Ops",

    # ── EPS ───────────────────────────────────────────────────────────────────
    "earnings per share": "EPS",
    "eps": "EPS",
    "basic earnings per share": "Basic EPS",
    "basic eps": "Basic EPS",
    "diluted earnings per share": "Diluted EPS",
    "diluted eps": "Diluted EPS",
    "net income per share": "EPS",
    "earnings per share (basic)": "Basic EPS",
    "earnings per share (diluted)": "Diluted EPS",
    "shares outstanding": "Shares Outstanding",
    "weighted average shares outstanding": "Weighted Average Shares Outstanding",
    "weighted average shares": "Weighted Average Shares Outstanding",
    "diluted weighted average shares": "Diluted Weighted Average Shares",
    "diluted shares": "Diluted Weighted Average Shares",

    # ── Comprehensive Income ─────────────────────────────────────────────────
    "other comprehensive income": "Other Comprehensive Income",
    "oci": "Other Comprehensive Income",
    "total comprehensive income": "Total Comprehensive Income",
    "foreign currency translation adjustment": "Foreign Currency Translation Adjustment",
    "fcta": "Foreign Currency Translation Adjustment",
    "currency translation": "Foreign Currency Translation Adjustment",
    "unrealised gain/loss on securities": "Unrealised Gain/Loss on Securities",
    "unrealized gain/loss on securities": "Unrealised Gain/Loss on Securities",
    "unrealized gains/losses": "Unrealised Gain/Loss on Securities",
    "pension adjustment": "Pension Adjustment",
    "remeasurement of defined benefit plans": "Pension Adjustment",
    "cash flow hedge adjustment": "Cash Flow Hedge Adjustment",
    "accumulated other comprehensive income": "Accumulated Other Comprehensive Income",
    "accumulated oci": "Accumulated Other Comprehensive Income",
    "aoci": "Accumulated Other Comprehensive Income",
    "accumulated other comprehensive income beginning": "Accumulated Other Comprehensive Income Beginning",
    "current period other comprehensive income": "Current Period Other Comprehensive Income",
    "accumulated other comprehensive income end": "Accumulated Other Comprehensive Income End",

    # ── Assets (Current) ──────────────────────────────────────────────────────
    "current assets": "Current Assets",
    "total current assets": "Current Assets",
    "cash and cash equivalents": "Cash and Cash Equivalents",
    "cash": "Cash and Cash Equivalents",
    "cash & cash equivalents": "Cash and Cash Equivalents",
    "restricted cash": "Cash and Cash Equivalents",
    "accounts receivable": "Accounts Receivable",
    "trade receivables": "Accounts Receivable",
    "trade accounts receivable": "Accounts Receivable",
    "ar": "Accounts Receivable",
    "inventory": "Inventory",
    "inventories": "Inventory",
    "total inventory": "Total Inventory",
    "prepaid expenses": "Prepaid Expenses",
    "prepaid": "Prepaid Expenses",
    "other current assets": "Other Current Assets",
    "short-term investments": "Other Current Assets",
    "held-to-maturity investments": "Other Current Assets",
    "contract assets": "Contract Assets",
    "unbilled receivables": "Contract Assets",

    # ── Assets (Non-Current) ─────────────────────────────────────────────────
    "non-current assets": "Non-Current Assets",
    "total non-current assets": "Non-Current Assets",
    "non current assets": "Non-Current Assets",
    "property plant and equipment": "Property Plant and Equipment",
    "property, plant and equipment": "Property Plant and Equipment",
    "pp&e": "Property Plant and Equipment",
    "ppe": "Property Plant and Equipment",
    "fixed assets": "Property Plant and Equipment",
    "tangible assets": "Property Plant and Equipment",
    "gross property plant and equipment": "Gross Property Plant and Equipment",
    "accumulated depreciation": "Accumulated Depreciation",
    "accumulated amortisation": "Accumulated Amortisation",
    "net property plant and equipment": "Net Property Plant and Equipment",
    "net pp&e": "Net Property Plant and Equipment",
    "intangible assets": "Intangible Assets",
    "intangibles": "Intangible Assets",
    "gross intangible assets": "Gross Intangible Assets",
    "accumulated amortisation": "Accumulated Amortisation",
    "net intangible assets": "Net Intangible Assets",
    "goodwill": "Goodwill",
    "long-term investments": "Long-Term Investments",
    "investments": "Long-Term Investments",
    "investment in associates": "Long-Term Investments",
    "other non-current assets": "Other Non-Current Assets",
    "ROU assets": "ROU Assets - Operating Leases",
    "rou assets": "ROU Assets - Operating Leases",
    "right-of-use assets": "ROU Assets - Operating Leases",
    "operating lease right-of-use assets": "ROU Assets - Operating Leases",
    "finance lease right-of-use assets": "ROU Assets - Finance Leases",
    "rou assets - operating leases": "ROU Assets - Operating Leases",
    "rou assets - finance leases": "ROU Assets - Finance Leases",
    "accumulated depreciation - rou": "Accumulated Depreciation - ROU",
    "net rou assets": "Net ROU Assets",
    "total assets": "Total Assets",
    "total assets acquired": "Total Assets Acquired",

    # ── Liabilities (Current) ─────────────────────────────────────────────────
    "current liabilities": "Current Liabilities",
    "total current liabilities": "Current Liabilities",
    "accounts payable": "Accounts Payable",
    "trade payables": "Accounts Payable",
    "ap": "Accounts Payable",
    "accrued expenses": "Accrued Expenses",
    "accrued liabilities": "Accrued Expenses",
    "short-term debt": "Short-Term Debt",
    "st debt": "Short-Term Debt",
    "current portion of long-term debt": "Short-Term Debt",
    "short-term borrowings": "Short-Term Debt",
    "current tax payable": "Current Tax Payable",
    "tax payable": "Current Tax Payable",
    "current income tax payable": "Current Tax Payable",
    "contract liabilities": "Contract Liabilities",
    "deferred revenue": "Contract Liabilities",
    "other current liabilities": "Other Current Liabilities",
    "dividends payable": "Dividends Payable",
    "change in dividends payable": "Change in Dividends Payable",

    # ── Liabilities (Non-Current) ─────────────────────────────────────────────
    "non-current liabilities": "Non-Current Liabilities",
    "total non-current liabilities": "Non-Current Liabilities",
    "non current liabilities": "Non-Current Liabilities",
    "long-term debt": "Long-Term Debt",
    "lt debt": "Long-Term Debt",
    "long term debt": "Long-Term Debt",
    "bonds payable": "Long-Term Debt",
    "deferred tax liabilities": "Deferred Tax Liabilities",
    "dtl": "Deferred Tax Liabilities",
    "other non-current liabilities": "Other Non-Current Liabilities",
    "lease liabilities": "Total Lease Liabilities",
    "current lease liabilities": "Current Lease Liabilities",
    "non-current lease liabilities": "Non-Current Lease Liabilities",
    "operating lease liabilities": "Current Lease Liabilities",
    "total liabilities": "Total Liabilities",
    "total liabilities assumed": "Total Liabilities Assumed",
    "net assets acquired": "Net Assets Acquired",

    # ── Equity ────────────────────────────────────────────────────────────────
    "total equity": "Total Equity",
    "total shareholders equity": "Total Equity",
    "total stockholders equity": "Total Equity",
    "shareholders equity": "Total Equity",
    "stockholders equity": "Total Equity",
    "total equity attributable to shareholders": "Total Equity",
    "common stock": "Common Stock",
    "ordinary shares": "Common Stock",
    "preferred stock": "Preferred Stock",
    "preference shares": "Preferred Stock",
    "additional paid-in capital": "Additional Paid-In Capital",
    "additional paid in capital": "Additional Paid-In Capital",
    "apic": "Additional Paid-In Capital",
    "share premium": "Additional Paid-In Capital",
    "retained earnings": "Retained Earnings",
    "accumulated profits": "Retained Earnings",
    "retained earnings beginning of year": "Retained Earnings Beginning of Year",
    "beginning retained earnings": "Retained Earnings Beginning of Year",
    "retained earnings end of year": "Retained Earnings End of Year",
    "ending retained earnings": "Retained Earnings End of Year",
    "treasury stock": "Treasury Stock",
    "treasury shares": "Treasury Stock",
    "own shares": "Treasury Stock",
    "net assets": "Total Equity",
    "total capital": "Total Capital",

    # ── Cash Flow ──────────────────────────────────────────────────────────────
    "operating cash flow": "Operating Cash Flow",
    "cash from operations": "Operating Cash Flow",
    "cash generated from operations": "Operating Cash Flow",
    "net cash from operating activities": "Operating Cash Flow",
    "ocf": "Operating Cash Flow",
    "cfo": "Operating Cash Flow",
    "investing cash flow": "Investing Cash Flow",
    "cash used in investing activities": "Investing Cash Flow",
    "net cash used in investing activities": "Investing Cash Flow",
    "financing cash flow": "Financing Cash Flow",
    "cash used in financing activities": "Financing Cash Flow",
    "net cash from financing activities": "Financing Cash Flow",
    "net cash flow": "Net Cash Flow",
    "net increase in cash": "Net Cash Flow",
    "net change in cash": "Net Cash Flow",
    "free cash flow": "Free Cash Flow",
    "fcf": "Free Cash Flow",
    "free cash flow to firm": "Free Cash Flow to Firm",
    "fcff": "Free Cash Flow to Firm",
    "capital expenditures": "Capital Expenditures",
    "capex": "Capital Expenditures",
    "acquisition of property, plant and equipment": "Capital Expenditures",
    "change in net working capital": "Change in Net Working Capital",
    "change in accounts receivable": "Change in Accounts Receivable",
    "change in inventory": "Change in Inventory",
    "change in accounts payable": "Change in Accounts Payable",
    "non-cash charges": "Depreciation & Amortization",
    "non-cash items": "Depreciation & Amortization",
    "cash and cash equivalents beginning of period": "Cash and Cash Equivalents Beginning of Period",
    "cash and cash equivalents end of period": "Cash and Cash Equivalents End of Period",
    "beginning cash": "Cash and Cash Equivalents Beginning of Period",
    "ending cash": "Cash and Cash Equivalents End of Period",

    # ── Segment ───────────────────────────────────────────────────────────────
    "segment revenue": "Segment Revenue",
    "segment profit": "Segment Profit",
    "segment assets": "Segment Assets",
    "total operating profit": "Total Operating Profit",
    "geographic revenue": "Revenue",

    # ── Working Capital & Liquidity ────────────────────────────────────────────
    "working capital": "Working Capital",
    "net working capital": "Working Capital",
    "current ratio": "Current Ratio",
    "quick ratio": "Quick Ratio",
    "acid test ratio": "Quick Ratio",
    "quick ratio numerator": "Quick Ratio Numerator",

    # ── Debt & Leverage ───────────────────────────────────────────────────────
    "total debt": "Total Debt",
    "total debt": "Total Debt",
    "debt-to-equity ratio": "Debt-to-Equity Ratio",
    "debt to equity": "Debt-to-Equity Ratio",
    "debt to equity ratio": "Debt-to-Equity Ratio",
    "debt-to-capital ratio": "Debt-to-Capital Ratio",
    "debt to capital": "Debt-to-Capital Ratio",
    "interest coverage ratio": "Interest Coverage Ratio",
    "interest coverage": "Interest Coverage Ratio",
    "long-term debt beginning of period": "Long-Term Debt Beginning of Period",
    "lt debt beginning": "Long-Term Debt Beginning of Period",
    "long-term debt issued": "Long-Term Debt Issued",
    "debt issued": "Long-Term Debt Issued",
    "long-term debt repaid": "Long-Term Debt Repaid",
    "debt repaid": "Long-Term Debt Repaid",
    "long-term debt end of period": "Long-Term Debt End of Period",
    "lt debt ending": "Long-Term Debt End of Period",

    # ── Book Value ────────────────────────────────────────────────────────────
    "book value per share": "Book Value Per Share",
    "tangible book value per share": "Tangible Book Value Per Share",
    "tangible book value": "Tangible Book Value",
    "tbvps": "Tangible Book Value Per Share",

    # ── Purchase Price Allocation / Goodwill ──────────────────────────────────
    "purchase consideration": "Purchase Consideration",
    "purchase consideration paid": "Purchase Consideration Paid",
    "fair value of net identifiable assets": "Fair Value of Net Identifiable Assets",
    "fair value net assets": "Fair Value of Net Identifiable Assets",
    "fair value of net assets acquired": "Fair Value of Net Identifiable Assets",
    "fair value of identifiable net assets": "Fair Value of Net Identifiable Assets",

    # ── ROU / Lease schedules ─────────────────────────────────────────────────
    "new rou assets": "New ROU Assets",
    "rou assets beginning of period": "ROU Assets Beginning of Period",
    "rou assets end of period": "ROU Assets End of Period",
    "depreciation of rou assets": "Depreciation of ROU Assets",
    "total lease liabilities": "Total Lease Liabilities",
    "new leases": "New ROU Assets",
    "proceeds from share issuance": "Proceeds from Share Issuance",
    "option proceeds": "Option Proceeds",

    # ── Dividends & Equity transactions ───────────────────────────────────────
    "dividends declared": "Dividends Declared",
    "dividends paid": "Dividends Paid",
    "cash dividends paid": "Cash Dividends Paid",
    "dividends per share": "Dividends Paid",
    "share repurchased": "Shares Repurchased",
    "shares repurchased": "Shares Repurchased",
    "average repurchase price": "Average Repurchase Price",
    "total share repurchase cost": "Total Share Repurchase Cost",
    "stock options exercised": "Stock Options Exercised",
    "exercise price": "Exercise Price",
    "rsus vested": "RSUs Vested",
    "restricted stock units vested": "RSUs Vested",
    "total shares issued": "Total Shares Issued",

    # ── Pension ────────────────────────────────────────────────────────────────
    "projected benefit obligation": "Projected Benefit Obligation",
    "pbo": "Projected Benefit Obligation",
    "plan assets at fair value": "Plan Assets at Fair Value",
    "plan assets": "Plan Assets at Fair Value",
    "funded status": "Funded Status",
    "net pension liability": "Net Pension Liability",

    # ── Contract Balances (IFRS 15 / ASC 606) ─────────────────────────────────
    "total billed and unbilled": "Total Billed and Unbilled",
    "net contract position": "Net Contract Position",

    # ── Contingencies ─────────────────────────────────────────────────────────
    "litigation reserve beginning": "Litigation Reserve Beginning",
    "litigation reserve added": "Litigation Reserve Added",
    "litigation reserve used": "Litigation Reserve Used",
    "litigation reserve ending": "Litigation Reserve Ending",
    "contingent liabilities": "Contingent Liabilities",
    "contingent liabilities and commitments": "Contingent Liabilities",
    "guarantees outstanding": "Guarantees Outstanding",
    "commitments under contract": "Commitments under Contract",
    "operating lease commitments": "Operating Lease Commitments",
    "capital commitments": "Commitments under Contract",

    # ── Related Party ─────────────────────────────────────────────────────────
    "key management compensation": "Key Management Compensation",
    "transactions with affiliates": "Transactions with Affiliates",
    "related party receivables": "Related Party Receivables",
    "related party payables": "Related Party Payables",

    # ── Margins ───────────────────────────────────────────────────────────────
    "gross margin": "Gross Margin",
    "operating margin": "Operating Margin",
    "net profit margin": "Net Profit Margin",
    "net margin": "Net Profit Margin",
    "ebitda margin": "EBITDA Margin",

    # ── Quarterly / Period ─────────────────────────────────────────────────────
    "q1": "Q1",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Q4",
    "quarter 1": "Q1",
    "quarter 2": "Q2",
    "quarter 3": "Q3",
    "quarter 4": "Q4",
    "h1": "Q1",
    "h2": "Q2",
    "first half": "Q1",
    "second half": "Q2",
    "annual": "Annual",
    "full year": "Full Year",
    "fiscal year": "Annual",
    "yoy change": "YoY Change",
    "year-over-year change": "YoY Change",
    "period over period": "YoY Change",

    # ── Miscellaneous / Soft ──────────────────────────────────────────────────
    "total": "Total",
    "subtotal": "Subtotal",
    "total assets before treasury": "Total Equity Before Treasury",
    "equity before treasury": "Total Equity Before Treasury",
    "change in net assets": "Net Cash Flow",
    "non-controlling interest": "Non-Controlling Interest",
    "nci": "Non-Controlling Interest",
    "net income attributable to parent": "Net Income Attributable to Parent",
    "parent": "Net Income Attributable to Parent",
}


def normalize_header(h: str) -> str:
    """Map a header string to its canonical form via the synonym table."""
    h_lower = h.lower().strip()
    if h_lower in _HEADER_SYNONYMS:
        return _HEADER_SYNONYMS[h_lower]
    # Fallback: title-case the header if no synonym match
    return " ".join(w.capitalize() for w in h.split())


def _fuzzy_match_headers(
    table_headers: list[str],
    template: AccountingTemplate,
) -> float:
    """
    Compute a 0–1 confidence score between *table_headers* and *template*.

    Uses two signals:
        1. Canonical synonym matches  (strict — exact canonical name)
        2. Substring coverage          (lenient — partial header overlap)

    Score = (# matched canonicals) / max(len(table), len(template.headers))
    A template with empty headers (dynamic / period-comparison) gets a flat 0.3
    so it is never selected over a concrete match.
    """
    if not template.headers:
        return 0.3   # soft template — low priority

    table_canonical = [normalize_header(h) for h in table_headers]

    # Count canonical exact matches
    matched = sum(
        1 for th in template.headers
        if th in table_canonical
    )

    # Bonus: substring matches for headers that are truncated in the source table
    # e.g. table has "Cash" but template expects "Cash and Cash Equivalents"
    substring_bonus = 0
    for th in template.headers:
        if th not in table_canonical:
            for tc in table_canonical:
                if th.lower() in tc.lower() or tc.lower() in th.lower():
                    substring_bonus += 0.5
                    break

    total_matched = matched + substring_bonus
    denom = max(len(table_headers), len(template.headers))
    return total_matched / denom if denom > 0 else 0.0


def match_template(
    headers: list[str],
) -> tuple[AccountingTemplate | None, float]:
    """
    Return the best-matching accounting template for *headers*.

    Args:
        headers: List of column / row header strings from the source table.

    Returns:
        (template, confidence) where confidence ∈ [0, 1].
        Returns (None, 0.0) if confidence < 0.4.
    """
    if not headers:
        return None, 0.0

    best_tpl: AccountingTemplate | None = None
    best_score = 0.0

    for tpl in TEMPLATES.values():
        score = _fuzzy_match_headers(headers, tpl)
        if score > best_score:
            best_score = score
            best_tpl = tpl

    if best_score < 0.4:
        return None, 0.0
    return best_tpl, round(best_score, 3)


def match_template_for_kg(
    kg_headers: list[str],
) -> tuple[AccountingTemplate | None, float, list[str]]:
    """
    Match a template and return both the matched template and the
    *normalised* header list so the KG builder can use canonical names.

    Returns (template, confidence, normalised_headers).
    """
    normalised = [normalize_header(h) for h in kg_headers]
    tpl, conf = match_template(normalised)
    return tpl, conf, normalised


def get_all_template_names() -> list[str]:
    """Return sorted list of all registered template names."""
    return sorted(TEMPLATES.keys())


def count_templates() -> int:
    """Return the total number of registered templates."""
    return len(TEMPLATES)
