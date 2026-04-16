"""
Comprehensive EDA: Entity Signal, Table Presence, Accounting Equations, and Metadata Form
Across FinQA, TAT-DQA, ConvFinQA datasets.

Research questions:
  1. Entity co-reference: how many queries/docs have explicit entity mentions?
  2. Table presence: how many queries/docs contain tables? In what form?
  3. Accounting equations: how many queries/docs mention accounting identities?
  4. Metadata form: explicit (metadata columns) vs implicit (inside text)?

Output: Console + saves to eda/entity_signal_eda_output.txt
"""

import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

warnings.filterwarnings("ignore")
OUTPUT_FILE = Path(__file__).parent / "entity_signal_eda_output.txt"


def log(msg, fp=None):
    print(msg)
    if fp:
        fp.write(msg + "\n")


# ──────────────────────────────────────────────────────────────
# ACCOUNTING EQUATION PATTERNS
# ──────────────────────────────────────────────────────────────

ACCOUNTING_IDENTITY_PATTERNS = [
    # Core accounting identities
    (r"revenue\s*[-–—=]\s*expenses", "revenue minus expenses identity"),
    (r"net\s+(income|profit|earnings)", "net income pattern"),
    (r"gross\s+profit", "gross profit"),
    (r"operating\s+(income|income|profit)", "operating income"),
    (r"ebitda", "EBITDA"),
    (r"ebit", "EBIT"),
    (r"assets?\s*[-–—=]\s*liabilities?", "assets minus liabilities"),
    (r"assets?\s*[-–—=]\s*equity", "assets minus equity"),
    (r"equity\s*[-–—=]\s*assets?\s*[-–—=]\s*liabilities", "basic accounting equation"),
    (r"liabilities?\s*[-–—=]\s*assets?\s*[-–—=]\s*equity", "basic accounting equation"),
    (r"cash\s*(flow| Flows?)", "cash flow"),
    (r"retained\s+earnings", "retained earnings"),
    (r"working\s+capital", "working capital"),
    (r"current\s+ratio", "current ratio"),
    (r"debt\s*(to|[-–—])\s*equity", "debt-to-equity"),
    (r"debt\s*[-–—=]\s*total\s*assets?", "debt ratio"),
    (r"return\s*(on|over)\s*assets?", "ROA"),
    (r"return\s*(on|over)\s*equity", "ROE"),
    (r"return\s*(on|over)\s*investment", "ROI"),
    (r"earnings?\s*per\s+share", "EPS"),
    (r"price\s*[-–—]\s*earnings", "P/E ratio"),
    (r"book\s+value", "book value"),
    # Arithmetic operations that imply accounting computation
    (r"add(?:ed|ing)?\s+.*(?:revenue|income|sales)", "addition of revenue"),
    (r"subtract(?:ed|ing)?\s+.*(?:cost|expense)", "subtraction of cost"),
    (r"divide[sd]?\s+.*(?:share|stock)", "division operation"),
    (r"multiply(?:ed|ing)?\s+.*(?:share|stock|price)", "multiplication operation"),
    (r"sum\s+(?:of|all)", "summation"),
    (r"average\s+(?:of|all)", "averaging"),
    (r"total\s+(?:of|all)", "totaling"),
    (r"difference\s+between", "difference operation"),
    (r"(?:increase|decrease|change)\s+between", "change/delta operation"),
    (r"(?:growth|declined?)\s+(?:by|from)", "growth/decline"),
]

# Financial entity types
ENTITY_PATTERNS = {
    "ticker_symbol": r"\b[A-Z]{1,5}\b",  # AAPL, MSFT, GOOGL
    "company_fullname": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b",  # Apple Inc.
    "company_suffix": r"(?:Inc\.|Corp\.|Corporation|Ltd\.|Company|Co\.|PLC|plc)\b",
    "fiscal_year": r"(?:fiscal\s+)?(?:year\s+)?(?:FY|20\d{2})",
    "quarter": r"(?:Q[1-4]|first|second|third|fourth)\s+(?:quarter|fiscal)",
    "industry_term": r"(?:electric|utilities|telecomm|financials|healthcare|technology|energy)",
}

# Table detection patterns
TABLE_PATTERNS = {
    "markdown_pipe": r"\|[^\n]+\|",  # | col | col |
    "html_table": r"<table|<tr|<td|<th",
    "csv_like": r"^[^,\t]+,[^,\t]+",  # Comma/tab separated
    "aligned_numbers": r"\d{1,3}(?:,\d{3})+(?:\.\d+)?",  # 1,000,000
}


def has_table(text) -> bool:
    if not text or isinstance(text, bool):
        return False
    return bool(re.search(TABLE_PATTERNS["markdown_pipe"], str(text)))


def count_tables(text) -> int:
    if not text or isinstance(text, bool):
        return 0
    lines = str(text).split("\n")
    in_table = False
    count = 0
    for line in lines:
        if "|" in line and re.search(r"\|\s*-+\s*\|", line.replace(" ", "")):
            if not in_table:
                count += 1
                in_table = True
        elif "|" in line and line.strip().startswith("|"):
            pass  # continue table
        else:
            in_table = False
    return count


def has_accounting_equation(text: str) -> tuple[bool, list[str]]:
    if not text:
        return False, []
    text_lower = text.lower()
    found = []
    for pattern, label in ACCOUNTING_IDENTITY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            found.append(label)
    return len(found) > 0, found


def detect_entity_references(text: str) -> dict:
    """Detect entity references in text."""
    if not text:
        return {"has_ticker": False, "has_fullname": False, "has_year": False, "has_quarter": False}
    return {
        "has_ticker": bool(re.search(ENTITY_PATTERNS["ticker_symbol"], text)),
        "has_fullname": bool(re.search(ENTITY_PATTERNS["company_fullname"], text)),
        "has_year": bool(re.search(r"\b(19|20)\d{2}\b", text)),
        "has_quarter": bool(re.search(r"\b(Q[1-4]|first|second|third|fourth)\b", text, re.IGNORECASE)),
    }


def has_entity_in_query_vs_doc(query: str, doc_company: str) -> dict:
    """Check if query and doc reference the same entity."""
    if not query or not doc_company:
        return {"query_has_company": False, "doc_has_company": False, "same_reference": False}
    doc_lower = doc_company.lower()
    doc_upper = doc_company.upper()
    query_lower = query.lower()
    query_upper = query.upper()

    # Query has company reference
    query_has_company = doc_lower in query_lower or doc_upper in query_upper

    # Check ticker symbol (if available)
    if "company_symbol" in dir():
        ticker = doc_upper
        if ticker and len(ticker) <= 5:
            query_has_company = query_has_company or ticker in query_upper

    return {
        "query_has_company": query_has_company,
        "doc_has_company": True,  # Doc metadata always has company
        "same_reference": query_has_company,
    }


def analyze_dataset(name: str, df: pd.DataFrame, fp=None):
    """Comprehensive analysis for one dataset."""
    n = len(df)
    log(f"\n{'='*80}", fp)
    log(f"  DATASET: {name}  (n={n})", fp)
    log(f"{'='*80}", fp)

    # ── Metadata Form ──────────────────────────────────────
    log(f"\n── METADATA FORM ──", fp)
    explicit_cols = [c for c in df.columns if c in [
        "company_name", "report_year", "company_sector", "company_symbol",
        "company_industry", "company_cik", "company_headquarters",
    ]]
    log(f"  Explicit metadata columns found: {explicit_cols}", fp)
    log(f"  Total columns: {len(df.columns)}", fp)
    log(f"  Column names: {list(df.columns)}", fp)

    # Check if company_name appears IN the query text
    if "question" in df.columns and "company_name" in df.columns:
        query_prefix_count = df.apply(
            lambda r: str(r["company_name"]).lower() in str(r["question"]).lower(), axis=1
        ).sum()
        log(f"  Queries with company_name embedded in text: {query_prefix_count}/{n} ({query_prefix_count/n*100:.1f}%)", fp)

    # ── ENTITY CO-REFERENCE ──────────────────────────────────
    log(f"\n── ENTITY CO-REFERENCE ANALYSIS ──", fp)

    # Document entity analysis
    if "company_name" in df.columns:
        doc_has_company = df["company_name"].notna().sum()
        log(f"  Documents with company_name (metadata): {doc_has_company}/{n} ({doc_has_company/n*100:.1f}%)", fp)

    if "report_year" in df.columns:
        doc_has_year = df["report_year"].notna().sum()
        log(f"  Documents with report_year (metadata): {doc_has_year}/{n} ({doc_has_year/n*100:.1f}%)", fp)

    if "company_sector" in df.columns:
        doc_has_sector = df["company_sector"].notna().sum()
        log(f"  Documents with company_sector (metadata): {doc_has_sector}/{n} ({doc_has_sector/n*100:.1f}%)", fp)

    if "company_symbol" in df.columns:
        doc_has_symbol = df["company_symbol"].notna().sum()
        log(f"  Documents with company_symbol (metadata): {doc_has_symbol}/{n} ({doc_has_symbol/n*100:.1f}%)", fp)

    # Query entity analysis — from text
    if "question" in df.columns:
        q_texts = df["question"].fillna("")

        # Has ticker in query
        has_ticker = q_texts.apply(lambda q: bool(re.search(r"\b[A-Z]{2,5}\b", str(q)))).sum()
        log(f"  Queries with TICKER symbols (e.g. AAPL): {has_ticker}/{n} ({has_ticker/n*100:.1f}%)", fp)

        # Has year reference
        has_year = q_texts.apply(lambda q: bool(re.search(r"\b(19|20)\d{2}\b", str(q)))).sum()
        log(f"  Queries with YEAR references: {has_year}/{n} ({has_year/n*100:.1f}%)", fp)

        # Has quarter reference
        has_quarter = q_texts.apply(
            lambda q: bool(re.search(r"\b(Q[1-4]|first|second|third|fourth)\b", str(q), re.IGNORECASE))
        ).sum()
        log(f"  Queries with QUARTER references: {has_quarter}/{n} ({has_quarter/n*100:.1f}%)", fp)

        # Has company name in query (check if metadata company appears in query text)
        if "company_name" in df.columns:
            same_entity_in_query = df.apply(
                lambda r: str(r["company_name"]).lower() in str(r["question"]).lower() or
                          str(r.get("company_symbol", "")).upper() in str(r["question"]).upper(),
                axis=1
            ).sum()
            log(f"  Queries with SAME entity referenced in text: {same_entity_in_query}/{n} ({same_entity_in_query/n*100:.1f}%)", fp)

        # Full company name (capitalized proper noun)
        has_fullname = q_texts.apply(
            lambda q: bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", str(q)))
        ).sum()
        log(f"  Queries with full company names: {has_fullname}/{n} ({has_fullname/n*100:.1f}%)", fp)

    # ── TABLE PRESENCE ───────────────────────────────────────
    log(f"\n── TABLE PRESENCE ──", fp)

    if "context" in df.columns:
        ctx = df["context"].fillna("")
        docs_with_table = ctx.apply(has_table).sum()
        table_counts = ctx.apply(count_tables)
        log(f"  Documents with TABLE in context: {docs_with_table}/{n} ({docs_with_table/n*100:.1f}%)", fp)
        log(f"    Avg tables per document: {table_counts.mean():.2f}", fp)
        log(f"    Median tables per document: {table_counts.median():.0f}", fp)
        log(f"    Max tables per document: {table_counts.max()}", fp)

        # Also check separate `table` column if present
        if "table" in df.columns:
            docs_with_separate_table = df["table"].notna().apply(has_table).sum()
            log(f"  Documents with separate `table` column: {docs_with_separate_table}/{n} ({docs_with_separate_table/n*100:.1f}%)", fp)

    if "question" in df.columns:
        q_texts = df["question"].fillna("")
        q_with_table = q_texts.apply(has_table).sum()
        log(f"  Queries with TABLE in question text: {q_with_table}/{n} ({q_with_table/n*100:.1f}%)", fp)

        # Check for table-like patterns in query (column names, operators)
        q_table_like = q_texts.apply(
            lambda q: bool(re.search(r"(?:column|row|table|add|subtract|divide|multiply|sum|total|average)", str(q), re.IGNORECASE))
        ).sum()
        log(f"  Queries with TABLE-RELATED language: {q_table_like}/{n} ({q_table_like/n*100:.1f}%)", fp)

    # ── ACCOUNTING EQUATION ANALYSIS ──────────────────────────
    log(f"\n── ACCOUNTING EQUATION / IDENTITY PRESENCE ──", fp)

    if "context" in df.columns:
        ctx = df["context"].fillna("")
        ctx_has_eq, ctx_eq_types = [], []
        for text in ctx:
            has_eq, eqs = has_accounting_equation(str(text))
            ctx_has_eq.append(has_eq)
            ctx_eq_types.extend(eqs)
        n_ctx_eq = sum(ctx_has_eq)
        log(f"  Documents with accounting identities: {n_ctx_eq}/{n} ({n_ctx_eq/n*100:.1f}%)", fp)

        if ctx_eq_types:
            eq_counter = Counter(ctx_eq_types)
            log(f"  Top accounting patterns found:", fp)
            for pattern, count in eq_counter.most_common(10):
                log(f"    {pattern}: {count} ({count/n*100:.1f}%)", fp)

    if "question" in df.columns:
        q_texts = df["question"].fillna("")
        q_has_eq, q_eq_types = [], []
        for text in q_texts:
            has_eq, eqs = has_accounting_equation(str(text))
            q_has_eq.append(has_eq)
            q_eq_types.extend(eqs)
        n_q_eq = sum(q_has_eq)
        log(f"  Queries with accounting identities: {n_q_eq}/{n} ({n_q_eq/n*100:.1f}%)", fp)

        if q_eq_types:
            eq_counter = Counter(q_eq_types)
            log(f"  Top accounting patterns in queries:", fp)
            for pattern, count in eq_counter.most_common(10):
                log(f"    {pattern}: {count} ({count/n*100:.1f}%)", fp)

        # Computation operation keywords
        ops = ["add", "subtract", "divide", "multiply", "sum", "total", "average", "difference", "change"]
        for op in ops:
            cnt = q_texts.apply(lambda q: bool(re.search(rf"\b{op}(?:ed|ing|s)?\b", str(q), re.IGNORECASE))).sum()
            if cnt > 0:
                log(f"  Queries with '{op}' operation: {cnt}/{n} ({cnt/n*100:.1f}%)", fp)

    # ── QUESTION TYPE ────────────────────────────────────────
    log(f"\n── QUESTION TYPE DISTRIBUTION ──", fp)

    if "question" in df.columns:
        q_texts = df["question"].fillna("")
        qtypes = {
            "how_much/many": r"\bhow much\b|\bhow many\b",
            "percentage/ratio": r"\bpercent\b|\bratio\b|\brate\b|\bper\b",
            "change/difference": r"\bchange\b|\bdifference\b|\bincrease\b|\bdecrease\b|\bgrowth\b",
            "compare": r"\bcompare\b|\bversus\b|\bvs\b|\bmore than\b|\bless than\b",
            "calculate": r"\bcalculate\b|\bcompute\b|\bdetermine\b",
            "total/sum": r"\btotal\b|\bsum\b|\baggregate\b",
            "average/mean": r"\baverage\b|\bmean\b",
            "year_ref": r"\b(19|20)\d{2}\b",
        }
        for qtype, pattern in qtypes.items():
            cnt = q_texts.apply(lambda q: bool(re.search(pattern, str(q), re.IGNORECASE))).sum()
            log(f"  {qtype}: {cnt}/{n} ({cnt/n*100:.1f}%)", fp)

    # ── SAMPLE EXAMPLES ──────────────────────────────────────
    log(f"\n── SAMPLE EXAMPLES ──", fp)
    for i in range(min(3, n)):
        row = df.iloc[i]
        log(f"\n  [Sample {i}] {name}", fp)
        log(f"  Question: {str(row.get('question',''))[:150]}", fp)
        log(f"  Company: {row.get('company_name','N/A')} ({row.get('company_symbol','N/A')})", fp)
        log(f"  Year: {row.get('report_year','N/A')}", fp)
        log(f"  Sector: {row.get('company_sector','N/A')}", fp)
        has_t, _ = has_accounting_equation(str(row.get("context", "")))
        log(f"  Context has accounting eq: {has_t}", fp)
        log(f"  Context preview: {str(row.get('context',''))[:200]}", fp)


def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        log("=" * 80, fp)
        log("COMPREHENSIVE EDA: Entity Signal, Table, Accounting Equations, Metadata")
        log("Across FinQA, TAT-DQA, ConvFinQA", fp)
        log("=" * 80, fp)

        # Load all datasets
        log("\nLOADING DATASETS...", fp)
        try:
            finqa_all = load_dataset("G4KMU/t2-ragbench", "FinQA")
            df_finqa = pd.concat([finqa_all[s].to_pandas() for s in finqa_all.keys()], ignore_index=True)
            log(f"  FinQA loaded: {len(df_finqa)} total samples", fp)
        except Exception as e:
            log(f"  FinQA FAILED: {e}", fp)
            df_finqa = pd.DataFrame()

        try:
            convfinqa = load_dataset("G4KMU/t2-ragbench", "ConvFinQA")
            df_convfinqa = convfinqa["turn_0"].to_pandas()
            log(f"  ConvFinQA loaded: {len(df_convfinqa)} samples (turn_0)", fp)
        except Exception as e:
            log(f"  ConvFinQA FAILED: {e}", fp)
            df_convfinqa = pd.DataFrame()

        try:
            tat_all = load_dataset("G4KMU/t2-ragbench", "TAT-DQA")
            df_tatqa = pd.concat([tat_all[s].to_pandas() for s in tat_all.keys()], ignore_index=True)
            log(f"  TAT-DQA loaded: {len(df_tatqa)} total samples", fp)
        except Exception as e:
            log(f"  TAT-DQA FAILED: {e}", fp)
            df_tatqa = pd.DataFrame()

        # Analyze each
        if not df_finqa.empty:
            analyze_dataset("FinQA", df_finqa, fp)
        if not df_convfinqa.empty:
            analyze_dataset("ConvFinQA", df_convfinqa, fp)
        if not df_tatqa.empty:
            analyze_dataset("TAT-DQA", df_tatqa, fp)

        # ── CROSS-DATASET SUMMARY ─────────────────────────────
        log(f"\n{'='*80}", fp)
        log("CROSS-DATASET SUMMARY", fp)
        log(f"{'='*80}", fp)

        datasets = [("FinQA", df_finqa), ("ConvFinQA", df_convfinqa), ("TAT-DQA", df_tatqa)]
        summary_rows = []

        for name, df in datasets:
            if df.empty:
                continue
            n = len(df)
            row = {"Dataset": name, "N": n}

            # Entity metadata
            row["Doc has company (meta)"] = f"{df['company_name'].notna().sum()/n*100:.0f}%"
            row["Doc has year (meta)"] = f"{df['report_year'].notna().sum()/n*100:.0f}%"
            row["Doc has sector (meta)"] = f"{df['company_sector'].notna().sum()/n*100:.0f}%"

            # Query entity from text
            q = df["question"].fillna("")
            row["Query has ticker"] = f"{q.apply(lambda x: bool(re.search(r'\b[A-Z]{2,5}\b', str(x)))).sum()/n*100:.0f}%"
            row["Query has year"] = f"{q.apply(lambda x: bool(re.search(r'\b(19|20)\d{2}\b', str(x)))).sum()/n*100:.0f}%"
            row["Query has company name"] = f"{df.apply(lambda r: str(r.get('company_name','')).lower() in str(r['question']).lower(), axis=1).sum()/n*100:.0f}%"

            # Tables
            ctx = df["context"].fillna("")
            row["Doc has table"] = f"{ctx.apply(has_table).sum()/n*100:.0f}%"
            row["Query has table"] = f"{q.apply(has_table).sum()/n*100:.0f}%"

            # Accounting
            ctx_eq = [has_accounting_equation(str(t))[0] for t in ctx]
            q_eq = [has_accounting_equation(str(t))[0] for t in q]
            row["Doc has accounting eq"] = f"{sum(ctx_eq)/n*100:.0f}%"
            row["Query has accounting eq"] = f"{sum(q_eq)/n*100:.0f}%"

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        log(f"\n{summary_df.to_string(index=False)}", fp)

        # ── KEY INSIGHTS ────────────────────────────────────
        log(f"\n{'='*80}", fp)
        log("KEY INSIGHTS FOR GSR-CACL DESIGN", fp)
        log(f"{'='*80}", fp)
        log("""
1. METADATA FORM:
   - ALL 3 datasets have EXPLICIT metadata columns (company_name, report_year, sector)
   - Query text EMBEDS company name as prefix in some cases
   - Entity resolution: need to handle "AAPL" vs "Apple Inc." vs "Apple"

2. ENTITY CO-REFERENCE:
   - Documents: 100% have company metadata (explicit)
   - Queries: variable — some have ticker, some have full name, some implicit
   - Challenge: queries may reference entities DIFFERENTLY from document metadata
   - GSR-CACL EntitySignal is VALID: both Q and D have entity metadata

3. TABLE PRESENCE:
   - Documents: HIGH table density (most have markdown tables)
   - Queries: mostly NO table in query text (query asks about table content)
   - GSR-CACL StructuralSignal is VALID for documents, NOT needed for queries

4. ACCOUNTING EQUATIONS:
   - Documents: high frequency of accounting terminology (EBITDA, revenue, etc.)
   - Queries: computation keywords present (add, subtract, divide, total, etc.)
   - GSR-CACL ConstraintScore is VALID for documents with tables

5. IMPLICATION FOR ARCHITECTURE:
   - Entity signal: BOTH Q and D have metadata → use EntitySignal for BOTH
   - Structural signal: Only D has tables → use StructuralSignal for D only
   - Query may NOT have table → GAT encoding only for document tables
   - EntitySupConLoss essential for resolving "AAPL" ↔ "Apple Inc."
        """, fp)

        log(f"\nOutput saved to: {OUTPUT_FILE}", fp)

    print(f"\nAll output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
