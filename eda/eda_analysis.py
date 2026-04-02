"""
EDA Analysis for T²-RAGBench: Why General Retrieval Methods Fail on Financial Documents
"""

import os
import json
import re
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD ALL DATASETS
# ============================================================
print("=" * 80)
print("1. LOADING DATASETS")
print("=" * 80)

# Load ALL splits and concatenate for comprehensive analysis
finqa_all = load_dataset("G4KMU/t2-ragbench", "FinQA")
convfinqa_all = load_dataset("G4KMU/t2-ragbench", "ConvFinQA")
tatdqa_all = load_dataset("G4KMU/t2-ragbench", "TAT-DQA")

df_finqa = pd.concat([finqa_all[s].to_pandas() for s in finqa_all.keys()], ignore_index=True)
df_convfinqa = convfinqa_all["turn_0"].to_pandas()  # Only has turn_0 split
df_tatdqa = pd.concat([tatdqa_all[s].to_pandas() for s in tatdqa_all.keys()], ignore_index=True)

print(f"FinQA:     {len(df_finqa)} samples, columns: {list(df_finqa.columns)}")
print(f"ConvFinQA: {len(df_convfinqa)} samples, columns: {list(df_convfinqa.columns)}")
print(f"TAT-DQA:   {len(df_tatdqa)} samples, columns: {list(df_tatdqa.columns)}")

# Try loading VAQA
try:
    vaqa = load_dataset("G4KMU/va_qa", split="test")
    df_vaqa = vaqa.to_pandas()
    print(f"VAQA:      {len(df_vaqa)} samples, columns: {list(df_vaqa.columns)}")
    has_vaqa = True
except Exception as e:
    print(f"VAQA: Could not load - {e}")
    has_vaqa = False

# ============================================================
# 2. BASIC STATISTICS
# ============================================================
print("\n" + "=" * 80)
print("2. BASIC STATISTICS PER DATASET")
print("=" * 80)

def basic_stats(df, name, context_col="context", question_col="question", answer_col="program_answer"):
    """Compute basic stats for a dataset."""
    stats = {"name": name, "n_samples": len(df)}
    
    # Context stats
    if context_col in df.columns:
        ctx = df[context_col].dropna().astype(str)
        stats["n_unique_contexts"] = ctx.nunique()
        ctx_lengths = ctx.str.len()
        ctx_words = ctx.str.split().str.len()
        stats["ctx_char_mean"] = ctx_lengths.mean()
        stats["ctx_char_median"] = ctx_lengths.median()
        stats["ctx_char_std"] = ctx_lengths.std()
        stats["ctx_char_min"] = ctx_lengths.min()
        stats["ctx_char_max"] = ctx_lengths.max()
        stats["ctx_word_mean"] = ctx_words.mean()
        stats["ctx_word_median"] = ctx_words.median()
        stats["ctx_word_std"] = ctx_words.std()
    
    # Question stats
    if question_col in df.columns:
        q = df[question_col].dropna().astype(str)
        q_lengths = q.str.len()
        q_words = q.str.split().str.len()
        stats["q_char_mean"] = q_lengths.mean()
        stats["q_word_mean"] = q_words.mean()
        stats["q_word_median"] = q_words.median()
        stats["q_word_std"] = q_words.std()
        stats["q_word_min"] = q_words.min()
        stats["q_word_max"] = q_words.max()
    
    # Answer stats
    if answer_col in df.columns:
        a = df[answer_col].dropna().astype(str)
        stats["n_answers"] = len(a)
        a_words = a.str.split().str.len()
        stats["a_word_mean"] = a_words.mean()
        
        # Check how many answers are numeric
        def is_numeric(x):
            try:
                float(str(x).replace(",", "").replace("%", "").replace("$", "").strip())
                return True
            except:
                return False
        stats["pct_numeric_answers"] = df[answer_col].dropna().apply(is_numeric).mean() * 100
    
    return stats

stats_list = [
    basic_stats(df_finqa, "FinQA"),
    basic_stats(df_convfinqa, "ConvFinQA"),
    basic_stats(df_tatdqa, "TAT-DQA", answer_col="program_answer"),
]

for s in stats_list:
    print(f"\n--- {s['name']} ---")
    for k, v in s.items():
        if k == "name":
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

# ============================================================
# 3. CONTEXT STRUCTURE ANALYSIS (Tables, Text)
# ============================================================
print("\n" + "=" * 80)
print("3. CONTEXT STRUCTURE ANALYSIS")
print("=" * 80)

def analyze_context_structure(df, name, context_col="context"):
    """Analyze the structure of context: table presence, ratio of table vs text."""
    ctx = df[context_col].dropna().astype(str)
    
    # Detect markdown tables (lines with |)
    def count_table_lines(text):
        lines = text.split("\n")
        table_lines = [l for l in lines if "|" in l and l.strip().startswith("|")]
        return len(table_lines)
    
    def count_total_lines(text):
        return len(text.split("\n"))
    
    def has_table(text):
        return "|" in text and any(l.strip().startswith("|") for l in text.split("\n"))
    
    def count_tables(text):
        """Count number of separate table blocks."""
        in_table = False
        table_count = 0
        for line in text.split("\n"):
            if "|" in line and line.strip().startswith("|"):
                if not in_table:
                    table_count += 1
                    in_table = True
            else:
                in_table = False
        return table_count
    
    def count_table_rows(text):
        lines = text.split("\n")
        table_lines = [l for l in lines if "|" in l and l.strip().startswith("|")]
        # Subtract separator lines (|:---|:---|)
        data_lines = [l for l in table_lines if not re.match(r"^\|\s*[-:]+\s*\|", l.replace(" ", ""))]
        return len(data_lines)
    
    def count_table_cols(text):
        lines = text.split("\n")
        table_lines = [l for l in lines if "|" in l and l.strip().startswith("|")]
        if table_lines:
            # Count columns from first table line
            return len(table_lines[0].split("|")) - 2  # subtract leading/trailing empty
        return 0

    def count_numbers_in_text(text):
        """Count numeric values in text."""
        numbers = re.findall(r'\b\d[\d,.]*\b', text)
        return len(numbers)
    
    results = {
        "name": name,
        "pct_with_table": ctx.apply(has_table).mean() * 100,
        "avg_tables_per_doc": ctx.apply(count_tables).mean(),
        "avg_table_lines": ctx.apply(count_table_lines).mean(),
        "avg_total_lines": ctx.apply(count_total_lines).mean(),
        "avg_table_rows": ctx.apply(count_table_rows).mean(),
        "avg_table_cols": ctx.apply(count_table_cols).mean(),
        "avg_numbers_per_doc": ctx.apply(count_numbers_in_text).mean(),
        "table_line_ratio": ctx.apply(count_table_lines).mean() / max(ctx.apply(count_total_lines).mean(), 1),
    }
    
    # Text portion analysis
    def extract_text_portion(text):
        lines = text.split("\n")
        text_lines = [l for l in lines if not ("|" in l and l.strip().startswith("|"))]
        return " ".join(text_lines)
    
    text_portions = ctx.apply(extract_text_portion)
    results["avg_text_words"] = text_portions.str.split().str.len().mean()
    results["avg_text_chars"] = text_portions.str.len().mean()
    
    return results

for df_curr, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    res = analyze_context_structure(df_curr, name)
    print(f"\n--- {res['name']} ---")
    for k, v in res.items():
        if k == "name":
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

# ============================================================
# 4. QUESTION CHARACTERISTICS 
# ============================================================
print("\n" + "=" * 80)
print("4. QUESTION CHARACTERISTICS")
print("=" * 80)

def analyze_questions(df, name, question_col="question"):
    """Classify question types and characteristics."""
    q = df[question_col].dropna().astype(str)
    
    # Question type patterns
    patterns = {
        "what": r"\bwhat\b",
        "how_much": r"\bhow much\b",
        "how_many": r"\bhow many\b",
        "percentage/ratio": r"\bpercent|ratio|proportion|rate\b",
        "change/difference": r"\bchange|difference|increase|decrease|growth|decline\b",
        "compare": r"\bcompare|comparison|versus|vs\b|\bmore than\b|\bless than\b",
        "calculate": r"\bcalculate|compute|determine\b",
        "total/sum": r"\btotal|sum|aggregate|combined\b",
        "average/mean": r"\baverage|mean\b",
        "year_reference": r"\b20\d{2}\b|\b19\d{2}\b",
    }
    
    print(f"\n--- {name}: Question Type Distribution ---")
    for pname, pattern in patterns.items():
        pct = q.str.contains(pattern, case=False, regex=True).mean() * 100
        print(f"  {pname}: {pct:.1f}%")
    
    # Questions requiring multi-step reasoning
    multi_step_keywords = r"\band\b.*\band\b|\bthen\b|\bafter\b|\bfirst\b.*\bthen\b|\bratio of.*to\b"
    pct_multi = q.str.contains(multi_step_keywords, case=False, regex=True).mean() * 100
    print(f"  multi_step_indicator: {pct_multi:.1f}%")
    
    # Questions with specific company/entity references
    # Look for capitalized words that appear to be entity names
    def has_entity(text):
        # Match patterns like company names, ticker symbols
        return bool(re.search(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b', text)) or \
               bool(re.search(r'\b[A-Z]{2,5}\b', text))
    pct_entity = q.apply(has_entity).mean() * 100
    print(f"  has_entity_reference: {pct_entity:.1f}%")

for df_curr, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    analyze_questions(df_curr, name)

# ============================================================
# 5. VOCABULARY & DOMAIN-SPECIFIC TERMINOLOGY ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("5. VOCABULARY & DOMAIN-SPECIFIC TERMINOLOGY")
print("=" * 80)

# Financial domain specific terms
FINANCIAL_TERMS = [
    "revenue", "income", "earnings", "ebitda", "ebit", "profit", "loss",
    "assets", "liabilities", "equity", "debt", "depreciation", "amortization",
    "cash flow", "operating", "capital", "dividend", "share", "stock",
    "margin", "return", "roi", "roe", "roa", "eps", "pe ratio",
    "fiscal", "quarter", "annual", "year-over-year", "yoy",
    "balance sheet", "income statement", "cash flow statement",
    "gross", "net", "total", "weighted average", "diluted",
    "accrued", "deferred", "impairment", "goodwill", "intangible",
    "receivable", "payable", "inventory", "segment", "subsidiary",
    "consolidated", "provision", "contingent", "derivative",
    "fair value", "book value", "market cap", "enterprise value",
    "basis points", "bps", "hedge", "swap", "maturity",
]

ABBREVIATIONS = [
    "EBITDA", "EBIT", "EPS", "ROI", "ROE", "ROA", "P/E", "GAAP", "IFRS",
    "SEC", "CFO", "CEO", "IPO", "M&A", "FCF", "CAPEX", "OPEX",
    "YoY", "QoQ", "TTM", "LTM", "FY", "Q1", "Q2", "Q3", "Q4",
    "USD", "EUR", "GBP", "NYSE", "NASDAQ", "S&P", "B/S", "P&L",
    "DCF", "NPV", "IRR", "WACC", "D/E", "P/B", "PEG", "NAV",
    "SGA", "R&D", "PP&E", "COGS", "WIP", "AR", "AP",
]

def analyze_vocabulary(df, name, context_col="context", question_col="question"):
    """Analyze domain-specific vocabulary in contexts and questions."""
    ctx = df[context_col].dropna().astype(str)
    q = df[question_col].dropna().astype(str)
    
    # Financial term frequency in contexts
    print(f"\n--- {name}: Financial Terms in Contexts (top 20) ---")
    all_ctx_text = " ".join(ctx.values).lower()
    term_counts = {}
    for term in FINANCIAL_TERMS:
        count = len(re.findall(r'\b' + re.escape(term) + r'\b', all_ctx_text, re.IGNORECASE))
        if count > 0:
            term_counts[term] = count
    sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for term, count in sorted_terms:
        print(f"  {term}: {count}")
    
    # Abbreviation analysis in contexts
    print(f"\n--- {name}: Abbreviation Frequency in Contexts ---")
    abbr_counts = {}
    for abbr in ABBREVIATIONS:
        count = len(re.findall(r'\b' + re.escape(abbr) + r'\b', " ".join(ctx.values)))
        if count > 0:
            abbr_counts[abbr] = count
    sorted_abbr = sorted(abbr_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for abbr, count in sorted_abbr:
        print(f"  {abbr}: {count}")
    
    # Abbreviation density (per document)
    def count_abbreviations(text):
        count = 0
        for abbr in ABBREVIATIONS:
            count += len(re.findall(r'\b' + re.escape(abbr) + r'\b', text))
        return count
    
    abbr_per_doc = ctx.apply(count_abbreviations)
    print(f"\n  Avg abbreviations per document: {abbr_per_doc.mean():.2f}")
    print(f"  Max abbreviations per document: {abbr_per_doc.max()}")
    print(f"  % docs with abbreviations: {(abbr_per_doc > 0).mean() * 100:.1f}%")
    
    # Financial terms in questions
    print(f"\n--- {name}: Financial Terms in Questions (top 15) ---")
    all_q_text = " ".join(q.values).lower()
    q_term_counts = {}
    for term in FINANCIAL_TERMS:
        count = len(re.findall(r'\b' + re.escape(term) + r'\b', all_q_text, re.IGNORECASE))
        if count > 0:
            q_term_counts[term] = count
    sorted_q_terms = sorted(q_term_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for term, count in sorted_q_terms:
        print(f"  {term}: {count}")

for df_curr, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    analyze_vocabulary(df_curr, name)

# ============================================================
# 6. NUMERICAL DENSITY & COMPLEXITY
# ============================================================
print("\n" + "=" * 80)
print("6. NUMERICAL DENSITY & COMPLEXITY")
print("=" * 80)

def analyze_numerical_density(df, name, context_col="context", question_col="question"):
    """Analyze numerical density that makes retrieval hard."""
    ctx = df[context_col].dropna().astype(str)
    q = df[question_col].dropna().astype(str)
    
    def count_numbers(text):
        return len(re.findall(r'-?\$?\d[\d,.]*%?', text))
    
    def count_monetary_values(text):
        return len(re.findall(r'\$[\d,.]+|\d[\d,.]*\s*(?:million|billion|thousand|mn|bn|USD|EUR)', text, re.IGNORECASE))
    
    def count_percentages(text):
        return len(re.findall(r'\d[\d,.]*\s*%', text))
    
    def count_years(text):
        return len(re.findall(r'\b(?:19|20)\d{2}\b', text))
    
    ctx_nums = ctx.apply(count_numbers)
    ctx_monetary = ctx.apply(count_monetary_values)
    ctx_pct = ctx.apply(count_percentages)
    ctx_years = ctx.apply(count_years)
    ctx_words = ctx.str.split().str.len()
    
    q_nums = q.apply(count_numbers)
    
    print(f"\n--- {name}: Numerical Density ---")
    print(f"  Context:")
    print(f"    Avg numbers per doc: {ctx_nums.mean():.1f}")
    print(f"    Avg monetary values per doc: {ctx_monetary.mean():.1f}")
    print(f"    Avg percentages per doc: {ctx_pct.mean():.1f}")
    print(f"    Avg year references per doc: {ctx_years.mean():.1f}")
    print(f"    Number-to-word ratio: {(ctx_nums / ctx_words.clip(lower=1)).mean():.4f}")
    print(f"    Median numbers per doc: {ctx_nums.median():.0f}")
    print(f"    Max numbers per doc: {ctx_nums.max()}")
    print(f"  Questions:")
    print(f"    Avg numbers per question: {q_nums.mean():.2f}")
    print(f"    % questions with numbers: {(q_nums > 0).mean() * 100:.1f}%")

for df_curr, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    analyze_numerical_density(df_curr, name)

# ============================================================
# 7. LEXICAL OVERLAP ANALYSIS (Query-Document)
# ============================================================
print("\n" + "=" * 80)
print("7. LEXICAL OVERLAP ANALYSIS (Query vs Context)")
print("=" * 80)

def analyze_lexical_overlap(df, name, context_col="context", question_col="question"):
    """Measure lexical overlap between questions and their ground-truth contexts."""
    
    stopwords = set(["the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "could",
                     "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                     "on", "with", "at", "by", "from", "as", "into", "through", "during",
                     "before", "after", "above", "below", "between", "and", "but", "or",
                     "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
                     "every", "all", "any", "few", "more", "most", "other", "some", "such",
                     "than", "too", "very", "just", "about", "it", "its", "this", "that",
                     "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
                     "him", "his", "she", "her", "they", "them", "their", "what", "which",
                     "who", "whom", "how", "when", "where", "why", "if", "then", "else",
                     "there", "here", "up", "out", "down", "off", "over", "under", "again"])
    
    def tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(t for t in tokens if t not in stopwords and len(t) > 1)
    
    overlaps = []
    jaccard_scores = []
    q_coverage = []
    
    valid_pairs = df[[question_col, context_col]].dropna()
    
    for _, row in valid_pairs.iterrows():
        q_tokens = tokenize(str(row[question_col]))
        c_tokens = tokenize(str(row[context_col]))
        
        if len(q_tokens) == 0 or len(c_tokens) == 0:
            continue
        
        overlap = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        
        overlaps.append(overlap)
        jaccard_scores.append(overlap / union if union > 0 else 0)
        q_coverage.append(overlap / len(q_tokens) if len(q_tokens) > 0 else 0)
    
    overlaps = np.array(overlaps)
    jaccard_scores = np.array(jaccard_scores)
    q_coverage = np.array(q_coverage)
    
    print(f"\n--- {name}: Lexical Overlap ---")
    print(f"  Avg overlapping tokens: {overlaps.mean():.2f}")
    print(f"  Avg Jaccard similarity: {jaccard_scores.mean():.4f}")
    print(f"  Avg query coverage (% query tokens found in context): {q_coverage.mean():.4f}")
    print(f"  Median query coverage: {np.median(q_coverage):.4f}")
    print(f"  % queries with <30% coverage: {(q_coverage < 0.3).mean() * 100:.1f}%")
    print(f"  % queries with <50% coverage: {(q_coverage < 0.5).mean() * 100:.1f}%")

for df_curr, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    analyze_lexical_overlap(df_curr, name)

# ============================================================
# 8. HARD NEGATIVE ANALYSIS (Context Similarity)
# ============================================================
print("\n" + "=" * 80)
print("8. HARD NEGATIVE ANALYSIS - Context Similarity") 
print("=" * 80)

def analyze_hard_negatives(df, name, context_col="context", n_sample=500):
    """Analyze how similar different contexts are to each other (hard negatives)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get unique contexts
    unique_ctx = df[context_col].dropna().astype(str).unique()
    
    # Sample if too many
    if len(unique_ctx) > n_sample:
        np.random.seed(42)
        unique_ctx = np.random.choice(unique_ctx, n_sample, replace=False)
    
    print(f"\n--- {name}: Hard Negative Analysis (n={len(unique_ctx)} unique contexts) ---")
    
    # TF-IDF similarity
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(unique_ctx)
    
    # Compute pairwise similarities (only upper triangle)
    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity
    
    # Get upper triangle similarities
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    
    print(f"  TF-IDF Cosine Similarity Distribution:")
    print(f"    Mean: {upper_triangle.mean():.4f}")
    print(f"    Median: {np.median(upper_triangle):.4f}")
    print(f"    Std: {upper_triangle.std():.4f}")
    print(f"    P75: {np.percentile(upper_triangle, 75):.4f}")
    print(f"    P90: {np.percentile(upper_triangle, 90):.4f}")
    print(f"    P95: {np.percentile(upper_triangle, 95):.4f}")
    print(f"    P99: {np.percentile(upper_triangle, 99):.4f}")
    print(f"    Max: {upper_triangle.max():.4f}")
    
    # Hard negatives: pairs with similarity > 0.5
    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    for t in thresholds:
        pct = (upper_triangle > t).mean() * 100
        print(f"    % pairs with sim > {t}: {pct:.3f}%")
    
    # What makes contexts similar? Analyze high-similarity pairs
    high_sim_indices = np.argwhere(sim_matrix > 0.7)
    if len(high_sim_indices) > 0:
        print(f"\n  Example high-similarity pair (sim > 0.7):")
        idx = high_sim_indices[0]
        print(f"    Doc A (first 200 chars): {unique_ctx[idx[0]][:200]}")
        print(f"    Doc B (first 200 chars): {unique_ctx[idx[1]][:200]}")
        print(f"    Similarity: {sim_matrix[idx[0], idx[1]]:.4f}")

for df_curr, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    analyze_hard_negatives(df_curr, name)

# ============================================================
# 9. CONTEXT DUPLICATION & MULTI-QUESTION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("9. CONTEXT DUPLICATION & MULTI-QUESTION ANALYSIS")
print("=" * 80)

def analyze_context_sharing(df, name, context_col="context", question_col="question"):
    """How many questions share the same context?"""
    ctx_groups = df.groupby(context_col)[question_col].count()
    
    print(f"\n--- {name}: Context Sharing ---")
    print(f"  Total questions: {len(df)}")
    print(f"  Unique contexts: {len(ctx_groups)}")
    print(f"  Questions per context:")
    print(f"    Mean: {ctx_groups.mean():.2f}")
    print(f"    Median: {ctx_groups.median():.0f}")
    print(f"    Max: {ctx_groups.max()}")
    print(f"    % contexts with >1 question: {(ctx_groups > 1).mean() * 100:.1f}%")
    print(f"    % contexts with >3 questions: {(ctx_groups > 3).mean() * 100:.1f}%")
    print(f"    % contexts with >5 questions: {(ctx_groups > 5).mean() * 100:.1f}%")

for df_curr, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    analyze_context_sharing(df_curr, name)

# ============================================================
# 10. PROGRAM/REASONING COMPLEXITY ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("10. REASONING COMPLEXITY (FinQA Programs)")
print("=" * 80)

if "program_solution" in df_finqa.columns:
    programs = df_finqa["program_solution"].dropna().astype(str)
    
    # Count operations per program
    ops = ["add", "subtract", "multiply", "divide", "exp", "greater",
           "table_sum", "table_average", "table_max", "table_min"]
    
    def count_ops(prog):
        return sum(len(re.findall(r'\b' + op + r'\b', prog)) for op in ops)
    
    def get_op_types(prog):
        found = []
        for op in ops:
            if re.search(r'\b' + op + r'\b', prog):
                found.append(op)
        return found
    
    op_counts = programs.apply(count_ops)
    print(f"  Avg operations per program: {op_counts.mean():.2f}")
    print(f"  Max operations per program: {op_counts.max()}")
    print(f"  Distribution:")
    for n_ops in range(1, 7):
        pct = (op_counts == n_ops).mean() * 100
        print(f"    {n_ops} ops: {pct:.1f}%")
    pct_more = (op_counts > 6).mean() * 100
    print(f"    >6 ops: {pct_more:.1f}%")
    
    # Most common operation types
    all_ops = []
    for prog in programs:
        all_ops.extend(get_op_types(prog))
    op_counter = Counter(all_ops)
    print(f"\n  Operation frequency:")
    for op, count in op_counter.most_common():
        print(f"    {op}: {count} ({count/len(programs)*100:.1f}%)")

# ============================================================
# 11. SAMPLE EXAMPLES - Illustrate Retrieval Challenges
# ============================================================
print("\n" + "=" * 80)
print("11. SAMPLE EXAMPLES - Illustration of Retrieval Challenges")
print("=" * 80)

print("\n--- Example 1: FinQA question vs context ---")
example = df_finqa.iloc[0]
print(f"  Question: {example.get('question', 'N/A')}")
print(f"  Answer: {example.get('program_answer', 'N/A')}")
print(f"  Context (first 500 chars): {str(example.get('context', 'N/A'))[:500]}")

print("\n--- Example 2: Another FinQA ---")
example = df_finqa.iloc[100]
print(f"  Question: {example.get('question', 'N/A')}")
print(f"  Answer: {example.get('program_answer', 'N/A')}")
print(f"  Context (first 500 chars): {str(example.get('context', 'N/A'))[:500]}")

print("\n--- Example 3: TAT-DQA ---")
example = df_tatdqa.iloc[0]
print(f"  Question: {example.get('question', 'N/A')}")
print(f"  Answer: {example.get('program_answer', 'N/A')}")
print(f"  Context (first 500 chars): {str(example.get('context', 'N/A'))[:500]}")

# ============================================================
# 12. COMPANY/SECTOR DISTRIBUTION
# ============================================================
print("\n" + "=" * 80)
print("12. COMPANY & SECTOR DISTRIBUTION")
print("=" * 80)

for df_curr, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n--- {name} ---")
    if "company_sector" in df_curr.columns:
        sector_dist = df_curr["company_sector"].value_counts().head(10)
        print(f"  Top sectors:")
        for sector, count in sector_dist.items():
            print(f"    {sector}: {count} ({count/len(df_curr)*100:.1f}%)")
    if "company_name" in df_curr.columns:
        n_companies = df_curr["company_name"].nunique()
        print(f"  Unique companies: {n_companies}")

# ============================================================
# 13. SUMMARY STATISTICS TABLE
# ============================================================
print("\n" + "=" * 80)
print("13. SUMMARY - KEY FINDINGS FOR RETRIEVAL DIFFICULTY")
print("=" * 80)

print("""
KEY FACTORS WHY GENERAL RETRIEVAL FAILS ON FINANCIAL DOCUMENTS:

1. HIGH TABLE DENSITY: Financial documents are dominated by markdown tables
   with dense numerical data. General retrievers trained on natural text 
   struggle with structured tabular content.

2. EXTREME NUMERICAL DENSITY: Documents contain massive amounts of numbers
   (monetary values, percentages, dates, ratios). These numbers are critical 
   for answering questions but are poorly handled by text embeddings.

3. LOW LEXICAL OVERLAP: Questions are often reformulated to be context-independent,
   reducing direct keyword overlap with source documents. This hurts BM25-style 
   sparse retrieval.

4. HARD NEGATIVES: Financial documents from the same company/sector/period share
   very similar vocabulary and structure, creating many near-duplicate contexts that
   confuse dense retrievers.

5. DOMAIN-SPECIFIC TERMINOLOGY: Heavy use of financial abbreviations (EBITDA, EPS, 
   GAAP, etc.) and technical terms that may not be well-represented in general-purpose
   embedding models.

6. MULTI-STEP REASONING REQUIREMENTS: Questions often require extracting multiple 
   values from tables and performing calculations, but the retrieval query doesn't
   capture this computational intent.

7. CONTEXT SHARING: Multiple different questions may reference the same context but
   require different pieces of information, making single-vector representations 
   insufficient.
""")

print("\nEDA ANALYSIS COMPLETE!")
