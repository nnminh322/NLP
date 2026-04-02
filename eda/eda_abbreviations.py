"""
Deep EDA: Abbreviations & Domain-Specific Terminology in T²-RAGBench
- Phát hiện mọi viết tắt (uppercase patterns) trong các dataset  
- Đánh giá ảnh hưởng lên retrieval
"""
import re
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset

warnings.filterwarnings("ignore")

# Load datasets
print("Loading datasets...")
finqa_all = load_dataset("G4KMU/t2-ragbench", "FinQA")
convfinqa_all = load_dataset("G4KMU/t2-ragbench", "ConvFinQA")
tatdqa_all = load_dataset("G4KMU/t2-ragbench", "TAT-DQA")

df_finqa = pd.concat([finqa_all[s].to_pandas() for s in finqa_all.keys()], ignore_index=True)
df_convfinqa = convfinqa_all["turn_0"].to_pandas()
df_tatdqa = pd.concat([tatdqa_all[s].to_pandas() for s in tatdqa_all.keys()], ignore_index=True)

# ============================================================
# 1. AUTO-DETECT ALL ABBREVIATIONS (uppercase ≥2 chars)
# ============================================================
print("=" * 80)
print("1. AUTO-DETECTED ABBREVIATIONS IN CONTEXTS (Uppercase ≥2 chars)")
print("=" * 80)

def find_all_abbreviations(text):
    """Find all uppercase abbreviation-like patterns in text."""
    # Pattern 1: Pure uppercase (e.g., EBITDA, SEC, USA, EPS)
    pure_upper = re.findall(r'\b[A-Z]{2,}\b', text)
    
    # Pattern 2: Uppercase with digits (e.g., Q1, Q2, FY2019, S&P500)
    upper_digit = re.findall(r'\b[A-Z][A-Z0-9]{1,}\b', text)
    
    # Pattern 3: Abbreviations with special chars (e.g., P/E, R&D, D/E, P&L, M&A)
    special_abbr = re.findall(r'\b[A-Z][A-Z&/][A-Z]?\b', text)
    
    # Pattern 4: Dot-separated abbreviations (e.g., U.S., S.E.C.)
    dot_abbr = re.findall(r'\b(?:[A-Z]\.){2,}', text)
    
    # Pattern 5: Mixed acronyms like "10-K", "10-Q", "8-K" (SEC filings)
    sec_filings = re.findall(r'\b(?:10-[KQ]|8-K|20-F|6-K)\b', text)
    
    # Pattern 6: Abbreviations with hyphens (e.g., ROIC, year-over-year -> YoY)
    camel_abbr = re.findall(r'\b[A-Z][a-z]?[A-Z][a-z]?\b', text)
    
    all_abbr = pure_upper + upper_digit + special_abbr + dot_abbr + sec_filings
    return all_abbr

def find_financial_abbreviations(text):
    """More targeted: find known financial abbreviation patterns."""
    patterns = {
        # Accounting & Financial Metrics
        "EBITDA": r'\bEBITDA\b',
        "EBIT": r'\bEBIT\b(?!DA)',
        "EPS": r'\bEPS\b',
        "ROI": r'\bROI\b',
        "ROE": r'\bROE\b', 
        "ROA": r'\bROA\b',
        "ROIC": r'\bROIC\b',
        "GAAP": r'\bGAAP\b',
        "IFRS": r'\bIFRS\b',
        "P/E": r'P/E|P\/E',
        "P/B": r'P/B|P\/B',
        "D/E": r'D/E|D\/E',
        "NPV": r'\bNPV\b',
        "IRR": r'\bIRR\b',
        "WACC": r'\bWACC\b',
        "DCF": r'\bDCF\b',
        "NAV": r'\bNAV\b',
        "FCF": r'\bFCF\b',
        
        # Business & Corporate
        "CEO": r'\bCEO\b',
        "CFO": r'\bCFO\b',
        "COO": r'\bCOO\b',
        "CTO": r'\bCTO\b',
        "IPO": r'\bIPO\b',
        "M&A": r'M&A|M\&A',
        "R&D": r'R&D|R\&D',
        "LLC": r'\bLLC\b',
        "LLP": r'\bLLP\b',
        "Inc": r'\bInc\b\.',
        "Corp": r'\bCorp\b\.',
        "Ltd": r'\bLtd\b\.',
        
        # Regulatory & Filing
        "SEC": r'\bSEC\b',
        "FASB": r'\bFASB\b',
        "ASC": r'\bASC\b',
        "ASU": r'\bASU\b',
        "PCAOB": r'\bPCAOB\b',
        "SOX": r'\bSOX\b',
        "10-K": r'\b10-K\b',
        "10-Q": r'\b10-Q\b',
        "8-K": r'\b8-K\b',
        "20-F": r'\b20-F\b',
        
        # Financial Statements & Line Items
        "SGA/SG&A": r'SG&A|SGA|SG\&A',
        "COGS": r'\bCOGS\b',
        "CAPEX": r'\bCAPEX\b|capex',
        "OPEX": r'\bOPEX\b|opex',
        "PP&E": r'PP&E|PPE|PP\&E',
        "AR (Accts Receivable)": r'\bAR\b',
        "AP (Accts Payable)": r'\bAP\b',
        "NI (Net Income)": r'\bNI\b',
        "OCI": r'\bOCI\b',
        "AOCI": r'\bAOCI\b',
        
        # Market & Trading
        "NYSE": r'\bNYSE\b',
        "NASDAQ": r'\bNASDAQ\b',
        "S&P": r'S&P|S\&P',
        "ETF": r'\bETF\b',
        "CDO": r'\bCDO\b',
        "CDS": r'\bCDS\b',
        "OTC": r'\bOTC\b',
        "ABS": r'\bABS\b',
        "MBS": r'\bMBS\b',
        
        # Time Periods
        "YoY": r'\bYoY\b|year-over-year|year over year',
        "QoQ": r'\bQoQ\b|quarter-over-quarter',
        "TTM": r'\bTTM\b',
        "LTM": r'\bLTM\b',
        "FY": r'\bFY\b',
        "Q1": r'\bQ1\b',
        "Q2": r'\bQ2\b',
        "Q3": r'\bQ3\b',
        "Q4": r'\bQ4\b',
        "YTD": r'\bYTD\b',
        "MTD": r'\bMTD\b',
        
        # Currency & Units
        "USD": r'\bUSD\b',
        "EUR": r'\bEUR\b',
        "GBP": r'\bGBP\b',
        "JPY": r'\bJPY\b',
        "BPS/bps": r'\bbps\b|\bBPS\b|basis points',
        "mn/million": r'\bmn\b|\bMN\b',
        "bn/billion": r'\bbn\b|\bBN\b',
        
        # Tax & Legal
        "IRS": r'\bIRS\b',
        "NOL": r'\bNOL\b',
        "DTA": r'\bDTA\b',
        "DTL": r'\bDTL\b',
        "VIE": r'\bVIE\b',
        "SPE": r'\bSPE\b',
        
        # Country / Standard  
        "U.S.": r'U\.S\.',
        "UK": r'\bUK\b',
        "EU": r'\bEU\b',
    }
    return patterns

for df, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    ctx = df["context"].dropna().astype(str)
    q = df["question"].dropna().astype(str)
    
    # --- A) Auto-detect all uppercase abbreviations ---
    all_abbr = []
    for text in ctx.values:
        all_abbr.extend(find_all_abbreviations(text))
    
    abbr_counter = Counter(all_abbr)
    # Filter out common English words that happen to be uppercase
    noise_words = {"THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD",
                   "WILL", "CAN", "MAY", "ALL", "ANY", "OUR", "ITS", "HIS", "HER",
                   "NEW", "ONE", "TWO", "END", "USE", "SET", "PUT", "GET", "LET",
                   "RUN", "SAY", "SAW", "SEE", "OWN", "DID", "GOT", "OLD", "BIG",
                   "ADD", "TRY", "OFF", "OUT", "NOR", "YET", "FEW", "DUE", "PER",
                   "VIA", "TAX", "NET"}
    
    filtered = {k: v for k, v in abbr_counter.items() if k not in noise_words and v >= 10}
    sorted_abbr = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  [A] Auto-Detected Abbreviations in CONTEXTS (top 40):")
    print(f"  {'Abbreviation':<20} {'Count':>8}  {'Per-doc avg':>12}")
    print(f"  {'-'*44}")
    for abbr, count in sorted_abbr[:40]:
        per_doc = count / len(ctx)
        print(f"  {abbr:<20} {count:>8}  {per_doc:>12.3f}")
    
    total_unique_abbr = len(filtered)
    total_abbr_instances = sum(filtered.values())
    print(f"\n  Total unique abbreviations (≥10 occurrences): {total_unique_abbr}")
    print(f"  Total abbreviation instances: {total_abbr_instances}")
    print(f"  Avg abbreviation instances per doc: {total_abbr_instances / len(ctx):.2f}")
    
    # --- B) TARGETED financial abbreviation detection --- 
    fin_patterns = find_financial_abbreviations("")
    
    print(f"\n  [B] Financial Abbreviation Frequency in CONTEXTS:")
    print(f"  {'Category':<30} {'Count':>8}  {'% docs':>8}  {'Per-doc':>8}")
    print(f"  {'-'*58}")
    
    fin_abbr_results = {}
    for abbr_name, pattern in fin_patterns.items():
        matches_per_doc = ctx.str.count(pattern)
        total_count = matches_per_doc.sum()
        pct_docs = (matches_per_doc > 0).mean() * 100
        per_doc = matches_per_doc.mean()
        if total_count > 0:
            fin_abbr_results[abbr_name] = {
                "count": total_count, "pct_docs": pct_docs, "per_doc": per_doc
            }
    
    sorted_fin = sorted(fin_abbr_results.items(), key=lambda x: x[1]["count"], reverse=True)
    for abbr_name, stats in sorted_fin:
        print(f"  {abbr_name:<30} {stats['count']:>8}  {stats['pct_docs']:>7.1f}%  {stats['per_doc']:>8.3f}")
    
    # --- C) Abbreviations in QUESTIONS ---
    print(f"\n  [C] Financial Abbreviation Frequency in QUESTIONS:")
    q_abbr_results = {}
    for abbr_name, pattern in fin_patterns.items():
        matches_per_q = q.str.count(pattern)
        total_count = matches_per_q.sum()
        pct_q = (matches_per_q > 0).mean() * 100
        if total_count > 0:
            q_abbr_results[abbr_name] = {"count": total_count, "pct_q": pct_q}
    
    sorted_q = sorted(q_abbr_results.items(), key=lambda x: x[1]["count"], reverse=True)
    print(f"  {'Abbreviation':<30} {'Count':>8}  {'% questions':>12}")
    print(f"  {'-'*54}")
    for abbr_name, stats in sorted_q:
        print(f"  {abbr_name:<30} {stats['count']:>8}  {stats['pct_q']:>11.1f}%")

# ============================================================
# 2. ABBREVIATION DENSITY DISTRIBUTION
# ============================================================
print("\n" + "=" * 80)
print("2. ABBREVIATION DENSITY PER DOCUMENT")
print("=" * 80)

for df, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    ctx = df["context"].dropna().astype(str)
    
    # Count all financial abbreviations per doc
    fin_patterns = find_financial_abbreviations("")
    
    def count_all_fin_abbr(text):
        total = 0
        for pattern in fin_patterns.values():
            total += len(re.findall(pattern, text))
        return total
    
    abbr_counts = ctx.apply(count_all_fin_abbr)
    
    # Also count auto-detected uppercase
    def count_auto_abbr(text):
        return len(find_all_abbreviations(text))
    
    auto_counts = ctx.apply(count_auto_abbr)
    
    print(f"\n--- {name} ---")
    print(f"  Financial abbreviations per doc:")
    print(f"    Mean: {abbr_counts.mean():.2f}")
    print(f"    Median: {abbr_counts.median():.0f}")
    print(f"    Std: {abbr_counts.std():.2f}")
    print(f"    Max: {abbr_counts.max()}")
    print(f"    % docs with ≥1: {(abbr_counts >= 1).mean() * 100:.1f}%")
    print(f"    % docs with ≥5: {(abbr_counts >= 5).mean() * 100:.1f}%")
    print(f"    % docs with ≥10: {(abbr_counts >= 10).mean() * 100:.1f}%")
    print(f"    % docs with ≥20: {(abbr_counts >= 20).mean() * 100:.1f}%")
    
    print(f"  Auto-detected uppercase tokens per doc:")
    print(f"    Mean: {auto_counts.mean():.2f}")
    print(f"    Median: {auto_counts.median():.0f}")
    print(f"    Max: {auto_counts.max()}")
    
    # Distribution buckets
    print(f"  Distribution of financial abbreviation count:")
    for low, high in [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 999)]:
        pct = ((abbr_counts >= low) & (abbr_counts <= high)).mean() * 100
        label = f"{low}-{high}" if high < 999 else f"{low}+"
        print(f"    {label}: {pct:.1f}%")

# ============================================================
# 3. ABBREVIATION-QUERY MISMATCH: Query uses full form, context uses abbreviation (or vice versa)
# ============================================================
print("\n" + "=" * 80)
print("3. ABBREVIATION MISMATCH: Query vs Context Form")
print("=" * 80)

# Known abbreviation-fullform pairs
ABBR_FULLFORM_PAIRS = [
    ("EBITDA", r"earnings before interest,? taxes,? depreciation,? and amortization"),
    ("EBIT", r"earnings before interest and taxes"),
    ("EPS", r"earnings per share"),
    ("ROE", r"return on equity"),
    ("ROA", r"return on assets"),
    ("ROI", r"return on investment"),
    ("ROIC", r"return on invested capital"),
    ("GAAP", r"generally accepted accounting principles"),
    ("IFRS", r"international financial reporting standards"),
    ("SG&A", r"selling,? general and administrative"),
    ("R&D", r"research and development"),
    ("PP&E", r"property,? plant,? and equipment"),
    ("COGS", r"cost of goods sold"),
    ("CAPEX", r"capital expenditure"),
    ("OCI", r"other comprehensive income"),
    ("AOCI", r"accumulated other comprehensive income"),
    ("VIE", r"variable interest entit"),
    ("NOL", r"net operating loss"),
    ("FCF", r"free cash flow"),
    ("IPO", r"initial public offering"),
    ("M&A", r"mergers? and acquisitions?"),
    ("YoY", r"year[- ]over[- ]year"),
    ("BPS", r"basis points?"),
    ("NAV", r"net asset value"),
    ("DCF", r"discounted cash flow"),
    ("NPV", r"net present value"),
    ("WACC", r"weighted average cost of capital"),
    ("ASC", r"accounting standards codification"),
    ("ASU", r"accounting standards update"),
    ("SEC", r"securities and exchange commission"),
    ("FY", r"fiscal year"),
]

for df, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n--- {name} ---")
    
    ctx = df["context"].dropna().astype(str)
    q = df["question"].dropna().astype(str)
    
    print(f"  {'Abbr':<10} {'Ctx has abbr':>13} {'Ctx has full':>13} {'Q has abbr':>11} {'Q has full':>11}  Mismatch pattern")
    print(f"  {'-'*75}")
    
    mismatch_count = 0
    total_relevant = 0
    
    for abbr, fullform_pattern in ABBR_FULLFORM_PAIRS:
        # In contexts
        if "&" in abbr or "/" in abbr:
            abbr_pattern_ctx = re.escape(abbr)
        else:
            abbr_pattern_ctx = r'\b' + re.escape(abbr) + r'\b'
        
        ctx_has_abbr = ctx.str.contains(abbr_pattern_ctx, regex=True).mean() * 100
        ctx_has_full = ctx.str.contains(fullform_pattern, case=False, regex=True).mean() * 100
        q_has_abbr = q.str.contains(abbr_pattern_ctx, regex=True).mean() * 100
        q_has_full = q.str.contains(fullform_pattern, case=False, regex=True).mean() * 100
        
        if ctx_has_abbr > 0.1 or ctx_has_full > 0.1 or q_has_abbr > 0.1 or q_has_full > 0.1:
            # Detect mismatch pattern
            mismatch = ""
            if ctx_has_abbr > 1 and q_has_full > 0.1 and q_has_abbr < 0.1:
                mismatch = "⚠️ Q=full, Ctx=abbr"
                mismatch_count += 1
            elif ctx_has_full > 1 and q_has_abbr > 0.1 and q_has_full < 0.1:
                mismatch = "⚠️ Q=abbr, Ctx=full" 
                mismatch_count += 1
            elif ctx_has_abbr > 1 and ctx_has_full > 1:
                mismatch = "Mixed in ctx"
            
            total_relevant += 1
            print(f"  {abbr:<10} {ctx_has_abbr:>12.1f}% {ctx_has_full:>12.1f}% {q_has_abbr:>10.1f}% {q_has_full:>10.1f}%  {mismatch}")
    
    print(f"\n  Mismatch pairs (Q uses different form than Ctx): {mismatch_count}/{total_relevant}")

# ============================================================
# 4. IMPACT ON RETRIEVAL: queries with abbreviations vs without
# ============================================================
print("\n" + "=" * 80)
print("4. ABBREVIATION IMPACT ON TF-IDF RETRIEVAL")
print("=" * 80)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for df, name in [(df_finqa, "FinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n--- {name} ---")
    
    fin_patterns_dict = find_financial_abbreviations("")
    
    ctx = df["context"].dropna().astype(str)
    q = df["question"].dropna().astype(str)
    
    # Classify queries: has financial abbreviation or not
    def has_fin_abbr(text):
        for pattern in fin_patterns_dict.values():
            if re.search(pattern, text):
                return True
        return False
    
    q_has_abbr = q.apply(has_fin_abbr)
    
    print(f"  % queries with financial abbreviations: {q_has_abbr.mean() * 100:.1f}%")
    print(f"  # queries WITH abbr: {q_has_abbr.sum()}")
    print(f"  # queries WITHOUT abbr: {(~q_has_abbr).sum()}")
    
    # Sample and compute retrieval performance for each group
    unique_contexts = ctx.unique()
    
    np.random.seed(42)
    # Sample from both groups
    n_per_group = 150
    
    with_abbr_idx = np.where(q_has_abbr.values)[0]
    without_abbr_idx = np.where(~q_has_abbr.values)[0]
    
    if len(with_abbr_idx) >= n_per_group:
        sample_with = np.random.choice(with_abbr_idx, n_per_group, replace=False)
    else:
        sample_with = with_abbr_idx
    
    sample_without = np.random.choice(without_abbr_idx, n_per_group, replace=False)
    
    all_sample_idx = np.concatenate([sample_with, sample_without])
    sampled_q = q.values[all_sample_idx]
    sampled_ctx_correct = ctx.values[all_sample_idx]
    
    all_texts = list(unique_contexts) + list(sampled_q)
    n_ctx = len(unique_contexts)
    
    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english")
    tfidf_all = vectorizer.fit_transform(all_texts)
    ctx_vectors = tfidf_all[:n_ctx]
    q_vectors = tfidf_all[n_ctx:]
    
    sims = cosine_similarity(q_vectors, ctx_vectors)
    ctx_to_idx = {c: i for i, c in enumerate(unique_contexts)}
    
    def compute_metrics(indices_in_sample):
        ranks = []
        for i in indices_in_sample:
            correct_idx = ctx_to_idx.get(sampled_ctx_correct[i])
            if correct_idx is None:
                continue
            sorted_ids = np.argsort(-sims[i])
            rank = np.where(sorted_ids == correct_idx)[0][0] + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        if len(ranks) == 0:
            return {}
        return {
            "n": len(ranks),
            "R@1": (ranks <= 1).mean() * 100,
            "R@3": (ranks <= 3).mean() * 100,
            "R@5": (ranks <= 5).mean() * 100,
            "MRR": np.mean(1.0 / ranks),
            "Mean rank": ranks.mean(),
        }
    
    n_with = len(sample_with)
    metrics_with = compute_metrics(range(0, n_with))
    metrics_without = compute_metrics(range(n_with, len(all_sample_idx)))
    
    print(f"\n  {'Metric':<15} {'With Abbr':>12} {'Without Abbr':>14} {'Delta':>10}")
    print(f"  {'-'*55}")
    for key in ["n", "R@1", "R@3", "R@5", "MRR", "Mean rank"]:
        v_with = metrics_with.get(key, 0)
        v_without = metrics_without.get(key, 0)
        delta = v_with - v_without
        if key == "n":
            print(f"  {key:<15} {v_with:>12.0f} {v_without:>14.0f}")
        else:
            print(f"  {key:<15} {v_with:>12.1f} {v_without:>14.1f} {delta:>+10.1f}")

# ============================================================
# 5. ABBREVIATION AMBIGUITY: Same abbreviation, different meanings
# ============================================================
print("\n" + "=" * 80)
print("5. ABBREVIATION AMBIGUITY EXAMPLES")
print("=" * 80)

ambiguous_abbr = {
    "AR": ["Accounts Receivable", "Annual Report", "Augmented Reality"],
    "AP": ["Accounts Payable", "Associated Press"],
    "NI": ["Net Income", "Northern Ireland"],
    "ABS": ["Asset-Backed Securities", "Absolute"],
    "CDS": ["Credit Default Swap", "Compact Disc"],
    "PP": ["Percentage Points", "PowerPoint"],
    "OCI": ["Other Comprehensive Income", "Oracle Cloud Infrastructure"],
    "LIBOR": ["London Interbank Offered Rate"],
    "SOFR": ["Secured Overnight Financing Rate"],
}

for df, name in [(df_finqa, "FinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n--- {name} ---")
    ctx = df["context"].dropna().astype(str)
    for abbr, meanings in ambiguous_abbr.items():
        count = ctx.str.contains(r'\b' + re.escape(abbr) + r'\b', regex=True).sum()
        if count > 0:
            print(f"  {abbr} ({', '.join(meanings)}): appears in {count} docs ({count/len(ctx)*100:.1f}%)")

# ============================================================  
# 6. SAMPLE EXAMPLES: Abbreviation-heavy documents
# ============================================================
print("\n" + "=" * 80)
print("6. SAMPLE: ABBREVIATION-HEAVY DOCUMENTS")
print("=" * 80)

for df, name in [(df_finqa, "FinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n--- {name}: Top 3 abbreviation-heavy docs ---")
    
    ctx = df["context"].dropna().astype(str)
    fin_patterns_dict = find_financial_abbreviations("")
    
    def count_all_fin_abbr(text):
        total = 0
        found = []
        for aname, pattern in fin_patterns_dict.items():
            c = len(re.findall(pattern, text))
            if c > 0:
                found.append(f"{aname}({c})")
                total += c
        return total, found
    
    doc_abbr = [(i, *count_all_fin_abbr(t)) for i, t in enumerate(ctx.values)]
    doc_abbr.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (idx, count, found) in enumerate(doc_abbr[:3]):
        print(f"\n  Doc #{rank+1} (index={idx}): {count} financial abbreviations")
        print(f"  Found: {', '.join(found[:20])}")
        print(f"  Question: {df['question'].iloc[idx][:200]}")
        print(f"  Context (first 300 chars): {ctx.iloc[idx][:300]}")

print("\n" + "=" * 80)
print("ABBREVIATION ANALYSIS COMPLETE!")
print("=" * 80)
