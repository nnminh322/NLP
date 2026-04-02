"""
Supplementary EDA: Cross-query similarity, BM25 failure modes, and embedding gaps
"""
import re
import warnings
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Load datasets
finqa_all = load_dataset("G4KMU/t2-ragbench", "FinQA")
tatdqa_all = load_dataset("G4KMU/t2-ragbench", "TAT-DQA")
convfinqa_all = load_dataset("G4KMU/t2-ragbench", "ConvFinQA")

df_finqa = pd.concat([finqa_all[s].to_pandas() for s in finqa_all.keys()], ignore_index=True)
df_convfinqa = convfinqa_all["turn_0"].to_pandas()
df_tatdqa = pd.concat([tatdqa_all[s].to_pandas() for s in tatdqa_all.keys()], ignore_index=True)

# ============================================================
# A. BM25 FAILURE MODE: Query matches WRONG context better
# ============================================================
print("=" * 80)
print("A. BM25-STYLE RETRIEVAL FAILURE ANALYSIS")
print("=" * 80)

for df, name in [(df_finqa, "FinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n--- {name} ---")
    
    # Get unique contexts
    unique_contexts = df["context"].dropna().unique()
    questions = df["question"].dropna().values
    correct_ctx = df["context"].dropna().values
    
    # Sample for speed
    np.random.seed(42)
    n_sample = min(300, len(questions))
    sample_idx = np.random.choice(len(questions), n_sample, replace=False)
    
    sampled_q = questions[sample_idx]
    sampled_correct_ctx = correct_ctx[sample_idx]
    
    # Use TF-IDF as BM25 proxy
    all_texts = list(unique_contexts) + list(sampled_q)
    n_ctx = len(unique_contexts)
    
    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english")
    tfidf_all = vectorizer.fit_transform(all_texts)
    
    ctx_vectors = tfidf_all[:n_ctx]
    q_vectors = tfidf_all[n_ctx:]
    
    # For each query, compute similarity to all contexts
    sims = cosine_similarity(q_vectors, ctx_vectors)
    
    # Find rank of correct context
    ctx_to_idx = {ctx: i for i, ctx in enumerate(unique_contexts)}
    
    ranks = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    for i in range(n_sample):
        correct_idx = ctx_to_idx.get(sampled_correct_ctx[i])
        if correct_idx is None:
            continue
        sorted_indices = np.argsort(-sims[i])
        rank = np.where(sorted_indices == correct_idx)[0][0] + 1
        ranks.append(rank)
        if rank <= 1:
            top1_correct += 1
        if rank <= 3:
            top3_correct += 1
        if rank <= 5:
            top5_correct += 1
    
    ranks = np.array(ranks)
    print(f"  TF-IDF Retrieval (proxy for BM25):")
    print(f"    Recall@1: {top1_correct / len(ranks) * 100:.1f}%")
    print(f"    Recall@3: {top3_correct / len(ranks) * 100:.1f}%")
    print(f"    Recall@5: {top5_correct / len(ranks) * 100:.1f}%")
    print(f"    MRR: {np.mean(1.0 / ranks):.4f}")
    print(f"    Mean rank: {ranks.mean():.1f}")
    print(f"    Median rank: {np.median(ranks):.0f}")
    print(f"    % ranked >10: {(ranks > 10).mean() * 100:.1f}%")
    print(f"    % ranked >50: {(ranks > 50).mean() * 100:.1f}%")
    print(f"    % ranked >100: {(ranks > 100).mean() * 100:.1f}%")

# ============================================================
# B. QUERY-TO-WRONG-CONTEXT SIMILARITY (WHY HARD NEGATIVES CONFUSE)
# ============================================================
print("\n" + "=" * 80)
print("B. HARD NEGATIVE DEEP DIVE: Why wrong contexts score high")
print("=" * 80)

for df, name in [(df_finqa, "FinQA")]:
    unique_contexts = df["context"].dropna().unique()
    questions = df["question"].dropna().values
    correct_ctx = df["context"].dropna().values
    
    np.random.seed(42)
    n_sample = 200
    sample_idx = np.random.choice(len(questions), n_sample, replace=False)
    
    sampled_q = questions[sample_idx]
    sampled_correct_ctx = correct_ctx[sample_idx]
    
    all_texts = list(unique_contexts) + list(sampled_q)
    n_ctx = len(unique_contexts)
    
    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english")
    tfidf_all = vectorizer.fit_transform(all_texts)
    ctx_vectors = tfidf_all[:n_ctx]
    q_vectors = tfidf_all[n_ctx:]
    
    sims = cosine_similarity(q_vectors, ctx_vectors)
    ctx_to_idx = {ctx: i for i, ctx in enumerate(unique_contexts)}
    
    # Analyze gap between correct and top-1 wrong context
    gaps = []
    correct_sims_list = []
    wrong_sims_list = []
    
    confused_examples = []
    
    for i in range(n_sample):
        correct_idx = ctx_to_idx.get(sampled_correct_ctx[i])
        if correct_idx is None:
            continue
        correct_sim = sims[i, correct_idx]
        sorted_indices = np.argsort(-sims[i])
        
        # Find top wrong context
        for idx in sorted_indices:
            if idx != correct_idx:
                wrong_sim = sims[i, idx]
                break
        
        gap = correct_sim - wrong_sim
        gaps.append(gap)
        correct_sims_list.append(correct_sim)
        wrong_sims_list.append(wrong_sim)
        
        # Collect confused examples
        if gap < 0:  # Wrong context scores HIGHER
            confused_examples.append({
                "question": sampled_q[i][:150],
                "correct_sim": correct_sim,
                "wrong_sim": wrong_sim,
                "gap": gap
            })
    
    gaps = np.array(gaps)
    print(f"\n--- {name}: Similarity Gap (correct - top wrong) ---")
    print(f"  Mean gap: {gaps.mean():.4f}")
    print(f"  Median gap: {np.median(gaps):.4f}")
    print(f"  % where wrong > correct (negative gap): {(gaps < 0).mean() * 100:.1f}%")
    print(f"  % where gap < 0.01: {(gaps < 0.01).mean() * 100:.1f}%")
    print(f"  % where gap < 0.05: {(gaps < 0.05).mean() * 100:.1f}%")
    print(f"  Avg correct context sim: {np.mean(correct_sims_list):.4f}")
    print(f"  Avg top wrong context sim: {np.mean(wrong_sims_list):.4f}")
    
    if confused_examples:
        print(f"\n  Top 5 CONFUSED examples (wrong > correct):")
        for ex in sorted(confused_examples, key=lambda x: x["gap"])[:5]:
            print(f"    Q: {ex['question']}")
            print(f"    Correct sim: {ex['correct_sim']:.4f}, Wrong sim: {ex['wrong_sim']:.4f}, Gap: {ex['gap']:.4f}")
            print()

# ============================================================
# C. TABLE vs TEXT: Which part carries retrieval signal?
# ============================================================
print("=" * 80)
print("C. TABLE vs TEXT: Where is the retrieval signal?")
print("=" * 80)

for df, name in [(df_finqa, "FinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n--- {name} ---")
    
    # Separate table and text portions
    def extract_table(ctx):
        lines = str(ctx).split("\n")
        table_lines = [l for l in lines if "|" in l and l.strip().startswith("|")]
        return " ".join(table_lines)
    
    def extract_text(ctx):
        lines = str(ctx).split("\n")
        text_lines = [l for l in lines if not ("|" in l and l.strip().startswith("|"))]
        return " ".join(text_lines)
    
    questions = df["question"].dropna().values
    contexts = df["context"].dropna().values
    
    np.random.seed(42)
    n_sample = min(500, len(questions))
    sample_idx = np.random.choice(len(questions), n_sample, replace=False)
    
    sampled_q = questions[sample_idx]
    sampled_ctx = contexts[sample_idx]
    
    tables = [extract_table(c) for c in sampled_ctx]
    texts = [extract_text(c) for c in sampled_ctx]
    
    # TF-IDF similarity: query vs table-only, query vs text-only
    # Query vs Table
    all_qt = list(tables) + list(sampled_q)
    vec_qt = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_qt = vec_qt.fit_transform(all_qt)
    sim_qt = cosine_similarity(tfidf_qt[n_sample:], tfidf_qt[:n_sample])
    diag_qt = np.diag(sim_qt)
    
    # Query vs Text
    all_qx = list(texts) + list(sampled_q)
    vec_qx = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_qx = vec_qx.fit_transform(all_qx)
    sim_qx = cosine_similarity(tfidf_qx[n_sample:], tfidf_qx[:n_sample])
    diag_qx = np.diag(sim_qx)
    
    print(f"  Query-Table similarity (avg): {diag_qt.mean():.4f}")
    print(f"  Query-Text similarity (avg):  {diag_qx.mean():.4f}")
    print(f"  Ratio (table/text): {diag_qt.mean() / max(diag_qx.mean(), 0.001):.4f}")
    print(f"  % where table > text: {(diag_qt > diag_qx).mean() * 100:.1f}%")
    print(f"  % where table sim < 0.01: {(diag_qt < 0.01).mean() * 100:.1f}%")

# ============================================================
# D. NUMERICAL VALUE OVERLAP: Key to financial retrieval
# ============================================================
print("\n" + "=" * 80)
print("D. NUMERICAL VALUE OVERLAP BETWEEN QUERY AND CONTEXT")
print("=" * 80)

for df, name in [(df_finqa, "FinQA"), (df_tatdqa, "TAT-DQA")]:
    print(f"\n--- {name} ---")
    
    def extract_numbers(text):
        nums = re.findall(r'-?\$?\d[\d,.]*%?', str(text))
        # Normalize: remove $, %, commas
        normalized = set()
        for n in nums:
            clean = n.replace("$", "").replace("%", "").replace(",", "").strip(".")
            if clean:
                normalized.add(clean)
        return normalized
    
    questions = df["question"].dropna().values
    contexts = df["context"].dropna().values
    
    q_num_overlaps = []
    q_num_in_q = []
    q_num_in_ctx = []
    
    for q, c in zip(questions, contexts):
        q_nums = extract_numbers(q)
        c_nums = extract_numbers(c)
        
        q_num_in_q.append(len(q_nums))
        q_num_in_ctx.append(len(c_nums))
        
        if len(q_nums) > 0:
            overlap = len(q_nums & c_nums) / len(q_nums)
            q_num_overlaps.append(overlap)
    
    q_num_overlaps = np.array(q_num_overlaps)
    print(f"  Avg numbers in question: {np.mean(q_num_in_q):.2f}")
    print(f"  Avg numbers in context: {np.mean(q_num_in_ctx):.2f}")
    print(f"  Avg numerical overlap (query nums found in context): {q_num_overlaps.mean():.4f}")
    print(f"  Median numerical overlap: {np.median(q_num_overlaps):.4f}")
    print(f"  % with 0% numeric overlap: {(q_num_overlaps == 0).mean() * 100:.1f}%")
    print(f"  % with >50% numeric overlap: {(q_num_overlaps > 0.5).mean() * 100:.1f}%")
    print(f"  % with 100% numeric overlap: {(q_num_overlaps == 1.0).mean() * 100:.1f}%")

# ============================================================
# E. DOCUMENT LENGTH DISTRIBUTION (tokens for embedding models)
# ============================================================
print("\n" + "=" * 80)
print("E. DOCUMENT LENGTH DISTRIBUTION (Token Estimation)")
print("=" * 80)

for df, name in [(df_finqa, "FinQA"), (df_convfinqa, "ConvFinQA"), (df_tatdqa, "TAT-DQA")]:
    ctx = df["context"].dropna().astype(str)
    # Rough token estimate: words * 1.3 (sub-word tokenization)
    word_counts = ctx.str.split().str.len()
    token_est = word_counts * 1.3
    
    print(f"\n--- {name} ---")
    print(f"  Avg estimated tokens: {token_est.mean():.0f}")
    print(f"  Median estimated tokens: {token_est.median():.0f}")
    print(f"  P90 estimated tokens: {token_est.quantile(0.9):.0f}")
    print(f"  P95 estimated tokens: {token_est.quantile(0.95):.0f}")
    print(f"  Max estimated tokens: {token_est.max():.0f}")
    print(f"  % > 512 tokens: {(token_est > 512).mean() * 100:.1f}%")
    print(f"  % > 1024 tokens: {(token_est > 1024).mean() * 100:.1f}%")
    print(f"  % > 2048 tokens: {(token_est > 2048).mean() * 100:.1f}%")

# ============================================================
# F. SAME-COMPANY CONFUSION: Inter-company vs Intra-company similarity
# ============================================================
print("\n" + "=" * 80)
print("F. SAME-COMPANY CONTEXT SIMILARITY (Intra vs Inter)")
print("=" * 80)

for df, name in [(df_finqa, "FinQA")]:
    if "company_name" not in df.columns:
        continue
    
    # Get unique (context, company) pairs
    ctx_company = df[["context", "company_name"]].dropna().drop_duplicates(subset="context")
    
    np.random.seed(42)
    n_sample = min(500, len(ctx_company))
    sampled = ctx_company.sample(n_sample, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf = vectorizer.fit_transform(sampled["context"].values)
    sim_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(sim_matrix, 0)
    
    companies = sampled["company_name"].values
    
    intra_sims = []
    inter_sims = []
    
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            if companies[i] == companies[j]:
                intra_sims.append(sim_matrix[i, j])
            else:
                inter_sims.append(sim_matrix[i, j])
    
    print(f"\n--- {name} ---")
    print(f"  Same-company pairs: {len(intra_sims)}")
    print(f"  Different-company pairs: {len(inter_sims)}")
    if intra_sims:
        print(f"  Avg intra-company similarity: {np.mean(intra_sims):.4f}")
        print(f"  Avg inter-company similarity: {np.mean(inter_sims):.4f}")
        print(f"  Ratio (intra/inter): {np.mean(intra_sims) / max(np.mean(inter_sims), 0.001):.2f}x")
        print(f"  Median intra-company similarity: {np.median(intra_sims):.4f}")
        print(f"  Median inter-company similarity: {np.median(inter_sims):.4f}")

print("\nSUPPLEMENTARY ANALYSIS COMPLETE!")
