# GSR-CACL: Architecture Reference

> **Full title:** Graph-Structured Retrieval + Constraint-Aware Contrastive Learning for Financial Document Retrieval  
> **Benchmark:** T²-RAGBench (EACL 2026) — FinQA · ConvFinQA · TAT-DQA

---

## 0. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            GSR-CACL System                                      │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                 C1: GSR  (Graph-Structured Retrieval)                    │  │
│   │                         INFERENCE  ONLY                                 │  │
│   │                                                                          │  │
│   │     corpus D  ──► KG Construction ──► GAT Encoder ──► Joint Scorer ──►  │  │
│   │     query  Q  ──► BGE Embedding  ───────────────────────────────────►  │  │
│   │                                                              Top-K docs  │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │           C2: CACL  (Constraint-Aware Contrastive Learning)              │  │
│   │                           TRAINING ONLY                                 │  │
│   │                                                                          │  │
│   │     corpus D  ──► CHAP Perturbation ──► Hard Negatives                  │  │
│   │                   (A/S/E types)                                          │  │
│   │     (Q, C⁺, C⁻_CHAP) ──► L_CACL = L_triplet + λ·L_constraint           │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Two independent contributions:**

| | C1: GSR | C2: CACL |
|---|---|---|
| **Phase** | Inference | Training |
| **Input** | (Query, Corpus) | (Query, Positive doc, CHAP Negatives) |
| **Output** | Ranked Top-K documents | Trained JointScorer weights |
| **Key idea** | Accounting KG + ε-tolerance constraint scoring | Constraint-violating hard negatives |
| **Code** | `methods/gsr_retrieval.py` | `training/`, `negative_sampler/` |

---

## 1. Core Architecture: GSR Retrieval Pipeline

### 1.1 Two-phase architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PHASE 1 — OFFLINE INDEXING  (per document D)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Document D                                                                  ║
║  ┌─────────────────────────────────────────┐                                 ║
║  │  page_content = [narrative text]        │                                 ║
║  │  [markdown table]                       │                                 ║
║  │  meta = {company, year, sector}         │                                 ║
║  └──────────────┬──────────────────────────┘                                 ║
║                 │                                                             ║
║        ┌────────▼────────┐   ┌──────────────────────────────────────┐        ║
║        │  extract_table  │   │  BGE Embedding                       │        ║
║        │  (regex + │)    │   │  embed_query(page_content)           │        ║
║        └────────┬────────┘   └──────────────────┬───────────────────┘        ║
║                 │                               │                            ║
║        ┌────────▼────────┐             d_text ∈ ℝ⁷⁶⁸                        ║
║        │  KG Construction│                     │                            ║
║        │  (§2 below)     │                     │                            ║
║        └────────┬────────┘                     │                            ║
║                 │ G_D = (V, E, template)        │                            ║
║        ┌────────▼────────┐                     │                            ║
║        │  GAT Encoder    │                     │                            ║
║        │  (§3 below)     │                     │                            ║
║        └────────┬────────┘                     │      FAISS Index           ║
║                 │ d_KG ∈ ℝ²⁵⁶                  │──────────────────────────► ║
║                 │                               │                            ║
║        ┌────────▼───────────────────────────────▼───────────────────────┐    ║
║        │  CACHE  per document:  (d_text, d_KG, G_D, meta_D)             │    ║
║        └───────────────────────────────────────────────────────────────-┘    ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 2 — ONLINE RETRIEVAL  (per query Q)                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Query Q  +  meta_Q = {company, year, sector}                                ║
║       │                                                                      ║
║       │  BGE embed                                                           ║
║       │  q ∈ ℝ⁷⁶⁸                                                            ║
║       │                                                                      ║
║       ▼                                                                      ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │  FAISS  similarity_search(q, k = 4·top_k)                           │    ║
║  │  → Candidate set C = {D_c₁, D_c₂, ..., D_c_{4K}}                   │    ║
║  └─────────────────────────────────┬────────────────────────────────────┘    ║
║                                    │  for each D_c                           ║
║                    ┌───────────────┴────────────────┐                        ║
║                    │                                │                        ║
║         ┌──────────▼──────────┐      ┌──────────────▼──────────────────┐     ║
║         │  Text Similarity    │      │  From CACHE[D_c]                 │     ║
║         │  s_text =           │      │                                  │     ║
║         │  cos_sim(q, d_text) │      │  ┌──────────────────────────┐    │     ║
║         └──────────┬──────────┘      │  │ Entity Score             │    │     ║
║                    │                 │  │ s_entity = match(        │    │     ║
║                    │                 │  │   company_Q, company_D)  │    │     ║
║                    │                 │  │ + match(year_Q, year_D)  │    │     ║
║                    │                 │  │ + match(sector_Q, sect_D)│    │     ║
║                    │                 │  └──────────────────────────┘    │     ║
║                    │                 │                                  │     ║
║                    │                 │  ┌──────────────────────────┐    │     ║
║                    │                 │  │ Constraint Score         │    │     ║
║                    │                 │  │ CS(G_D) = mean over      │    │     ║
║                    │                 │  │ accounting edges of      │    │     ║
║                    │                 │  │ exp(-residual / max(     │    │     ║
║                    │                 │  │   |v_v|, ε))             │    │     ║
║                    │                 │  └──────────────────────────┘    │     ║
║                    │                 └──────────────┬───────────────────┘     ║
║                    │                                │                        ║
║                    └────────────────────┬───────────┘                        ║
║                                         │                                    ║
║                              ┌──────────▼──────────────────────────┐         ║
║                              │  Joint Score                         │         ║
║                              │  s(Q,D) = α·s_text + β·s_entity     │         ║
║                              │          + γ·CS(G_D)                │         ║
║                              └──────────┬───────────────────────────┘         ║
║                                         │                                    ║
║                              Top-K by joint score                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 1.2 Joint Score (formal definition)

$$s(Q, D) = \alpha \cdot s_{\text{text}}(Q, D) + \beta \cdot s_{\text{entity}}(Q, D) + \gamma \cdot \text{CS}(G_D)$$

| Signal | Formula | Range | Code |
|---|---|---|---|
| **Text similarity** | $s_{\text{text}} = \cos(q, d_{\text{text}})$ | $[-1, 1]$ | `torch.cosine_similarity` |
| **Entity score** | $s_{\text{entity}} = \frac{1}{3}\sum_{k \in \{\text{co,yr,se}\}} \mathbb{1}[m_Q^k = m_D^k]$ | $[0, 1]$ | `_compute_entity_score` |
| **Constraint score** | $\text{CS}(G_D) = \frac{1}{|E_c|}\sum_{(u,v,\omega) \in E_c} \exp\!\left(-\frac{|\omega \cdot v_u - v_v|}{\max(|v_v|, \varepsilon)}\right)$ | $(0, 1]$ | `compute_constraint_score` |

**Learnable weights:** $\alpha = \text{softplus}(\log\hat{\alpha})$, $\beta = \text{softplus}(\log\hat{\beta})$, $\gamma = \text{softplus}(\log\hat{\gamma})$ — always positive.  
**Defaults:** $\alpha = 0.5$, $\beta = 0.3$, $\gamma = 0.2$.

---

## 2. KG Construction Architecture

```
Markdown table string  (page_content)
          │
 ┌────────▼─────────────────────────────────────────────────────┐
 │  parse_markdown_rows(table_md)                               │
 │  → rows[0] = headers  [H₁, H₂, ..., Hₙ]                    │
 │  → rows[1:] = data    cell[r][c] ∈ ℝ or str                 │
 └────────┬─────────────────────────────────────────────────────┘
          │
 ┌────────▼─────────────────────────────────────────────────────┐
 │  Template Matching  (§ below)                                │
 │  canonical_headers = [normalize_header(h) for h in headers] │
 │  template, conf = match_template(canonical_headers)          │
 └────────┬─────────────────────────────────────────────────────┘
          │
          ├──── conf ≥ 0.5 ──────────────────────────────────────┐
          │                                                      │
 ┌────────▼──────────────────────┐    ┌────────────────────────▼─┐
 │  Build nodes                  │    │  Build accounting edges   │
 │  for each cell(r, c):         │    │  for each constraint in   │
 │    v_{r,c} = KGNode(          │    │  template:                │
 │      id = "v_{r}_{c}",        │    │    for each (lhs → rhs):  │
 │      row_idx = r,             │    │      KGEdge(omega = ω,    │
 │      col_idx = c,             │    │        type="accounting") │
 │      value = parse_number(),  │    └────────────────────────┬─┘
 │      header_canonical,        │              │              │
 │      is_total                 │              │ conf < 0.7   │
 │    )                          │              ▼              │
 └────────┬──────────────────────┘    ┌──────────────────────┐ │
          │                           │  Fallback: positional│ │
          │                           │  edges (same-row,    │ │
          │                           │  same-col, ω = 0)    │ │
          │                           └──────────────────────┘ │
          │                                        │            │
          └────────────────────┬───────────────────┘            │
                               │                                │
                    ┌──────────▼──────────────────────────────┐ │
                    │  ConstraintKG = (V, E, template, conf)  │◄┘
                    │  nodes:  {v_{r,c}}                      │
                    │  edges:  accounting ⊕ positional        │
                    │  omega:  +1 (additive), -1 (subtractive)│
                    │          0 (positional)                  │
                    └─────────────────────────────────────────┘
```

### 2.1 Accounting Constraint Edge Semantics

For template constraint `(lhs: [A, B], rhs: C, ω)`:

$$\omega \cdot v_A - v_C = 0 \quad \text{and} \quad \omega \cdot v_B - v_C = 0$$

| ω | Meaning | Example |
|---|---|---|
| `+1` | additive — LHS sums to RHS | Current Assets + Non-Current Assets = Total Assets |
| `-1` | subtractive — RHS minus LHS | Revenue − COGS = Gross Profit |
| `0` | positional — structural proximity | same-row, same-column |

### 2.2 Template Library (15 templates)

```
Template Registry  TEMPLATES: dict[str, AccountingTemplate]
│
├── income_statement          Revenue − COGS = GP; GP − OpEx = EBIT; EBIT − Tax = NI
├── balance_sheet_assets      CurrAssets + NonCurrAssets = TotalAssets
├── balance_sheet_le          CurrLiab + NonCurrLiab = TotalLiab
├── cash_flow                 OCF + ICF + FCF = NetCF
├── revenue_segment           Seg₁ + … + Segₙ = TotalRevenue
├── gross_margin_ratio        GrossProfit / Revenue = GrossMargin
├── yoy_change                Ordered (Year, Value) pairs
├── quarterly_breakdown       Q1 + Q2 + Q3 + Q4 = Annual
├── eps                       NetIncome / SharesOutstanding = EPS
├── debt_schedule             LTDebt + STDebt = TotalDebt
├── shareholder_equity        Common + Preferred + Retained = TotalEquity
├── ebitda                    Revenue − COGS − SG&A + D&A = EBITDA
├── operating_margin          OperatingIncome / Revenue = OperatingMargin
├── net_margin                NetIncome / Revenue = NetProfitMargin
└── current_ratio             CurrentAssets / CurrentLiabilities = CurrentRatio
```

**Matching score:**

$$\text{conf} = \frac{|\{\, h \in \text{headers}_{\text{table}} \mid \text{normalize}(h) \in \text{headers}_{\text{template}} \,\}|}{\max(|\text{headers}_{\text{table}}|,\ |\text{headers}_{\text{template}}|)}$$

---

## 3. GAT Encoder Architecture

```
Input: ConstraintKG  with  V nodes, E edges
                │
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  Node Feature Construction  (per node v = cell at row r, col c)              │
│                                                                              │
│   BGE(cell_text)       →  x_cell  ∈ ℝ⁷⁶⁸   (placeholder: zeros in code)    │
│   SinPE_row(r)         →  x_row   ∈ ℝ¹⁹²   (embed_dim // 4)                │
│   SinPE_col(c)         →  x_col   ∈ ℝ¹⁹²   (embed_dim // 4)                │
│                                                                              │
│   h⁽⁰⁾  =  Linear(1152→256) → LayerNorm → ReLU → Dropout                   │
│         =  InputProj( [x_cell ⊕ x_row ⊕ x_col] )   ∈ ℝ²⁵⁶                 │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  GATLayer 1  (hidden=256, heads=4, head_dim=64)                              │
│                                                                              │
│  For each edge (u, v) with weight ω_{uv}:                                    │
│                                                                              │
│    a_{uv} = (W_q·h_u · W_k·h_v) / √64  +  EdgeProj([ω_{uv}])               │
│    α_{uv} = softmax over i∈N(v) of  a_{ui}      (per head)                  │
│                                                                              │
│    msg_{u→v} = W_v · h_u · α_{uv} · ω_{uv}      (edge-weighted message)    │
│                                                                              │
│    h⁽¹⁾_v  =  W_o( ∑_{u∈N(v)}  msg_{u→v} )  →  LeakyReLU(0.2)             │
│                                                                              │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  h⁽¹⁾ ∈ ℝ^{V×256}
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  GATLayer 2  (identical structure)                                           │
│  h⁽²⁾ ∈ ℝ^{V×256}                                                            │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │
                ▼  Mean Pooling over nodes
           d_KG = mean(h⁽²⁾)  ∈ ℝ²⁵⁶      ← graph-level representation
```

### 3.1 Sinusoidal Positional Encoding

$$\text{PE}(i, 2k) = \sin\!\left(\frac{i}{10000^{2k/d}}\right), \quad \text{PE}(i, 2k+1) = \cos\!\left(\frac{i}{10000^{2k/d}}\right)$$

- Row PE: $d = 768 / 4 = 192$, max_len = 256
- Col PE: $d = 192$, max_len = 64

### 3.2 Edge-Aware Attention (per head $h$)

$$e_{uv}^{(h)} = \frac{\langle W_q^{(h)} \mathbf{h}_u,\; W_k^{(h)} \mathbf{h}_v \rangle}{\sqrt{d_h}} + \text{EdgeProj}_h(\omega_{uv})$$

$$\alpha_{uv}^{(h)} = \frac{\exp(e_{uv}^{(h)})}{\sum_{i \in \mathcal{N}(v)} \exp(e_{iv}^{(h)})}$$

$$\mathbf{h}_v^{(l+1)} = \text{W}_o \left[ \bigoplus_{h=1}^{H} \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(h)} \cdot \omega_{uv} \cdot W_v^{(h)} \mathbf{h}_u^{(l)} \right]$$

---

## 4. Joint Scorer Architecture

```
                     ┌─────────────────────────────────────────┐
                     │            JointScorer  (nn.Module)     │
                     │                                         │
  q_text ∈ ℝ⁷⁶⁸ ───►│   s_text = cos_sim(q_text, d_text)     │
  d_text ∈ ℝ⁷⁶⁸ ───►│          · (0.5 + 0.5·σ(gate(q_text))) │
  d_KG   ∈ ℝ²⁵⁶ ───►│   Linear(768+256 → 64) [text_proj]     │
                     │                                         │
  q_meta ∈ ℝ³   ───►│   s_entity = 1 − tanh(|q_meta−d_meta|) │
  d_meta ∈ ℝ³   ───►│               .mean(−1)                 │
                (co,yr,se)                                     │
                     │                                         │
  G_D (ConstraintKG)►│   CS(G_D) = mean over accounting edges │
                     │            exp(−residual / denom)       │
                     │                                         │
                     │   ┌──────────────────────────────────┐  │
                     │   │  α = softplus(log_α̂)  ← learned │  │
                     │   │  β = softplus(log_β̂)             │  │
                     │   │  γ = softplus(log_γ̂)             │  │
                     │   └──────────────────────────────────┘  │
                     │                                         │
                     │   s(Q,D) = α·s_text + β·s_entity        │
                     │          + γ·CS(G_D)                    │
                     └────────────────────┬────────────────────┘
                                          │
                                    Score ∈ ℝ
```

---

## 5. C2: CACL Training Architecture

### 5.1 Three-stage training overview

```
Stage 1: Identity Pretraining
──────────────────────────────────────────────────────────────
   objective: learn (Company, Year) discrimination
   model:     JointScorer
   loss:      L_triplet = max(0, m − s⁺ + s⁻)
   data:      same-company pairs as positives
              different-company pairs as negatives

Stage 2: Structural Pretraining
──────────────────────────────────────────────────────────────
   objective: KG encoding + constraint scoring calibration
   model:     JointScorer (constraint pathway active)
   loss:      MSE(CS(G_D), 1.0)   [push well-formed KGs to score ≈ 1]

Stage 3: Joint CACL Finetuning  ← core contribution
──────────────────────────────────────────────────────────────
   objective: full contrastive training with CHAP negatives
   model:     JointScorer
   loss:      L_CACL = L_triplet + λ · L_constraint

    for each batch (Q, C⁺, {C⁻_CHAP}):
      G⁺ = build_constraint_kg(C⁺)
      G⁻ = build_constraint_kg(C⁻_CHAP)
      s⁺ = s(Q, C⁺)
      s⁻ = s(Q, C⁻_CHAP)
      loss = L_CACL(s⁺, s⁻, 1[violated])
      loss.backward()
```

### 5.2 CACL Loss

$$\mathcal{L}_{\text{CACL}} = \mathcal{L}_{\text{triplet}} + \lambda \cdot \mathcal{L}_{\text{constraint}}$$

$$\mathcal{L}_{\text{triplet}} = \frac{1}{B} \sum_{i=1}^{B} \max\!\left(0,\; m - s(Q_i, C_i^+) + s(Q_i, C_i^-)\right)$$

$$\mathcal{L}_{\text{constraint}} = -\frac{1}{N} \sum_{j=1}^{N} \mathbb{1}[\text{violates}(C_j^-)] \cdot \log \sigma\!\left(-s(Q, C_j^-)\right)$$

| Symbol | Value |
|---|---|
| $m$ (margin) | 0.2 |
| $\lambda$ | 0.5 |
| Optimizer | AdamW, lr = 5e-5, wd = 0.01 |
| Scheduler | CosineAnnealingLR |

---

## 6. CHAP Negative Sampler Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CHAP: Contrastive Hard-negative via Accounting Perturbations│
│                                                                             │
│  Input: ConstraintKG G_D   →   Output: PerturbedTable (violates KG identity)│
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CHAP-A  (prob = 0.5) — Additive Violation                          │   │
│  │                                                                      │   │
│  │  1. Find leaf node u s.t. u ∈ src of accounting edge, u ∉ tgt       │   │
│  │  2. Perturb: v_u ← v_u · factor,  factor ∈ {1.1, 1.2, 1.3, 0.7...}│   │
│  │  3. Reconstruction: replace cell in markdown table                   │   │
│  │                                                                      │   │
│  │  Before: v_Revenue − v_COGS = v_GrossProfit  ✓                      │   │
│  │  After:  v_Revenue · 1.2 − v_COGS ≠ v_GrossProfit  ✗               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CHAP-S  (prob = 0.3) — Scale Violation                             │   │
│  │                                                                      │   │
│  │  1. Find node with |v| > 1000  (millions/billions range)             │   │
│  │  2. Perturb: v ← v × 0.001   (B → M) or  v × 1000  (M → B)        │   │
│  │                                                                      │   │
│  │  Before: $500M revenue; $200M COGS → GP = $300M  ✓                  │   │
│  │  After:  $0.5M revenue; $200M COGS → GP = $300M  ✗ (scale broken)  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CHAP-E  (prob = 0.2) — Entity Swap                                 │   │
│  │                                                                      │   │
│  │  1. Same table structure, correct numerical identities               │   │
│  │  2. Prepend: [COMPANY: WrongCorp] [YEAR: 2020]                      │   │
│  │                                                                      │   │
│  │  Structure preserved ✓; Entity-query mismatch ✗                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Zero-Sum Property: CHAP negatives  violate  the accounting identity       │
│  ∴  s_entity(C⁻_CHAP) < s_entity(C⁺)  →  valid hard negatives            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. HybridGSR Architecture

```
Query Q
    │
    ├──────────────────────┬──────────────────────┐
    │                      │                      │
    ▼                      ▼                      │
GSRRetrieval.retrieve()   BM25Okapi               │
    │                      │                      │
    │  ranking_GSR          │  ranking_BM25         │
    │  {doc_id : rank}      │  {doc_id : rank}      │
    └──────────────────────┴───────────────────────┘
                           │
                 ┌─────────▼──────────────────────────┐
                 │  Reciprocal Rank Fusion (RRF)       │
                 │                                     │
                 │  rrf(D) =  1/(k + rank_GSR(D))      │
                 │          + 1/(k + rank_BM25(D))     │
                 │                                     │
                 │  k = 60  (RRF constant)             │
                 └───────────────┬─────────────────────┘
                                 │
                         Top-K by rrf score
```

$$\text{rrf}(D) = \frac{1}{k + \text{rank}_{\text{GSR}}(D)} + \frac{1}{k + \text{rank}_{\text{BM25}}(D)}$$

---

## 8. Full Data Flow Summary

```
T²-RAGBench  (HuggingFace: G4KMU/t2-ragbench)
      │  load_t2ragbench_split(config_name, split)
      │  dedup by context_id  →  |corpus| ≪ |QA pairs|
      │
      ▼
DatasetSplit
  ├── queries       : list[str]           — "{company}: {question}"
  ├── ground_truth_ids : list[str]        — context_id of relevant doc
  ├── corpus        : list[Document]      — unique documents
  └── meta_data     : list[dict]          — {company, year, sector}

      │  GSRRetrieval(corpus, embeddings)
      │
      ├── _build_faiss_index()      FAISS[corpus] + d_text[i] ∈ ℝ⁷⁶⁸
      ├── _build_all_kgs()          G_D[i] = ConstraintKG
      ├── _encode_all_kgs()         d_KG[i] = GAT(G_D[i]).mean(0) ∈ ℝ²⁵⁶
      │
      │  .retrieve_batch(queries, meta_data)
      │
      └── for each Q:
            candidates = FAISS.search(q, 4·K)
            for each cand in candidates:
              corpus_idx = _id_to_idx[cand.metadata["id"]]
              s_text     = cos_sim(q, doc_text_embeds[corpus_idx])
              s_entity   = _compute_entity_score(meta_Q, corpus_idx)
              CS         = compute_constraint_score(doc_kgs[corpus_idx])
              score      = α·s_text + β·s_entity + γ·CS
            return sorted(scores)[:K]

      │
      ▼
list[RetrievalResult]
  ├── query
  ├── retrieved_docs    : list[Document]  top-K
  └── ground_truth_id

      │  compute_mrr / compute_recall / compute_ndcg
      ▼
Metrics: MRR@3, Recall@1/3/5, NDCG@3
```

---

## 9. Dimension Reference Table

| Tensor | Shape | Description |
|---|---|---|
| `q`, `d_text` | `[768]` | BGE multilingual-e5-large-instruct embedding |
| `x_cell` | `[V, 768]` | Cell text embedding (input to GAT) |
| `x_row`, `x_col` | `[V, 192]` | Sinusoidal positional encoding (768//4) |
| `h⁽⁰⁾` | `[V, 256]` | After InputProj |
| `h⁽¹⁾`, `h⁽²⁾` | `[V, 256]` | After GATLayer 1, GATLayer 2 |
| `d_KG` | `[256]` | Graph-level mean pooling |
| `edge_index` | `[2, E]` | (src, tgt) indices |
| `edge_weight` | `[E]` | ω ∈ {−1, 0, +1} |
| `meta_feat` | `[3]` | [company_match, year_match, sector_match] |
| `s(Q,D)` | scalar | Joint retrieval score |

---

## 10. Module Dependency Graph

```
gsr_cacl/
│
├── core/               Document, RetrievalResult, DatasetSplit
│     (no deps)
│
├── kg/                 ConstraintKG, KGNode, KGEdge
│   ├── parser          parse_markdown_rows, parse_number
│   ├── builder   ◄─── templates/matching
│   └── data_structures
│
├── templates/          IFRS/GAAP template library
│   ├── data_structures AccountingConstraint, AccountingTemplate
│   ├── library         TEMPLATES dict (15 templates)
│   └── matching        match_template, normalize_header
│
├── encoders/           GAT-based KG encoder
│   ├── positional      SinusoidalPositionalEncoding
│   ├── gat_layer       GATLayer (edge-aware attention)
│   └── gat_encoder ◄── kg/, encoders/positional, encoders/gat_layer
│
├── scoring/            Scoring functions
│   ├── constraint_score compute_constraint_score, compute_entity_score
│   └── joint_scorer ◄── scoring/constraint_score
│
├── negative_sampler/   CHAP perturbation
│   └── chap        ◄── kg/
│
├── training/           CACL training loop
│   ├── data            RetrievalSample, RetrievalDataset
│   ├── losses          TripletLoss, ConstraintViolationLoss, CACLLoss
│   └── trainer     ◄── kg/, negative_sampler/, training/losses
│
├── methods/            Retrieval methods
│   └── gsr_retrieval ◄─ core/, kg/, encoders/, scoring/
│         GSRRetrieval, HybridGSR
│
└── datasets/           Dataset loading
    ├── gsr_document ◄── core/, kg/
    └── wrappers     ◄── core/, datasets/gsr_document
          load_t2ragbench_split, build_gsr_corpus
```
