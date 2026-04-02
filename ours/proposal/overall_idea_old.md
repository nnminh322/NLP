# Structured Knowledge-Enhanced Retrieval for Financial Documents

> **Version:** Final v2 — After Reviewer 4 critique + Dataset format confirmation
> **Philosophy:** 2 contributions. Phân tích xuất sắc + 2 ý tưởng khả thi, củng cố lẫn nhau.
> **Target Venues:** EMNLP / SIGIR
> **Benchmark:** T²-RAGBench (EACL 2026)

---

# PHẦN I: PHÂN TÍCH BẢN CHẤT VẤN ĐỀ
*(Giữ nguyên — reviewer đánh giá xuất sắc, §1.1–§4)*

## 1. Tại sao Standard Retrieval thất bại trên Tài liệu Tài chính?

### 1.1. Bối cảnh Benchmark

T²-RAGBench là benchmark đầu tiên đánh giá RAG trên dữ liệu text+table tài chính trong **unknown-context setting** (91.3% questions validated as context-independent). Mỗi sample bao gồm: `context` (markdown table + narrative text), `table` (extracted markdown), `pre_text`, `post_text`, cùng **metadata**: `company_name`, `report_year`, `company_sector`.

**Khoảng cách lớn nhất:** Hybrid BM25 đạt MRR@3 = 35.2% — kém Oracle Context **30 điểm phần trăm**. Lý do: standard dense retrievers được thiết kế cho văn bản tự nhiên, nhưng tài liệu tài chính là hệ thống mã hóa tuân theo **luật ngôn ngữ đặc thù + ràng buộc toán học bất biến**.

### 1.2. Bốn hiện tượng cốt lõi

**Hiện tượng 1: Lexical Overlap Illusion.** Intra-company similarity gấp **2.82×** inter-company. Trong **59.5% queries**, context sai có điểm similarity cao hơn context đúng.

**Hiện tượng 2: Mathematical Inconsistency.** Tài liệu tài chính bị ràng buộc bởi các phương trình kế toán ẩn: $\text{Revenue} - \text{COGS} = \text{Gross Profit}$, $\text{Assets} = \text{Liabilities} + \text{Equity}$. Flattening phá hủy quan hệ cha-con giữa các ô số.

**Hiện tượng 3: Semantic Shock.** Trong 48.8% cases TAT-DQA, query match tốt với phần text narrative nhưng câu trả lời nằm trong table.

**Hiện tượng 4: Numerical Density Paradox.** ~95% queries có tất cả con số trong context đúng. Nhưng cùng con số xuất hiện trong hàng chục contexts — số vừa là manh mối vừa là bẫy.

### 1.3. Bốn định luật ngôn ngữ tài chính

**Law of Formulaic Encoding** (Halliday, 1994): Ngữ nghĩa không nằm ở từ ngữ, mà ở **vị trí trong cấu trúc**. Mode (mixed text+tabular) bị phá hủy khi flatten.

**Law of Lexical Compression** (Shannon, 1948): Entropy/token trong specialized register thấp hơn general text — từ chuyên ngành có xác suất cố định (Zipf's Law deviation).

**Law of Numerical Anchoring:** Con số là điểm cố định — thay đổi số neo → ngữ nghĩa bị hủy bỏ.

**Law of Multi-layered Semantics:** Một context chứa trung bình 3.16 câu hỏi → biểu diễn phải đa diện.

### 1.4. Ba giả thuyết nghiên cứu

**H1 (Consistency):** Retrieval vượt trội khi bảo toàn tính tự hợp toán học.
**H2 (Structural Alignment):** Retrieval tỉ lệ thuận với căn chỉnh Textual + Structural + Numerical spaces.
**H3 (Hard Negative Validity):** Phân biệt chỉ thực sự được kiểm chứng qua mẫu nhiễu Constraint-Directed.

### 1.5. Lý thuyết Thông tin

**Định lý 1 (Information Loss Bound):**

$$H(T(C)) = H_T(T) + H_S(T) + H_N(T)$$

Flattening gây mất mát:

$$I_{loss} \geq I_S + I_{discriminative}(H_N)$$

→ Multi-space representation là **necessary**.

---

# PHẦN II: 2 CONTRIBUTIONS

## Thiết kế Paper: Một paper với đúng 2 contributions

| Contribution | Giải quyết | Novelty |
|-------------|-----------|---------|
| **C1: GSR** — Graph-Structured Retrieval | Hiện tượng 1 + 2 | Knowledge Graph dựa trên accounting identities — chưa có trong retrieval |
| **C2: CACL** — Constraint-Aware Contrastive Learning | Hiện tượng 2 + 4 + H3 | Accounting perturbation negatives — khác ConFIT ở cấp độ constraint structure |

---

## Contribution 1: GSR — Graph-Structured Retrieval

### 2.1. Dataset Format: Điểm then chốt

**T²-RAGBench cung cấp context ở dạng đã parsed sẵn:**

```
context         = pre_text + table + post_text
table          = extracted markdown table (| headers | separators | rows |)
metadata       = {company_name, report_year, company_sector, ...}
```

**Điều này thay đổi hoàn toàn thiết kế GSR:**

1. **Không cần table detection** — `table` column đã được extract sẵn
2. **Không cần markdown parsing** — table đã ở định dạng structured (headers, rows, cols)
3. **Metadata có sẵn** — `company_name`, `report_year`, `company_sector` → entity identity đã có

**Hệ quả:** GSR có thể tập trung vào **KG construction + constraint validation**, không tốn tài nguyên cho preprocessing.

### 2.2. Template Coverage Analysis

**Khảo sát sơ bộ trên cấu trúc dataset:**

Từ EDA, bảng tài chính trong T²-RAGBench tuân theo **IFRS/GAAP templates**:

| Dataset | Avg rows | Avg cols | Dominant patterns |
|---------|----------|---------|-----------------|
| FinQA | 5.4 | 4.9 | Income Statement, Revenue Breakdown |
| ConvFinQA | 6.4 | 3.9 | Sequential Financial Data |
| TAT-DQA | 15.8 | 3.6 | Multi-dimensional Tables |

**Template Library đề xuất** (15 templates, cover ~80-90% tables):

```
1. Income Statement: Revenue → COGS → Gross_Profit → OpEx → EBIT → EBT → Net_Income
2. Balance Sheet: Current_Assets + Non_Current_Assets = Total_Assets;
                  Assets = Liabilities + Equity
3. Cash Flow: Operating_CF + Investing_CF + Financing_CF = Net_CF
4. Revenue by Segment: Segment_1 + Segment_2 + ... = Total_Revenue
5. Ratio Analysis: Gross_Margin = Gross_Profit / Revenue
6. YoY Change: [Year_N, Value] pairs, ordered
7. Quarterly Breakdown: Q1 + Q2 + Q3 + Q4 = Annual
8. Per-share Metrics: Net_Income / Shares_Outstanding = EPS
9. Debt Schedule: LT_Debt + ST_Debt = Total_Debt
10. Shareholder Equity: Common + Preferred + Retained_Earnings = Total_Equity
```

**Coverage estimation:**
- FinQA (S&P 500 reports): ~90% match template 1-2 (standard financial statements)
- ConvFinQA: ~85% match template 1 + 6 (Income Statement + YoY)
- TAT-DQA: ~70% match template 1-5 (diverse sectors, more complex tables)

**Fallback strategy:** Với ~10-30% tables không match templates → dùng **positional graph** (row/column adjacency edges) thay vì constraint edges.

### 2.3. Core Insight

**Tài liệu tài chính có cấu trúc graph NGẦM — các ô số được nối bởi accounting identities. Knowledge Graph là ngôn ngữ tự nhiên để biểu diễn cấu trúc này.**

So với HELIOS/THYME/THoRR:
- Họ hiểu table để **trả lời câu hỏi** (table QA)
- **GSR xây dựng KG từ accounting constraints → phục vụ retrieval**

### 2.4. Kiến trúc GSR

```
Query Q
  │
  ├─► Metadata extraction: (company, year) từ metadata
  │
  ├─► Table KG Construction:
  │     table_md ──regex──► Cell nodes + Header mapping
  │                     │
  │                     └──► Template Matching
  │                           │
  │                           └──► Constraint edges (ω ∈ {+1, -1})
  │                                 │
  │                                 └──► Partial KG (fallback: positional graph)
  │
  └─► GAT Encoder: node embeddings + edge-aware aggregation
          │
          ▼
  Joint Scoring:
    s(Q, D) = α·sim_text(Q, D)
             + β·sim_entity(Q, G_D)
             + γ·ConstraintScore(G_D, Q)
          │
          ▼
  Top-k Retrieval
```

### 2.5. Component 1a: Template-Based KG Construction

**Pseudo-code:**

```python
def build_constraint_kg(table_md, headers, cell_values):
    # Step 1: Parse markdown table
    rows = parse_markdown_rows(table_md)  # [[cell, cell, ...], ...]
    header_row = rows[0]  # First row = headers

    # Step 2: Template matching
    template = match_template(headers)  # → Income Statement / Balance Sheet / ...

    # Step 3: Build nodes
    nodes = []
    for row_idx, row in enumerate(rows[1:]):  # skip header
        for col_idx, cell in enumerate(row):
            node = {
                'id': f'v_{row_idx}_{col_idx}',
                'value': parse_number(cell),
                'header': header_row[col_idx],
                'row_idx': row_idx,
                'col_idx': col_idx,
            }
            nodes.append(node)

    # Step 4: Build constraint edges from template
    edges = []
    for constraint in template.constraints:
        src_nodes = match_nodes(constraint.lhs, nodes)
        tgt_nodes = match_nodes(constraint.rhs, nodes)
        for src in src_nodes:
            for tgt in tgt_nodes:
                edges.append({
                    'src': src['id'],
                    'tgt': tgt['id'],
                    'omega': constraint.omega,  # +1 or -1
                    'type': 'accounting',
                    'constraint': constraint.name,
                })

    # Step 5: Positional fallback edges (if template match confidence < threshold)
    if template.confidence < 0.7:
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if same_row(node_i, node_j):
                    edges.append({'src': node_i['id'], 'tgt': node_j['id'], 'omega': 0, 'type': 'positional'})
                if same_col(node_i, node_j):
                    edges.append({'src': node_i['id'], 'tgt': node_j['id'], 'omega': 0, 'type': 'positional'})

    return {'nodes': nodes, 'edges': edges, 'template': template.name}
```

### 2.6. Component 1b: GAT Encoder

**Node embedding:**
$$h_v^{(0)} = \text{BGE}(cell\_text) \oplus \text{PosEnc}(row\_idx) \oplus \text{PosEnc}(col\_idx)$$

**Edge-aware message passing:**
$$h_v^{(l+1)} = \text{LeakyReLU}\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} \cdot \omega_{vu} \cdot W^{(l)} \cdot h_u^{(l)}\right)$$

Với $\omega_{vu} \in \{+1, -1, 0\}$:
- $\omega = +1$: additive constraint (Revenue + COGS → Gross Profit)
- $\omega = -1$: subtractive constraint (Revenue − COGS → Gross Profit)
- $\omega = 0$: positional edge (row/column adjacency)

### 2.7. Component 1c: Constraint-Aware Scoring

**Constraint Score** (ε-tolerance, differentiable):
$$\text{CS}(G, Q) = \frac{1}{|\mathcal{E}_c|} \sum_{(u,v) \in \mathcal{E}_c} \exp\left(-\frac{|\omega_{uv} \cdot v_u - v_v|}{\max(|v_v|, \epsilon)}\right)$$

**Entity Matching Score:**
$$s_{entity}(Q, G_D) = \text{sim}(\text{MD}(company_Q, year_Q), \text{MD}(company_D, year_D))$$

**Total Score:**
$$s(Q, D) = \alpha \cdot \text{sim}(q, e_D) + \beta \cdot s_{entity} + \gamma \cdot \text{CS}(G_D, Q)$$

### 2.8. Comparison với HELIOS/THYME/THoRR

| Method | Table Representation | Accounting Constraints | Purpose | GSR difference |
|--------|---------------------|-----------------------|---------|---------------|
| HELIOS (ACL 2025) | Bipartite subgraph (table ↔ text) | ❌ Không | Multi-hop QA | GSR dùng **accounting edges**, không phải table-text edges |
| THYME (EMNLP 2025) | Field-aware matching (header, body, caption) | ❌ Không | Field retrieval | GSR hiểu **field relationships** (qua constraints), không chỉ field types |
| THoRR (2024) | Header concatenation + refinement | ❌ Không | Two-stage retrieval | GSR xây dựng **full KG**, không chỉ headers |
| Singh & Gupta (NAACL 2023) | Constraint-guided KG | ✅ Có | Relation extraction | GSR dùng KG cho **retrieval**, không phải extraction |
| **GSR (Ours)** | **Constraint KG + GAT** | **✅ Có** | **Retrieval** | **Accounting identities làm primary retrieval signal** |

**Positioning:** *"While all prior methods use table structure for QA or field matching, GSR is the first to model accounting identities as a retrieval signal — treating constraint satisfaction as evidence of relevance."*

---

## Contribution 2: CACL — Constraint-Aware Contrastive Learning

### 3.1. Core Insight

**CHAP negatives (Contrastive Hard-negative viaphones Accounting Perturbations) khác với ConFIT's SPP (Semantic-Preserving Perturbation) ở cấp độ constraint structure:**

| Aspect | ConFIT SPP | CHAP Negatives |
|--------|-----------|---------------|
| Knowledge source | Loughran-McDonald lexicon + Wikidata | **Accounting identities** (GAAP/IFRS) |
| Perturbation type | Perplexity + NLI filtering | **Constraint-directed violation** |
| Target task | Extraction | **Retrieval** |
| Negative quality signal | Semantic similarity | **Constraint violation** |

**CHAP tập trung vào identity violations có cấu trúc rõ ràng** (thay đổi 1 số → vi phạm equation cụ thể). ConFIT dùng lexicon mềm hơn. Đây là sự khác biệt then chốt.

### 3.2. CHAP Negative Generation Protocol

**3 loại negatives, đều tuân theo Zero-Sum property:**

| Type | Generation | Constraint Status | Example |
|------|-----------|-----------------|---------|
| **CHAP-A** | Thay đổi 1 cell, recompute parent sums | $A + B \neq Total$ | Revenue=100M → 120M (Gross Profit broken) |
| **CHAP-S** | Đổi unit (M → B), giữ parent | Scale mismatch | Revenue 500M → 0.5B (ratio broken) |
| **CHAP-E** | Đổi year/entity, giữ structure | Temporal/entity mismatch | Apple/2023 → Apple/2022 |

**Zero-Sum property:** Thay đổi đúng 1 thành phần, constraint structure giữ nguyên → negatives "giống đúng" nhưng guaranteed to be negative.

### 3.3. CACL Training Objective

$$\mathcal{L} = \mathcal{L}_{triplet} + \lambda \cdot \mathcal{L}_{constraint}$$

**Triplet Loss:**
$$\mathcal{L}_{triplet} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, m - s(Q_i, C_i^+) + s(Q_i, C_i^-)\right)$$

**Constraint Violation Loss:**
$$\mathcal{L}_{constraint} = -\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{violates}(C_i^-, G_{C_i^-})] \cdot \log \sigma(s(Q_i, C_i^-))$$

**Intuition:** Model không chỉ được penalize khi negatives scored cao — mà còn bị penalize nặng hơn khi negatives **vi phạm constraints** (dù có thể lexically gần với positive).

### 3.4. GSR + CACL Joint Training

```
Training Loop:
  for (Q, C⁺, C⁻_CHAP) in batch:
    G⁺ = build_constraint_kg(C⁺)     # GSR KG Construction
    G⁻ = build_constraint_kg(C⁻_CHAP)

    # Forward pass
    s⁺ = scoring(Q, C⁺, G⁺)     # GSR scoring
    s⁻ = scoring(Q, C⁻_CHAP, G⁻)

    # Loss
    loss = triplet_loss(s⁺, s⁻) + λ · constraint_loss(s⁻, G⁻)

    # Backward
    loss.backward()
```

GSR cung cấp graph structure cho CACL học từ, và CACL cung cấp training signal để GSR học constraint semantics.

---

## 4. Experiments

### 4.1. Baselines

| Baseline | Mô tả | Priority |
|----------|--------|---------|
| **HELIOS (ACL 2025)** | Multi-granular table-text retrieval | 🔴 Critical |
| **THYME (EMNLP 2025)** | Field-aware hybrid matching | 🔴 Critical |
| **THoRR (2024)** | Two-stage table retrieval | 🔴 Critical |
| **ColBERT-v2** | Late interaction baseline | 🔴 Critical (cho MR-Fusion comparison) |
| **BM25** | Lexical baseline | ✅ |
| **E5-Mistral-7B** | Dense retrieval SOTA open | ✅ |
| **Hybrid BM25** | T²-RAGBench best reported | ✅ |
| **ConFIT (Agents4Science 2025)** | Semantic-Preserving Perturbation | ✅ (so sánh CHAP) |

### 4.2. Ablation Studies

| Model | Mô tả | Mục đích |
|-------|--------|----------|
| GSR − KG | Không có Constraint KG | KG contribution |
| GSR − Constraint | KG không có constraint score | Constraint scoring contribution |
| CACL − CHAP | Random negatives thay vì CHAP | H3 validation |
| CACL − L_constraint | Chỉ triplet loss | Constraint regularization |
| GSR + CACL | Full system | Main result |

### 4.3. CHAP Negative Evaluation

Đo accuracy trên 3 loại negatives:
- **Standard:** Random in-batch
- **BM25:** Top-k from BM25
- **CHAP:** Constraint-violating negatives

**H3 prediction:** GSR + CACL đạt accuracy tương đương trên cả 3 types — model đã học constraint semantics.

### 4.4. Cross-Domain Experiment

Train trên FinQA + ConvFinQA, test trên **TAT-DQA** (diverse sectors, khác biệt structure). Điều này đánh giá generalization của template-based approach.

---

## 5. Computational Cost

| Component | Complexity | Per-document | Notes |
|-----------|-----------|-------------|-------|
| KG Construction | $O(n_{cells})$ | ~57 cells (TAT-DQA avg) | Template matching = regex → fast |
| GAT Encoding | $O(V \cdot E)$ | $V \approx 57$, $E \approx 10$ | Sparse attention, 2 layers |
| Entity Matching | $O(1)$ | Constant | Metadata lookup |
| Constraint Scoring | $O(E_c)$ | $E_c \approx 5-10$ | Simple arithmetic |
| **Total inference** | | **~1.2-1.4× baseline** | |
| CHAP Gen (offline) | $O(n_{cells})$ | Pre-generate + cache | Zero runtime cost |

**Pre-indexing strategy:** KG được xây dựng offline cho mọi documents trong corpus → inference chỉ cần lookup và encode queries.

---

## 6. Paper Structure (EMNLP/SIGIR)

```
1. Introduction (1 page)
   - Problem: 4 phenomena + information loss from flattening
   - Contributions: C1: GSR + C2: CACL

2. Background & Related Work (1.5 pages)
   - T²-RAGBench benchmark
   - Table-aware retrieval: HELIOS, THYME, THoRR, TableFormer, TAPAS
   - Entity-conditioned retrieval: C-RECT, FiD-Light
   - Hard negative mining: ConFIT (Agents4Science 2025)
   - **Gap: No prior work uses accounting identities as retrieval signal**

3. Problem Analysis (1 page)
   - 4 phenomena + 4 linguistic laws (Halliday, Shannon)
   - Information-theoretic justification
   - 3 research hypotheses

4. GSR: Graph-Structured Retrieval — Contribution 1 (1.5 pages)
   - 4.1: Dataset format + Template Library coverage analysis
   - 4.2: KG Construction Algorithm (pseudo-code)
   - 4.3: GAT Encoder
   - 4.4: Constraint-Aware Scoring
   - 4.5: Comparison với HELIOS/THYME/THoRR

5. CACL: Constraint-Aware Contrastive Learning — Contribution 2 (1.5 pages)
   - 5.1: CHAP Negative Generation (3 types)
   - 5.2: CACL Training Objective
   - 5.3: Differentiation from ConFIT SPP
   - 5.4: Joint GSR + CACL Training

6. Experiments (2 pages)
   - 6.1: Setup + Baselines (8 methods)
   - 6.2: Main results
   - 6.3: Ablation studies
   - 6.4: CHAP negative evaluation
   - 6.5: Cross-domain validation
   - 6.6: Error analysis

7. Conclusion & Future Work (0.5 page)
```

---

## PHẦN III: PHẢN BIỆN VỚI REVIEWER 4

### Điểm đã giải quyết

| Critique | Hành động |
|----------|-----------|
| "Template coverage chưa khảo sát" | ✅ Bổ sung §2.2: Template Library + coverage estimation (80-90%) + fallback strategy |
| "ConFIT đã làm tương tự" | ✅ §3.1: Rõ ràng so sánh CHAP vs ConFIT SPP (constraint vs lexicon, retrieval vs extraction) |
| "Thiếu cross-domain experiment" | ✅ §4.4: Train FinQA+ConvFinQA → test TAT-DQA |
| "Computational cost chưa cụ thể" | ✅ §5: Bảng chi tiết O-notation |
| "HELIOS comparison chưa rõ" | ✅ §2.8: Comparison table + positioning statement |
| "Thiếu pseudo-code" | ✅ §2.5: Pseudo-code đầy đủ cho KG construction |

### Điểm còn lại — được ghi nhận nhưng cần experiments để chứng minh

1. **Template coverage 80-90%** là estimate — cần chạy khảo sát thực tế
2. **CHAP vs ConFIT differentiation** rõ ràng về mặt lý thuyết — cần so sánh thực nghiệm
3. **Cross-domain generalization** phụ thuộc vào template coverage ngoài financial domain

---

## PHỤ LỤC: Key References

### Graph-based Retrieval
- HELIOS (ACL 2025): Multi-granular table-text retrieval
- THYME (EMNLP 2025): Field-aware hybrid matching
- THoRR (2024): Table header retrieval + refinement
- Singh & Gupta (NAACL 2023): Constraint-guided accounting KG

### Hard Negative Mining
- ConFIT (Agents4Science 2025): Semantic-Preserving Perturbation for financial extraction
- DPR (Karpukhin et al., 2020): Dense passage retrieval
- ColBERT v2 (Khattab & Zaharia, 2020): Late interaction

### Language Theory
- Halliday (1994): An Introduction to Functional Grammar
- Shannon (1948): A Mathematical Theory of Communication
- Cover & Thomas (2006): Elements of Information Theory

### Benchmark
- Strich et al. (EACL 2026): T²-RAGBench
- Chen et al. (EMNLP 2021): FinQA
- Chen et al. (EMNLP 2022): ConvFinQA
- Zhu et al. (NAACL 2022): TAT-DQA
