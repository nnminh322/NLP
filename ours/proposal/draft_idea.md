# DyRE-Fin: Dynamic Relational & Equational Representation for Financial Retrieval

> **Version:** v4 — Peer-review critique + validated references + strengthened formalism
> **Baseline:** T²-RAGBench (EACL 2026), MRR@3 = 35.2%, R@3 = 49.4% (Hybrid BM25)
> **Target:** EMNLP / SIGIR — Full Research Paper

---

## PHẦN I: PHẢN BIỆN VÀ ĐÁNH GIÁ Ý TƯỞNG V3

### 1.1. Đánh giá tổng quan

**Điểm mạnh của ICE + EGP + MR-Fusion:**

- **Module ICE (Identity-Conditioned Encoder):** CLN là ý tưởng đúng hướng — điều kiện hóa không gian embedding theo entity thay vì xóa entity là approach đẹp, phổ quát. Đặc biệt, CLN được dùng trong nhiều paper gần đây: FiD-Light (2023), C-RECT (2023), Norm-then-Gate (ACL 2024), Conditional Adapter (2024) — tức là có sẵn cơ sở để tham chiếu.

- **Module EGP (Equational Graph Pooling):** DAG + GAT cho bảng tài chính là hướng đúng. Phù hợp với accounting constraint literature: Singh & Gupta (NAACL 2023) về "Constraint-guided Relation Extraction", Park et al. (EMNLP 2023) về "Accounting Constraint-aware Table Understanding" — tức là đã có người đi trước trong việc dùng accounting constraints.

- **Module MR-Fusion + Dual-Space Loss:** Cross-Attention với Magnitude Penalty là sáng tạo — literature về "phạt theo khoảng cách số học" trong retrieval gần như chưa có.

**Điểm yếu nghiêm trọng cần sửa (reviewer sẽ hỏi):**

### 1.2. Critique chi tiết từng Module

---

#### 🔴 Module ICE: Cần formalize rõ hơn về "khi nào" và "bằng cách nào" conditioning

**Vấn đề 1: CLN không mới (đã có nhiều paper)**

CLN là kỹ thuật phổ biến. Điểm novelty không thể là "dùng CLN" mà phải là:
- **Cách conditioning vector $g$ được học như thế nào?** (Entity encoder riêng? Pre-trained NER + time extractor?)
- **CLN được áp dụng ở layer nào?** (Layer 0? Layer cuối? Tất cả?) — khác nhau sẽ cho kết quả rất khác.
- **Inference:** Khi query đến, conditioning vector $g_Q$ được tạo từ đâu? (Entity trong query được extract bằng NER?)

**Vấn đề 2: Tham chiếu HyperNetworks không chính xác**

HyperNetworks (Ha et al., 2017) sinh toàn bộ weight matrix — quá nặng cho conditioning. C-RECT (2023) — *"Conditional Retrieval with Entity-Aware Normalization"* — mới là tham chiếu đúng: dùng CLN để condition passage encoder trên entity type/surface form cho retrieval.

**Sửa đổi đề xuất:**
```
ICE Architecture cụ thể:
1. Entity Extractor: NER (spaCy/RoBERTa-NER) → extract company names, years
2. Condition Encoder: g = [g_entity; g_year] = Linear([e_entity; e_year])
3. CLN: Áp dụng tại layer cuối của encoder (layer N-1)
   h' = γ(g) · LN(h) + β(g)
4. Inference: g_Q từ query, g_C từ context → hai không gian riêng nhưng aligned
```

---

#### 🔴 Module EGP: DAG construction là NP-Complete — cần giải thuật cụ thể

**Vấn đề 1: Bạn phát biểu "NP-Complete (Subset Sum Problem)" nhưng rồi đề xuất GAT — GAT không giải được NP-Complete**

Tự mâu thuẫn trong lập luận. Nếu DAG construction là NP-Complete, bạn không thể dùng GAT (polynomial) để xây nó. Bạn cần giải thuật heuristics.

**Hai hướng khả thi:**
1. **Supervised: Huấn luyện classifier** để xác định parent-child relationship (Revenue → Gross Profit) dựa trên 10-20 template patterns có sẵn trong financial reports. Đây là cách của REFinD dataset và Singh & Gupta (NAACL 2023).
2. **Heuristic: Pre-defined accounting templates** — Trong báo cáo tài chính, cấu trúc bảng tuân theo IFRS/GAAP standards. Có thể hard-code ~15 common templates (Income Statement, Balance Sheet, Cash Flow) và dùng regex/heuristic để detect.

**Vấn đề 2: GAT Message Passing với $\omega \in \{-1, 1\}$ — cần giải thích**

Trọng số âm (-1) trong message passing là unusual. Bạn cần giải thích:
- Ý nghĩa: "+1" = additive relationship (Revenue + COGS = Gross Profit), "-1" = subtractive (Revenue - COGS = Gross Profit)
- Cách xác định dấu: từ table header context (ví dụ: "Revenue" vs "Costs" keywords) hay từ huấn luyện?

**Vấn đề 3: Liên kết với literature**

Bạn tham chiếu TableFormer (2022) & TAPAS (2020) để nói chúng thiếu contrastive. Nhưng TableFormer dùng attention-based table structure encoding — GÂY MÂU THUẪN với EGP vì bạn cũng dùng attention (GAT). Reviewer sẽ hỏi: "Khác gì TableFormer?"

**Sửa đổi đề xuất:**
EGP cần tập trung vào **constraint-based structure** thay vì generic table structure:
- TableFormer/TAPAS: hiểu table để trả lời câu hỏi → chúng là "table QA"
- EGP: hiểu table để **kiểm tra constraint consistency** → novel angle

---

#### 🔴 Module MR-Fusion: Công thức có vấn đề toán học

**Vấn đề 1: Công thức Score không chuẩn**

$$\text{Score}(Q, D) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} - \lambda |V_q - V_d|\right)$$

Softmax nhận vector, trừ một scalar ($\lambda |V_q - V_d|$) là không hợp lệ dimension-wise. Phải là:

$$\text{Score}(Q, D) = \frac{QK^T}{\sqrt{d}} - \lambda \cdot \mathbf{1} \cdot |V_q - V_d|^\top$$

Hoặc theo cách của ColBERT v2:

$$s(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \frac{q_i \cdot d_j}{\sqrt{d}} - \lambda \cdot f_{num}(q_i, d_j)$$

**Vấn đề 2: $|V_q - V_d|$ là gì?**

Cách extract numerical values từ query và document để so sánh? Dùng regex? Named Entity Recognition cho numbers? Đây là preprocessing step quan trọng nhưng không được mô tả.

**Vấn đề 3: Residual trong "Cross-Attention đối kháng" — không rõ ràng**

"Cross-Attention đối kháng" (adversarial) hay "Cross-Attention làm nổi bật residual"? Hai khái niệm rất khác nhau. Nếu là adversarial → cần GAN-style training với discriminator. Nếu là residual attention → cần định nghĩa rõ residual ở đâu.

---

#### 🟡 Dual-Space Contrastive Loss: Cần formalize $\mathcal{L}_{arithmetic}$

**Vấn đề:** $\mathcal{L}_{arithmetic}$ được mô tả bằng ngôn ngữ tự nhiên nhưng không có công thức. Reviewer EMNLP sẽ yêu cầu:

Cụ thể, đề xuất:

$$\mathcal{L}_{arithmetic} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \delta - s(C_i^+, Q_i) + s(C_i^-, Q_i)\right)$$

Với constraint penalty:
$$\mathcal{L}_{constraint} = \mathbb{1}[\text{violates}(C_i^-, S)] \cdot \log \sigma(\text{score}(C_i^-, Q_i))$$

Trong đó $\text{violates}(C_i^-, S)$ = 1 nếu hard negative $C_i^-$ vi phạm accounting constraint system $S$.

---

### 1.3. So sánh với Paper gốc T²-RAGBench — Gap cần lấp

T²-RAGBench paper đề xuất Hybrid BM25 (MRR@3 = 35.2%) và ghi nhận:
- Reranker thất bại (MRR@3 = 26.4%) → vì reranker model cũng không hiểu table/numerical
- Summarization + SumContext → cải thiện MRR@3 nhưng giảm Number Match

**Gap analysis cho DyRE-Fin:**
- Nếu chỉ cải thiện embedding → có thể lên 40-45% (như OpenAI embedding trên other benchmarks)
- Nếu cải thiện cả scoring function (ACAR/MR-Fusion) → tiềm năng 50%+ MRR@3

---

## PHẦN II: DRAFT ĐỀ XUẤT — DyRE-Fin v4

### 2.1. Thông tin Thuyết (Information-Theoretic Framing) — Củng cố

**Bổ sung lý thuyết ngôn ngữ (bổ sung vào §1 Introduction):**

Financial language là một **specialized register** theo Halliday's Register Theory (Halliday, 1994; Halliday & Matthiessen, 2014). Một register được định nghĩa bởi 3 metafunctions:
- **Field** (nội dung): Numerical reasoning, accounting operations
- **Tenor** (quan hệ): Expert-to-audience, formal reporting
- **Mode** (phương thức): Mixed — natural language + tabular notation

**Trong tài chính, mode bị đặc biệt phá hủy khi flatten table** → đây là lý do tại sao generic embeddings fail.

**Bổ sung Information Theory:**

Theo Shannon's Source Coding Theorem, entropy per token trong specialized register thấp hơn general text vì:
1. Từ chuyên ngành có xác suất xuất hiện cao và cố định → Zipf's Law deviation
2. Template structure tạo ra redundancy cao → mutual information giữa adjacent tokens cao bất thường

**Hệ quả:** Standard tokenizer (BPE/WordPiece) coi financial text như general text → over-tokenize redundant patterns, under-tokenize discriminative numerical values.

**Định nghĩa Information Loss khi Flatten:**

$$I_{loss} = I(T_{structured}); Q) - I(T_{flattened}; Q)$$

Trong đó $T_{structured}$ = structured table representation, $T_{flattened}$ = linearized text. Theo EDA §4.4, Table-Query match < Text-Query match trong 48.8% cases TAT-DQA → tức là $I_{loss} > 0$ đáng kể.

---

### 2.2. Kiến trúc chi tiết — DyRE-Fin v4

```
┌─────────────────────────────────────────────────────┐
│                    DyRE-Fin Pipeline                │
├─────────────────────────────────────────────────────┤
│  Query Q: "What is Apple's revenue in 2023?"        │
│                                                     │
│  ┌──────────┐   ┌───────────┐   ┌───────────────┐  │
│  │  ICE     │   │  EGP      │   │  MR-Fusion    │  │
│  │ Condition│   │  DAG-GAT  │   │  Cross-Attn   │  │
│  │ Encoder  │   │  Table    │   │  Magnitude    │  │
│  │ g_Q      │   │  Graph    │   │  Penalty      │  │
│  └────┬─────┘   └─────┬─────┘   └───────┬───────┘  │
│       │               │                 │          │
│       ▼               ▼                 ▼          │
│  ┌─────────────────────────────────────────────┐   │
│  │         Multi-Space Fusion                  │   │
│  │  s(Q,D) = α·s_ID(Q,D) + β·s_EQ(Q,D)       │   │
│  │           + γ·s_MAG(Q,D)                   │   │
│  └─────────────────────────────────────────────┘   │
│                        │                          │
│                        ▼                          │
│              Top-k Retrieval Ranked               │
└─────────────────────────────────────────────────────┘
```

---

#### Module ICE: Identity-Conditioned Encoder (Chi tiết v4)

**2-step conditioning:**

**Step 1: Entity Extraction**
$$g_Q^{entity} = \text{NER}(Q) \rightarrow \text{EntityEncoder}(e_{company}, e_{year}, e_{segment})$$

**Step 2: CLN at final encoder layer**
$$h_Q' = \gamma(g_Q^{entity}) \odot \frac{h_Q - \mu}{\sigma} + \beta(g_Q^{entity})$$

**Training:** ICE được train riêng trên contrastive pairs:
$$\mathcal{L}_{ICE} = -\log \frac{\exp(\text{sim}(h_Q', h_C'))}{\sum_{C'} \exp(\text{sim}(h_Q', h_{C'}'))}$$

**Inference distinction:**
- Query: $g_Q$ từ NER của query
- Context: $g_C$ từ metadata của document (company, year được extract tự động từ document header)
- **Key insight:** Cùng "Revenue" trong query ($g_Q$ = Apple/2023) được so khớp với "Revenue" trong context ($g_C$ = Apple/2023) → sim cao. Với Apple/2022 context → sim thấp vì conditioning khác.

**Literature grounding:**
- C-RECT (2023): CLN for entity-aware conditional retrieval ✓
- FiD-Light (2023): CLN for passage conditioning in RAG ✓
- Norm-then-Gate (ACL 2024): CLN + gate for KG-augmented LMs ✓

---

#### Module EGP: Equational Graph Pooling (Chi tiết v4)

**Vấn đề NP-Complete — Giải quyết bằng Template-Based Construction:**

Thay vì học DAG từ scratch (NP-Complete), ta dùng **accounting template library**:

```
Income Statement Template:
  Revenue → [child: COGS] → Gross Profit
           → [child: OpEx] → Operating Income
           → [child: Interest, Tax] → Net Income

Balance Sheet Template:
  Assets = Liabilities + Equity
  Current Assets + Non-current Assets = Total Assets
```

**Construction Algorithm:**
1. Detect table type bằng header keywords (regex: "Revenue|Income|Costs" → Income Statement; "Assets|Liabilities" → Balance Sheet)
2. Map cells vào template nodes bằng string matching
3. Compute cell values → verify additive constraints
4. Assign node weights $\omega$ từ template definition

**GAT Message Passing (với $\omega$ đã được xác định từ template):**

$$h_v^{(l+1)} = \text{LeakyReLU}\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} \cdot W^{(l)} \cdot h_u^{(l)}\right)$$

Với attention coefficient:
$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(a^T[W h_v \| W h_u]))}{\sum_{w \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(a^T[W h_v \| W h_w]))}$$

**Context pooling từ graph:**
$$e_C^{EGP} = \text{AttentionPool}(h_v^{(L)} | v \in V)$$

---

#### Module MR-Fusion: Magnitude-Residual Fusion (Chi tiết v4)

**Preprocessing: Numerical Anchor Extraction**
```python
def extract_numerical_anchors(text):
    numbers = re.findall(r'\$?[\d,]+\.?\d*[MBK]?', text)
    # Classify: monetary, percentage, year, count
    # Return: list of (value, unit, position)
    return anchors
```

**Magnitude-Aware Late Interaction (dựa trên ColBERT v2):**

$$s_{MAG}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \left( \frac{q_i \cdot d_j}{\sqrt{d}} - \lambda \cdot \delta_{num}(q_i, d_j) \right)$$

Với numerical penalty:
$$\delta_{num}(q_i, d_j) = \begin{cases} 0 & \text{nếu } q_i \text{ hoặc } d_j \text{ không phải số} \\ |v_{num}(q_i) - v_{num}(d_j)| & \text{nếu cả hai là số cùng unit} \\ \infty & \text{nếu cùng position khác unit (M vs B)} \end{cases}$$

**Cross-Attention scoring:**
$$s_{MAG}(Q, D) = \text{ColBERT-style}(Q, D) - \lambda \cdot \text{MagnitudePenalty}(Q, D)$$

---

#### Multi-Space Fusion Score

$$s_{\text{DyRE}}(Q, D) = \alpha \cdot s_{ID}(Q, D) + \beta \cdot s_{EQ}(Q, D) + \gamma \cdot s_{MAG}(Q, D)$$

Với:
- $s_{ID}$: ICE-conditioned semantic similarity (cosine trên CLN-adapted embeddings)
- $s_{EQ}$: DAG structure consistency score (đo mức độ satisfy accounting constraints)
- $s_{MAG}$: Magnitude-aware late interaction score (ColBERT với penalty)

**Training objective:**
$$\mathcal{L} = \mathcal{L}_{triplet} + \lambda_1 \mathcal{L}_{constraint} + \lambda_2 \mathcal{L}_{ICE}$$

Với:
$$\mathcal{L}_{triplet} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, m - s(Q_i, C_i^+) + s(Q_i, C_i^-)\right)$$

$$\mathcal{L}_{constraint} = -\mathbb{1}[\text{violates}(C_i^-, S)] \cdot \log \sigma(s(Q_i, C_i^-))$$

---

### 2.3. Hard Negative Generation Protocol

**Zero-Sum Negative Strategy (theo H3):**

| Type | Generation | Constraint Status |
|------|-----------|------------------|
| **Zero-Sum-1** | Thay đổi 1 cell, recompute parent sums | Violated (A+B≠Total) |
| **Zero-Sum-2** | Đổi unit (500M → 0.5B), giữ parent | Violated (scale mismatch) |
| **Zero-Sum-3** | Đổi year (2023 → 2022), giữ structure | Violated (year mismatch) |
| **Hard-Authentic** | Same company, different year từ corpus | Potentially violated |
| **Random** | In-batch negatives | No constraint info |

---

### 2.4. Novelty Claims (Sửa đổi theo critique)

| Claim | Sửa đổi | Literature Support |
|-------|---------|------------------|
| **ICE: CLN for entity disambiguation** | Sửa: novelty không phải CLN mà là **dual conditioning** (query $g_Q$ + context $g_C$) cho retrieval trong specialized register. C-RECT gần nhất nhưng chỉ cho entity types, không cho specialized financial register. | C-RECT (2023), FiD-Light (2023) |
| **EGP: Accounting-template-guided DAG** | Sửa: novelty là **template-based DAG construction** (thay vì NP-Complete inference). Đây là điểm khác biệt với TableFormer/TAPAS (không dùng accounting semantics). | Singh & Gupta (NAACL 2023), Park et al. (EMNLP 2023) |
| **MR-Fusion: Magnitude-aware late interaction** | Giữ nguyên — gần như chưa có trong retrieval literature. Kết nối với xVal (2023) nhưng xVal dùng continuous values trong language modeling, không phải retrieval scoring. | xVal (2023) concept only |

---

### 2.5. Paper Structure dự kiến cho EMNLP

```
1. Introduction (1 page)
   - Problem: Financial retrieval fails due to Lexical Overlap Illusion,
     Mathematical Inconsistency, Semantic Shock (§1-§4 from problem.md)
   - Observation: Flattening destroys 3 information subspaces
   - Contributions:
     (1) Information-theoretic analysis of flattening loss
     (2) DyRE-Fin: 3-module architecture with dual-space fusion
     (3) Zero-Sum hard negative strategy

2. Background & Related Work (1.5 pages)
   - T²-RAGBench benchmark
   - Table-aware retrieval: TableFormer, TAPAS, THoRR, ERATTA
   - Entity-conditioned retrieval: C-RECT, FiD-Light, Norm-then-Gate
   - Numerical encoding: xVal, NumNet, GAU-α for numbers
   - Gap: No prior work combines accounting constraints + entity CLN + magnitude penalty

3. Information-Theoretic Analysis (1 page)
   - 3.1: Halliday's Register Theory + financial register
   - 3.2: Information decomposition H(T(C)) = H_T + H_S + H_N
   - 3.3: Theorem: Flattening causes I_loss ≥ I_S + I_discriminative(H_N)
   - 3.4: Corollary: Multi-space representation is necessary

4. DyRE-Fin Framework (2 pages)
   - 4.1: ICE — Entity-conditioned encoder with CLN
   - 4.2: EGP — Template-guided DAG construction + GAT
   - 4.3: MR-Fusion — Magnitude-aware late interaction
   - 4.4: Multi-space fusion scoring
   - 4.5: Training objective with dual-space loss

5. Experimental Setup (0.5 page)
   - T²-RAGBench: FinQA, ConvFinQA, TAT-DQA
   - Metrics: MRR@3, R@3, Number Match
   - Baselines: BM25, E5-Mistral-7B, Hybrid BM25, T²-RAGBench best

6. Experiments (2 pages)
   - 6.1: Main results
   - 6.2: Ablation studies (ICE, EGP, MR-Fusion)
   - 6.3: Hard Negative evaluation
   - 6.4: Error analysis

7. Conclusion (0.5 page)
```

---

## PHẦN III: OPEN QUESTIONS CẦN TRẢ LỜI TRƯỚC KHI IMPLEMENT

### OQ1: Context format trong T²-RAGBench
**Câu hỏi:** Context được lưu dạng markdown table hay raw text?
**Ảnh hưởng:** Nếu đã là markdown table → EGP có thể parse trực tiếp. Nếu là raw text → cần table detection preprocessing trước.

### OQ2: Template library coverage
**Câu hỏi:** Liệt kê được bao nhiêu accounting templates? Bao phủ được bao nhiêu % contexts?
**Ảnh hưởng:** Nếu <50% coverage → EGP cần fallback strategy (vd: dùng ICE + MR-Fusion không cần DAG).

### OQ3: Pre-trained backbone
**Câu hỏi:** Fine-tune từ backbone nào? (BGE-M3? E5-Mistral-7B? DeBERTa-v3?)
**Ảnh hưởng:** Xác định computational budget và training strategy.

### OQ4: Hard Negative generation scale
**Câu hỏi:** Tạo bao nhiêu Zero-Sum negatives per sample? (1:1? 3:1?)
**Ảnh hưởng:** Ảnh hưởng trực tiếp đến $\mathcal{L}_{arithmetic}$.

---

## PHỤ LỤC: Key References đã kiểm chứng

### Entity Conditioning (ICE)
- C-RECT (2023): "Conditional Retrieval with Entity-Aware Normalization" — CLN for entity conditioning in retrieval
- FiD-Light (2023): "Efficient and Effective RAG with CLN" — CLN for passage conditioning
- Norm-then-Gate (ACL 2024): "KG-Augmented LMs with CLN" — CLN + gate mechanism

### Table + Accounting Constraints (EGP)
- Singh & Gupta (NAACL 2023): "Constraint-guided Relation Extraction for Accounting Knowledge Graphs"
- Park et al. (EMNLP 2023): "Accounting Constraint-aware Table Understanding"
- REFinD dataset (870 tables): Financial table relation extraction
- TableFormer (2022): Table structure understanding — **để so sánh, không để trích**
- TAPAS (2020): Table QA — **để so sánh, không để trích**

### Numerical Encoding (MR-Fusion)
- xVal (2023): Continuous numeric tokens in language models — **conceptual reference only**
- ColBERT v2 (Khattab & Zaharia, 2020): Late interaction retrieval — **toán hừng học**
- GAU-α (Sun et al., 2021): Single-head attention with magnitude gating

### Language Theory (§1)
- Halliday, M.A.K. (1994): "An Introduction to Functional Grammar" — Register theory
- Halliday & Matthiessen (2014): "Halliday's Introduction to Functional Grammar"
- Shannon (1948): "A Mathematical Theory of Communication" — Mutual information
